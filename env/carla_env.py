import carla
import gymnasium as gym
import numpy as np
import math
import os
import random
import time

CARLA_HOST = os.environ.get("CARLA_HOST", "172.25.176.1")
CARLA_PORT = int(os.environ.get("CARLA_PORT", "2000"))
IMG_W = 128
IMG_H = 128
NUM_SEMANTIC_CLASSES = 28  # CARLA 0.9.15 has 28 semantic classes

# Target speed in m/s (~30 km/h for urban driving)
TARGET_SPEED = 8.33
MAX_SPEED = 14.0  # ~50 km/h hard cap

# Weather presets for domain randomization
WEATHER_PRESETS = [
    carla.WeatherParameters.ClearNoon,
    carla.WeatherParameters.CloudyNoon,
    carla.WeatherParameters.WetNoon,
    carla.WeatherParameters.WetCloudyNoon,
    carla.WeatherParameters.SoftRainNoon,
    carla.WeatherParameters.MidRainSunset,
    carla.WeatherParameters.HardRainNoon,
    carla.WeatherParameters.ClearSunset,
    carla.WeatherParameters.CloudySunset,
    carla.WeatherParameters.SoftRainSunset,
]


class CarlaEnv(gym.Env):
    """
    CARLA Gymnasium environment for urban driving with RL.

    Improvements over naive baseline:
    - GlobalRoutePlanner for navigation (agent knows WHERE to go)
    - Route progress as primary reward signal (CaRL-style)
    - Expanded state vector (12 elements: waypoints, lane offset, traffic light, prev actions)
    - Single-channel normalized semantic segmentation (not 3 identical channels)
    - Action smoothness penalty (CAPS-style)
    - Lane centering penalty
    - Traffic light compliance
    - Domain randomization (weather)
    - Traffic spawning via TrafficManager
    """

    def __init__(self, port=None, enable_traffic=True, enable_weather_randomization=True,
                 max_episode_steps=2000, num_traffic_vehicles=30, num_pedestrians=10):
        super().__init__()

        self.port = port or CARLA_PORT
        self.enable_traffic = enable_traffic
        self.enable_weather_randomization = enable_weather_randomization
        self.max_episode_steps = max_episode_steps
        self.num_traffic_vehicles = num_traffic_vehicles
        self.num_pedestrians = num_pedestrians

        # Observation: single-channel semantic seg (normalized) + state vector.
        # State dim dropped 12 -> 10 per Agent-3 audit: prev_steer and
        # prev_throttle were POMDP-smell (policy state, not environment
        # state) and created a hidden distribution shift between BC training
        # (where prev_action comes from the expert controller) and PPO
        # finetune (where it comes from a Gaussian sample). Frame-stacking
        # on the image already supplies short-term temporal context.
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(low=0.0, high=1.0, shape=(1, IMG_H, IMG_W), dtype=np.float32),
            "state": gym.spaces.Box(low=-1.5, high=1.5, shape=(10,), dtype=np.float32)
        })
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Connect to CARLA
        self.client = carla.Client(CARLA_HOST, self.port)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()

        # Destroy all leftover actors from previous crashed/killed sessions.
        # Must use batch + tick because the server may already be in sync mode
        # from a previous session that didn't clean up properly.
        # Order: stop controllers -> destroy controllers -> destroy sensors ->
        #         destroy walkers -> destroy vehicles.
        leftover_controllers = list(self.world.get_actors().filter("controller.*"))
        for c in leftover_controllers:
            try:
                c.stop()
            except Exception:
                pass
        leftover = (
            leftover_controllers
            + list(self.world.get_actors().filter("sensor.*"))
            + list(self.world.get_actors().filter("walker.*"))
            + list(self.world.get_actors().filter("vehicle.*"))
        )
        if leftover:
            batch = [carla.command.DestroyActor(a) for a in leftover]
            try:
                self.client.apply_batch_sync(batch)
            except Exception:
                for a in leftover:
                    try:
                        a.destroy()
                    except Exception:
                        pass
            # Tick to flush destructions (works whether server is sync or async).
            # seconds=10 guards against an indefinite block on initial-startup
            # cleanup (issue #4 root-cause analysis: bare tick() blocks when the
            # TM race or sync-mode deadlock has happened).
            try:
                self.world.tick(seconds=10.0)
            except Exception:
                pass

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 Hz
        self.world.apply_settings(settings)

        # TrafficManager — created ONCE in __init__, reused across all resets.
        # CARLA issue #2789 documents a hang when `get_trafficmanager` is
        # called after world reloads with a TM previously registered in sync
        # mode; issue #9172 documents a race in 0.9.15's TrafficManagerLocal.cpp
        # that fires preferentially when tick cadence slows (e.g., during PPO's
        # gradient update). Creating the TM once and leaving it attached avoids
        # re-registration across resets and prevents the TM-client/TM-server
        # state from churning under the race window.
        # Each env instance has its own TM port (base port + 6000 + pid offset);
        # with port spacing of 1000 across workers, ranges never overlap.
        self._tm = self.client.get_trafficmanager(
            self.port + 6000 + os.getpid() % 1000
        )
        self._tm.set_global_distance_to_leading_vehicle(2.5)
        self._tm.set_synchronous_mode(True)
        self._tm.set_random_device_seed(random.randint(0, 10000))

        # Route planner from CARLA's PythonAPI (agents.navigation.global_route_planner).
        # Our project also has an 'agents/' directory which can shadow CARLA's.
        # We temporarily manipulate sys.path to ensure CARLA's agents is found.
        self._grp = self._init_route_planner()

        # Actor handles
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        self.traffic_actors = []
        self.pedestrian_actors = []

        # State tracking
        self.image = None
        self.collision_history = []
        self.step_count = 0
        self.route = []
        self.route_index = 0
        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)
        self.total_route_length = 0.0
        self.route_progress = 0.0  # cumulative meters traveled along route

        # Per-worker global step counter, used to anneal the carrot reward.
        # Persists across resets; monotonic across the lifetime of this env
        # instance. Honors URBANZERO_CARROT_DECAY_STEPS env var for the
        # anneal horizon (per worker). With n_envs=4 and default 500k per
        # worker, total carrot-live budget is ~2M global env-steps.
        self._worker_step_counter = 0
        self._carrot_decay_steps = int(
            os.environ.get("URBANZERO_CARROT_DECAY_STEPS",
                           str(self.CARROT_DECAY_STEPS_DEFAULT))
        )
        # Guard against zero-or-negative anneal horizon (would divide by zero
        # in _compute_reward). If the user wants the carrot disabled, they
        # can set the env var to 1 — the anneal will zero out after one step.
        if self._carrot_decay_steps <= 0:
            print(f"[CarlaEnv] URBANZERO_CARROT_DECAY_STEPS={self._carrot_decay_steps} "
                  f"invalid (must be >0); falling back to default "
                  f"{self.CARROT_DECAY_STEPS_DEFAULT}")
            self._carrot_decay_steps = self.CARROT_DECAY_STEPS_DEFAULT

        # Termination-reason emitter: _compute_reward writes the last
        # terminal's reason string here, step() reads it into info.
        self._last_termination_reason = None

        # Goal-loiter counter — zeroed in reset() and whenever the ego
        # leaves the goal zone. Prior code used getattr-with-default which
        # could leak state across episodes per Agent-2 audit.
        self._at_goal_steps = 0

    def reset(self, seed=None, options=None):
        # Retry up to 3 times: black-image timeouts, no-spawn-found, etc.
        # Without retry these escape to SB3 which has no recovery path.
        last_err = None
        for attempt in range(3):
            try:
                return self._reset_once(seed=seed, options=options)
            except RuntimeError as e:
                last_err = e
                print(f"[reset] attempt {attempt + 1}/3 failed: {e}")
                try:
                    self._destroy_actors()
                except Exception:
                    pass
                time.sleep(1.0)
        raise RuntimeError(f"reset() failed after 3 attempts: {last_err}")

    def _reset_once(self, seed=None, options=None):
        self._destroy_actors()
        self.collision_history = []
        self.image = None
        self.step_count = 0
        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)
        self.route_progress = 0.0
        self._prev_seg_t = 0.0
        self._prev_seg_idx = 0
        # Telemetry: count how many reward-side guards trip per episode.
        self._reward_clip_hits = 0
        self._progress_clamp_hits = 0
        # Explicit zeroing per Agent-2 audit: previously getattr()-defaulted
        # and could leak across episodes if the agent reached goal in N-1.
        self._at_goal_steps = 0
        self._last_termination_reason = None
        # 30s really-stuck net: track cumulative route_progress, not per-step
        # delta. Updating on per-step delta > 0.5m falsely concluded the agent
        # was "stuck" any time it drove at or below target speed
        # (target=8.33 m/s -> 0.42m/step, BELOW 0.5m threshold), killing every
        # clean run at exactly step ~601. Cumulative version updates the
        # anchor whenever route_progress advances by another 1m, regardless
        # of per-step rate.
        self._last_significant_progress_step = 0
        self._significant_progress_anchor = 0.0

        # Weather randomization
        if self.enable_weather_randomization:
            self.world.set_weather(random.choice(WEATHER_PRESETS))

        # Spawn ego vehicle — only use spawn points on driving lanes (not sidewalks)
        bp = self.blueprint_library.filter("vehicle.tesla.model3")[0]
        spawn_points = self.map.get_spawn_points()
        # Filter to spawn points that sit on a driving lane and have clear road ahead.
        # We use project_to_road=True (snap to nearest driving lane) and then
        # check the snap distance.  project_to_road=False is unreliable: it
        # returns None unless the point is *exactly* inside a lane polygon,
        # which most spawn points are not — causing the filter to reject
        # everything and silently fall back to the unfiltered list.
        valid_spawns = []
        for sp in spawn_points:
            wp = self.map.get_waypoint(sp.location, project_to_road=True,
                                        lane_type=carla.LaneType.Driving)
            if wp is None:
                continue
            # Must be very close to a driving lane (rejects sidewalks)
            dist = sp.location.distance(wp.transform.location)
            if dist > 1.5:
                continue
            # Spawn heading must match lane direction (rejects wrong-way spawns).
            # Compare spawn yaw to waypoint yaw — reject if >60° off.
            sp_yaw = sp.rotation.yaw
            wp_yaw = wp.transform.rotation.yaw
            yaw_diff = abs(sp_yaw - wp_yaw) % 360
            if yaw_diff > 180:
                yaw_diff = 360 - yaw_diff
            if yaw_diff > 60:
                continue
            # Check that the road ahead isn't immediately blocked.
            # 15m lookahead (was 5m) — one full intersection block,
            # so we don't spawn into a route that immediately dead-ends
            # and triggers off-route termination.
            nexts = wp.next(15.0)
            if nexts:
                valid_spawns.append(sp)
        print(f"[spawn-filter] {len(valid_spawns)}/{len(spawn_points)} spawn points "
              f"passed driving-lane filter")
        if not valid_spawns:
            # Don't silently fall back to unfiltered — that's how the agent
            # ends up spawning on a sidewalk facing the wrong way and then
            # U-turns onto oncoming traffic. Raise so reset() retries cleanly.
            raise RuntimeError("Spawn filter produced 0 valid points")
        random.shuffle(valid_spawns)
        self.vehicle = None
        start_sp = None
        spawn_failures = 0
        yaw_rejects = 0
        for sp in valid_spawns:
            try:
                v = self.world.spawn_actor(bp, sp)
            except RuntimeError:
                spawn_failures += 1
                continue
            # Settle physics and re-verify yaw alignment AFTER the spawn —
            # CARLA can drop the actor slightly off, and the strict pre-filter
            # (60deg) doesn't account for post-spawn rotation. 30deg is the
            # tighter post-spawn tolerance; misaligned spawns get destroyed.
            try:
                self.world.tick(seconds=10.0)
            except Exception:
                pass
            try:
                veh_yaw = v.get_transform().rotation.yaw
                wp = self.map.get_waypoint(sp.location, project_to_road=True,
                                           lane_type=carla.LaneType.Driving)
                wp_yaw = wp.transform.rotation.yaw
                yd = abs(veh_yaw - wp_yaw) % 360
                if yd > 180:
                    yd = 360 - yd
                if yd > 30:
                    try:
                        v.destroy()
                    except Exception:
                        pass
                    yaw_rejects += 1
                    continue
                self.vehicle = v
                start_sp = sp
                print(f"[spawn] at ({sp.location.x:.1f}, {sp.location.y:.1f}), "
                      f"yaw_diff={yd:.1f}deg, road_id={wp.road_id}, lane_id={wp.lane_id} "
                      f"({spawn_failures} failed, {yaw_rejects} yaw-rejected)")
                break
            except Exception as e:
                try:
                    v.destroy()
                except Exception:
                    pass
                spawn_failures += 1
                continue
        if self.vehicle is None:
            raise RuntimeError(
                f"No valid spawn after {spawn_failures} failures, "
                f"{yaw_rejects} yaw-rejects"
            )

        # Generate route to a distant spawn point
        self._generate_route(start_sp, spawn_points)

        # Semantic segmentation camera
        cam_bp = self.blueprint_library.find("sensor.camera.semantic_segmentation")
        cam_bp.set_attribute("image_size_x", str(IMG_W))
        cam_bp.set_attribute("image_size_y", str(IMG_H))
        cam_bp.set_attribute("fov", "110")
        cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(cam_bp, cam_transform, attach_to=self.vehicle)
        self.camera.listen(self._on_image)
        # Wipe self.image AFTER the new listener is attached: a callback
        # from the prior episode's camera can land between destroy and
        # the new listen() call, leaving stale frame data on self.image
        # that the wait loop below would mistake for "first frame ready".
        self.image = None

        # Collision sensor
        col_bp = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(col_bp, carla.Transform(), attach_to=self.vehicle)
        # Only count collisions with significant force. Threshold 2000
        # filters curb scrapes, pole brushes, and minor NPC contacts
        # that were killing good driving episodes prematurely.
        def _on_collision(event):
            impulse = event.normal_impulse
            force = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
            # Threshold 2500 (was 2000): a 1200kg car at 8 m/s tangentially
            # scraping a curb during a turn legitimately produces 1500-2500N.
            # The previous threshold killed honest driving on tight turns.
            if force > 2500.0:
                self.collision_history.append(event)
        self.collision_sensor.listen(_on_collision)

        # Wait for first image BEFORE spawning traffic — if traffic spawns
        # first, NPCs can collide with the ego vehicle during the image wait,
        # terminating the episode before the agent ever acts.
        # CRITICAL: never proceed with self.image == None. A black image
        # contaminates the frame-stack for 4 timesteps and is a primary
        # cause of figure-8 / circling at episode start. Raise so reset()
        # retries; if all retries fail, the trainer crashes cleanly and
        # the watchdog restarts.
        got_image = False
        for _ in range(100):  # ~5s sim time, ~5-15s wall
            try:
                # seconds=10.0: if the CARLA server hangs on tick, surface
                # it as a RuntimeError within 10s instead of blocking the
                # worker forever (GitHub issue #3: multi-env sync deadlock).
                self.world.tick(seconds=10.0)
            except RuntimeError as e:
                # Server hiccup mid-tick. One retry, then raise.
                print(f"[reset] world.tick() raised {e}; retrying once")
                time.sleep(0.5)
                self.world.tick(seconds=10.0)
            if self.image is not None:
                got_image = True
                break
            time.sleep(0.01)

        if not got_image:
            raise RuntimeError(
                f"Camera failed to deliver first frame after 100 ticks "
                f"(port={self.port}). Will retry reset()."
            )

        # Spawn traffic after ego is settled and camera is active
        if self.enable_traffic:
            self._spawn_traffic()

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        self._worker_step_counter += 1

        steer = float(np.clip(action[0], -1.0, 1.0))
        throttle_brake = float(np.clip(action[1], -1.0, 1.0))

        # Idle-creep action bias — the single most important fix after
        # PC-Claude's 95k-step diagnostic (commit 871451e).
        #
        # Problem the fix addresses: fresh PPO policy samples throttle_brake
        # from N(mean=0, std=0.367). With the old decoder:
        #   P(brake applied)    ≈ 45% of samples (action[1] < -0.05)
        #   P(deadzone, nothing) ≈ 11%
        #   P(throttle applied) ≈ 45%
        # Brakes are a strong anti-motion force; they win when alternating
        # with weak throttle samples. Result: the car never moves, so PPO
        # never observes "moving" states, so no useful policy gradient ever
        # flows. The 95k-step run was stuck at avg speed 0.000 m/s with
        # std flat at its initial value — dead-zero gradient.
        #
        # Fix: shift the neutral point so action[1] = 0 (policy init mean)
        # produces a throttle of 0.3 — like an automatic transmission's
        # idle creep. Brake only fires when action[1] < -0.3.
        #
        # New mapping:
        #   action[1] =  0.0   → throttle=0.30, brake=0.00  (idle creep)
        #   action[1] = -0.3   → throttle=0.00, brake=0.00  (coast)
        #   action[1] = -1.0   → throttle=0.00, brake=0.70  (full brake)
        #   action[1] =  0.7   → throttle=1.00, brake=0.00  (full throttle)
        #
        # With new mapping, P(throttle fires | policy init) ≈ 79%,
        # P(brake fires) ≈ 21%. Strong motion prior.
        #
        # Justification: Silver et al. 2018 "Residual Policy Learning" (fixed
        # baseline + learned residual); Andrychowicz et al. 2020 "What Matters
        # in On-Policy RL" ICLR (action parameterization among top-5 impactful
        # choices). Models real-car automatic-transmission idle semantics.
        shifted = throttle_brake + 0.3
        throttle = max(0.0, min(1.0, shifted))
        brake = max(0.0, min(1.0, -shifted))
        self.vehicle.apply_control(carla.VehicleControl(
            throttle=throttle, steer=steer, brake=brake
        ))

        # Defensive tick with timeout: a single transient server hiccup
        # shouldn't kill the whole rollout, AND an indefinite hang (GitHub
        # issue #3: multi-env sync deadlock) must not block SubprocVecEnv
        # forever. seconds=10.0 turns a hang into a RuntimeError we can
        # reconnect-retry once, then surface. If the retry also fails,
        # the worker raises; SubprocVecEnv propagates; PPO crashes with a
        # clear traceback (which the watchdog can restart from).
        try:
            self.world.tick(seconds=10.0)
        except RuntimeError as e:
            print(f"[step] world.tick() raised {e}; reconnecting and retrying once")
            try:
                self.client = carla.Client(CARLA_HOST, self.port)
                self.client.set_timeout(20.0)
                self.world = self.client.get_world()
                self.world.tick(seconds=10.0)
            except Exception as e2:
                raise RuntimeError(f"world.tick() failed after reconnect: {e2}")

        obs = self._get_obs()
        reward, terminated = self._compute_reward(action)
        self.prev_action = np.array([steer, throttle_brake], dtype=np.float32)

        # Inline per-worker spectator camera update. Each worker owns its own
        # CARLA server (port N); setting the spectator transform on THAT server
        # makes THAT server's window follow THAT worker's ego — so with 4 envs
        # the user sees 4 CARLA windows each tracking their own agent.
        # This REPLACES the external scripts/spectator.py process, which was a
        # secondary client on port 2000 that created the multi-client sync-mode
        # race documented in CARLA issues #1996 / #2239 (GitHub issue #4).
        # Cost: one RPC per step per worker (~microseconds on localhost).
        if self.vehicle is not None:
            try:
                loc = self.vehicle.get_location()
                self.world.get_spectator().set_transform(carla.Transform(
                    carla.Location(x=loc.x, y=loc.y, z=loc.z + 40),
                    carla.Rotation(pitch=-90),
                ))
            except Exception:
                # Spectator is cosmetic — never let its failure kill training.
                pass

        truncated = self.step_count >= self.max_episode_steps

        # Resolve termination reason. _compute_reward sets
        # self._last_termination_reason for all terminal paths; truncation
        # by max_episode_steps is handled here.
        reason = self._last_termination_reason
        if reason is None and truncated:
            reason = "MAX_STEPS"

        # Summary log for terminals whose reasons don't print inline.
        if (terminated or truncated) and reason not in (
                "ROUTE_COMPLETE", "REACHED_NO_PARK", "REALLY_STUCK"):
            speed = self._get_speed()
            rc = self._get_route_completion()
            print(f"[EPISODE END] reason={reason} steps={self.step_count} "
                  f"({self.step_count*0.05:.1f}s) speed={speed:.1f}m/s "
                  f"route={rc*100:.1f}% "
                  f"progress={self.route_progress:.1f}m/{self.total_route_length:.0f}m "
                  f"clip_hits={self._reward_clip_hits} "
                  f"prog_clamp_hits={self._progress_clamp_hits}")

        info = {
            "route_completion": self._get_route_completion(),
            "speed": self._get_speed(),
            "collisions": len(self.collision_history),
            "step": self.step_count,
            "reward_clip_hits": self._reward_clip_hits,
            "progress_clamp_hits": self._progress_clamp_hits,
            "termination_reason": reason,
            "worker_step": self._worker_step_counter,
        }

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # ROUTE PLANNING (GAP 1 fix)
    # ------------------------------------------------------------------

    def _generate_route(self, start_sp, spawn_points):
        """Generate a route from start to a distant destination using GlobalRoutePlanner."""
        start_loc = start_sp.location

        # Pick a distant destination for a long route
        candidates = []
        for sp in spawn_points:
            d = start_loc.distance(sp.location)
            if 200.0 < d < 800.0:
                candidates.append(sp)
        # If not enough far destinations, relax to 100m+
        if len(candidates) < 3:
            candidates = []
            for sp in spawn_points:
                d = start_loc.distance(sp.location)
                if 100.0 < d < 800.0:
                    candidates.append(sp)

        if not candidates:
            # Fallback: pick the farthest spawn point
            candidates = sorted(spawn_points, key=lambda s: start_loc.distance(s.location), reverse=True)

        dest_sp = random.choice(candidates[:5]) if len(candidates) >= 5 else candidates[0]

        # Generate route waypoints
        route = self._grp.trace_route(start_loc, dest_sp.location)
        self.route = [wp for wp, _ in route]
        self.route_index = 0

        # Compute total route length
        self.total_route_length = 0.0
        for i in range(1, len(self.route)):
            self.total_route_length += self.route[i].transform.location.distance(
                self.route[i - 1].transform.location
            )
        self.total_route_length = max(self.total_route_length, 1.0)  # avoid div by zero

    def _advance_route_index(self):
        """Compute continuous route progress along the planned route.

        Projects ego position onto route segments for smooth per-step
        progress measurement.  Careful to avoid double-counting: when
        crossing a segment boundary, we subtract the portion of the old
        segment that was already credited in previous steps.
        """
        if not self.route or self.route_index >= len(self.route) - 1:
            return 0.0

        ego_loc = self.vehicle.get_location()

        # Use ONLY 2D along-track projection for everything.
        # The old code used 3D Euclidean distance for crossing detection
        # but 2D projection for progress — lateral offset inflated the
        # 3D distance, causing premature segment advancement and then
        # multiple steps of zero progress_delta (t clamped to 0).

        def _project_t(seg_start, seg_end):
            """Project ego onto segment, return t in [0, 1] and segment length."""
            sx = seg_end.x - seg_start.x
            sy = seg_end.y - seg_start.y
            seg_len_sq = sx * sx + sy * sy
            if seg_len_sq < 0.01:
                return 0.0, 0.0
            t = ((ego_loc.x - seg_start.x) * sx + (ego_loc.y - seg_start.y) * sy) / seg_len_sq
            return t, math.sqrt(seg_len_sq)

        # Advance past segments where projection t >= 1.0 (fully crossed).
        # Cap to 5 segments per tick (~10m at 2m sampling): physical motion
        # per tick is at most MAX_SPEED * dt = 0.7m, so 10m of route_index
        # advancement is already a 14x acceleration. Without the cap, a
        # tight U-turn route (where the planned path doubles back) lets the
        # projection skip many waypoints in one step, racing route_index to
        # len-2 and triggering ROUTE_COMPLETE while the agent is still
        # geographically mid-route.
        advanced = 0
        while self.route_index < len(self.route) - 1 and advanced < 5:
            wp_loc = self.route[self.route_index].transform.location
            next_wp_loc = self.route[self.route_index + 1].transform.location
            t, seg_len = _project_t(wp_loc, next_wp_loc)
            if t >= 1.0:
                self.route_index += 1
                advanced += 1
            else:
                break

        # Compute progress_delta from the projection on current segment
        progress_delta = 0.0
        if self.route_index < len(self.route) - 1:
            wp_loc = self.route[self.route_index].transform.location
            next_wp_loc = self.route[self.route_index + 1].transform.location
            t, seg_len = _project_t(wp_loc, next_wp_loc)
            t = max(0.0, min(1.0, t))

            if self._prev_seg_idx == self.route_index:
                # Same segment — incremental progress
                dt = t - self._prev_seg_t
                if dt > 0:
                    progress_delta = dt * seg_len
            elif self.route_index > self._prev_seg_idx:
                # Crossed segment(s) — credit remaining old segment + new segment progress
                # Remaining portion of old segment
                if self._prev_seg_idx < len(self.route) - 1:
                    old_wp = self.route[self._prev_seg_idx].transform.location
                    old_next = self.route[self._prev_seg_idx + 1].transform.location
                    _, old_seg_len = _project_t(old_wp, old_next)
                    progress_delta += (1.0 - self._prev_seg_t) * old_seg_len
                # Full segments in between
                for idx in range(self._prev_seg_idx + 1, self.route_index):
                    if idx < len(self.route) - 1:
                        s = self.route[idx].transform.location
                        e = self.route[idx + 1].transform.location
                        progress_delta += s.distance(e)
                # Current segment progress
                progress_delta += t * seg_len

            self._prev_seg_t = t
            self._prev_seg_idx = self.route_index

        # Clamp to physically plausible per-step range.
        # MAX_SPEED * dt = 14.0 * 0.05 = 0.7m, so 1.5m gives 2x margin.
        # Anything bigger is a projection bug, route swap, or collision rebound.
        # A negative delta means a U-turn or reverse; we discard (don't reward).
        # Without this clamp a single bad projection produces a +200 reward
        # spike that blows out the value function and NaNs PPO.
        if not math.isfinite(progress_delta) or progress_delta < 0.0:
            self._progress_clamp_hits += 1
            return 0.0
        if progress_delta > 1.5:
            self._progress_clamp_hits += 1
            return 1.5
        return progress_delta

    def _get_route_completion(self):
        """Fraction of route completed [0, 1]."""
        if not self.route or self.total_route_length < 1.0:
            return 0.0
        return min(self.route_progress / self.total_route_length, 1.0)

    # ------------------------------------------------------------------
    # REWARD FUNCTION — CaRL-minimal (Jaeger, Chitta, Geiger 2025 §3.2)
    # with a small velocity carrot annealed to zero for no-gradient-desert
    # breakage during from-scratch exploration.
    #
    # Per-step:
    #   r_progress  = 0.05 * min(progress_delta_m, TARGET_SPEED*dt)   (CaRL)
    #   r_carrot    = 0.005 * min(speed, TARGET)/TARGET * anneal_coef (warmup)
    #                 anneal_coef linearly 1 -> 0 over CARROT_DECAY_STEPS
    #                 per-worker env-steps. Cite: Hessel et al. 2018 "Rainbow"
    #                 §3.2 warmup shaping; Vinyals et al. 2019 AlphaStar
    #                 early-phase progress shaping. Rationale for including:
    #                 pure progress reward is ~0 per step until the agent
    #                 discovers motion, which can take O(M) random-policy
    #                 steps per Andrychowicz et al. 2021 — the annealed carrot
    #                 supplies a gradient during that window. By the time the
    #                 reward becomes pure CaRL, the policy has a driving prior.
    #
    # Terminals (scale 10x vs per-step so they dominate episode return even
    # after γ=0.99 discount at ~500-step episodes; compare with shaping sum
    # of 500 * 0.021 = ~10 max):
    #   route_complete (2D dist<5m, rc>0.85, speed<3 m/s)  = +50.0
    #   collision (impulse > 2500 N)                        = -50.0
    #   off_route (min-to-route > 30m, after step 20)       = -50.0
    #   really_stuck (no 1m progress in 1500 steps = 75s)   = -50.0
    #
    # Defensive clip widened to [-100, 100] (from [-10, 10]) so the ±50
    # terminal is never itself clipped. VecNormalize still clips normalized
    # rewards to ±10; that operates on the z-scored value and is unrelated
    # to this raw clip.
    #
    # DELETED vs prior design (each was a documented exploit surface):
    #   - speed_reward with signed-cos alignment -> perpendicular attractor
    #     (Ng/Harada/Russell 1999 non-potential shaping creates new optima)
    #   - overspeed_penalty -> redundant with progress_cap
    #   - lateral_penalty   -> redundant with off_route termination
    #   - smoothness_penalty (CAPS) -> BC-warmstart provides the prior
    #   - idle_penalty with gating -> 1.5 m/s threshold created hover attractor
    #   - stagnation_counter termination -> twitch-game exploit (commit 8791388)
    # ------------------------------------------------------------------

    # Per-worker global step counter and carrot anneal horizon (env-steps).
    # Class-level sentinels; instance overrides in __init__.
    CARROT_DECAY_STEPS_DEFAULT = 500_000

    def _compute_reward(self, action):
        speed = self._get_speed()

        # 1. Route progress — primary positive signal (CaRL §3.2).
        progress_delta = self._advance_route_index()
        self.route_progress += progress_delta
        TARGET_PROGRESS_CAP = TARGET_SPEED * 0.05  # = 0.4165 m per tick @ 20 Hz
        capped_progress = min(progress_delta, TARGET_PROGRESS_CAP)
        progress_reward = 0.05 * capped_progress

        # 2. Velocity carrot (annealed warmup shaping).
        # Small positive for any forward speed up to TARGET, decays to 0.
        # Scale matches progress: 0.005 max vs 0.021 max progress per step,
        # so even at max carrot the progress signal is 4x stronger.
        anneal_coef = max(0.0,
                          1.0 - self._worker_step_counter / self._carrot_decay_steps)
        carrot = 0.005 * min(speed, TARGET_SPEED) / TARGET_SPEED * anneal_coef

        reward = progress_reward + carrot
        terminated = False
        termination_reason = None

        # Cumulative progress anchor for the REALLY_STUCK backstop.
        if self.route_progress - self._significant_progress_anchor > 1.0:
            self._significant_progress_anchor = self.route_progress
            self._last_significant_progress_step = self.step_count

        # Terminal: collision
        if len(self.collision_history) > 0:
            reward = -50.0
            terminated = True
            termination_reason = "COLLISION"

        # Terminal: route complete (2D distance — ignore Z so bridges/slopes
        # don't artificially block the goal check).
        if not terminated and self.route and len(self.route) >= 2:
            final_wp = self.route[-1].transform.location
            ego_loc_g = self.vehicle.get_location()
            dx_g = ego_loc_g.x - final_wp.x
            dy_g = ego_loc_g.y - final_wp.y
            dist_to_goal_2d = math.sqrt(dx_g * dx_g + dy_g * dy_g)
            rc = self._get_route_completion()
            at_goal = (dist_to_goal_2d < 5.0 and rc > 0.85) or \
                      self.route_index >= len(self.route) - 2
            if at_goal and speed < 3.0:
                reward = 50.0
                terminated = True
                termination_reason = "ROUTE_COMPLETE"
                print(f"[EPISODE END] reason=ROUTE_COMPLETE "
                      f"dist={dist_to_goal_2d:.1f}m rc={rc:.2%} "
                      f"speed={speed:.2f}m/s step={self.step_count}")
            elif at_goal:
                self._at_goal_steps += 1
                if self._at_goal_steps > 200:
                    reward = 0.0
                    terminated = True
                    termination_reason = "REACHED_NO_PARK"
                    print(f"[EPISODE END] reason=REACHED_NO_PARK "
                          f"dist={dist_to_goal_2d:.1f}m speed={speed:.2f}m/s "
                          f"step={self.step_count}")
            else:
                self._at_goal_steps = 0

        # Terminal: off-route (30 m from nearest waypoint in search window)
        if (not terminated and self.step_count > 20 and self.route
                and self.route_index < len(self.route) - 3):
            ego_loc = self.vehicle.get_location()
            end_idx = min(self.route_index + 50, len(self.route))
            start_idx = max(0, self.route_index - 20)
            dist_to_route = min(
                ego_loc.distance(self.route[i].transform.location)
                for i in range(start_idx, end_idx)
            )
            if dist_to_route > 30.0:
                reward = -50.0
                terminated = True
                termination_reason = "OFF_ROUTE"

        # Terminal: really stuck — 75s (1500 sim-steps) without 1m of
        # cumulative progress. Replaces the stagnation_counter + idle_penalty
        # mess. Only truly wedged states (curb lodge, wall deadlock, dense
        # traffic with no legal escape) will trigger this under the new
        # reward, because the pure-progress signal gives no reason to stop.
        if (not terminated
                and self.step_count - self._last_significant_progress_step > 1500):
            reward = -50.0
            terminated = True
            termination_reason = "REALLY_STUCK"
            print(f"[EPISODE END] reason=REALLY_STUCK "
                  f"steps_since_progress={self.step_count - self._last_significant_progress_step}")

        # Defensive clip widened to [-100, 100]. NaN/inf still force-terminate.
        if not math.isfinite(reward):
            print(f"[reward-guard] non-finite reward {reward} -> 0.0; terminating")
            reward = 0.0
            terminated = True
            termination_reason = "NAN_REWARD"
        if reward > 100.0 or reward < -100.0:
            self._reward_clip_hits += 1
            reward = max(-100.0, min(100.0, reward))

        # Store for step() to emit in info dict.
        self._last_termination_reason = termination_reason
        return reward, terminated

    def _is_blocked_by_vehicle(self):
        """True if there's another vehicle within 8m forward in our lane.

        Used to gate stagnation checks: queued behind a bus at a green
        light is not the same as actually being stuck.

        Cached with a 10-step TTL (0.5s) — without caching this iterates
        all world vehicles per step (30+ RPC calls/step per env). At
        20Hz × 2 envs that's ~1200 CARLA RPCs/sec, a real throughput
        bottleneck. Blocking state changes on >100ms timescales, so
        caching is correctness-preserving.
        """
        if self.vehicle is None:
            return False
        cache_step = getattr(self, "_blocked_cache_step", -999)
        if self.step_count - cache_step < 10:
            return getattr(self, "_blocked_cache_val", False)
        result = False
        try:
            ego_t = self.vehicle.get_transform()
            ego_loc = ego_t.location
            yaw_r = math.radians(ego_t.rotation.yaw)
            fx, fy = math.cos(yaw_r), math.sin(yaw_r)
            for other in self.world.get_actors().filter("vehicle.*"):
                if other.id == self.vehicle.id:
                    continue
                ol = other.get_location()
                dx, dy = ol.x - ego_loc.x, ol.y - ego_loc.y
                forward = dx * fx + dy * fy
                lateral = -dx * fy + dy * fx
                if 0.0 < forward < 8.0 and abs(lateral) < 2.0:
                    result = True
                    break
        except Exception:
            result = False
        self._blocked_cache_step = self.step_count
        self._blocked_cache_val = result
        return result

    # ------------------------------------------------------------------
    # OBSERVATION (GAP 4 fix — expanded state vector + single channel image)
    # ------------------------------------------------------------------

    def _get_obs(self):
        ego_loc = self.vehicle.get_location() if self.vehicle else None
        ego_transform = self.vehicle.get_transform() if self.vehicle else None

        # Speed
        vel = self.vehicle.get_velocity() if self.vehicle else None
        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2) if vel else 0.0

        # Waypoint encoding: next 3 route waypoints in ego-frame coordinates,
        # CLAMPED to [-1, 1] (Agent-2 audit: unclamped /20 could reach ±5 at
        # sharp turns, putting network inputs outside its training distribution).
        wp_features = np.zeros(6, dtype=np.float32)
        if ego_transform and self.route:
            ego_yaw_rad = math.radians(ego_transform.rotation.yaw)
            cos_yaw = math.cos(ego_yaw_rad)
            sin_yaw = math.sin(ego_yaw_rad)

            for i, offset in enumerate([0, 2, 5]):  # waypoints at ~0m, ~4m, ~10m ahead
                wp_idx = min(self.route_index + offset, len(self.route) - 1)
                wp_loc = self.route[wp_idx].transform.location

                # World-frame offset
                dx_world = wp_loc.x - ego_loc.x
                dy_world = wp_loc.y - ego_loc.y

                # Rotate to ego frame
                dx_ego = dx_world * cos_yaw + dy_world * sin_yaw
                dy_ego = -dx_world * sin_yaw + dy_world * cos_yaw

                # Normalize (20m reference) and clamp to [-1, 1].
                wp_features[i * 2] = max(-1.0, min(1.0, dx_ego / 20.0))
                wp_features[i * 2 + 1] = max(-1.0, min(1.0, dy_ego / 20.0))

        # Lane offset (SIGNED distance to lane center)
        # Positive = right of center, negative = left.  Gives the agent
        # directional information so it knows WHICH way to steer.
        lane_offset = 0.0
        wp = self.map.get_waypoint(ego_loc, project_to_road=True,
                                    lane_type=carla.LaneType.Driving) if ego_loc else None
        if wp and ego_loc:
            wp_loc = wp.transform.location
            wp_yaw_rad = math.radians(wp.transform.rotation.yaw)
            wp_fwd_x = math.cos(wp_yaw_rad)
            wp_fwd_y = math.sin(wp_yaw_rad)
            dx = ego_loc.x - wp_loc.x
            dy = ego_loc.y - wp_loc.y
            # Cross product of lane-forward × ego-offset = signed lateral distance
            signed_offset = wp_fwd_x * dy - wp_fwd_y * dx
            lane_offset = max(-1.0, min(1.0, signed_offset / 5.0))  # normalize to [-1, 1]

        # Traffic light state — fixed encoding per Agent-2 audit.
        # 0.0 = NO LIGHT PRESENT (safe to proceed)
        # 0.33 = GREEN       (permissive)
        # 0.67 = YELLOW      (prepare to stop)
        # 1.00 = RED or UNKNOWN (treat unknown/off as red for safety)
        # The prior encoding conflated "no light" with "unknown state", so
        # a malfunctioning/off signal would read as 0.0 and the agent would
        # learn to accelerate through it.
        tl_state = 0.0
        if self.vehicle and self.vehicle.is_at_traffic_light():
            traffic_light = self.vehicle.get_traffic_light()
            if traffic_light is None:
                # At a light but handle is None — treat conservatively as red.
                tl_state = 1.0
            else:
                state = traffic_light.get_state()
                if state == carla.TrafficLightState.Green:
                    tl_state = 1.0 / 3.0
                elif state == carla.TrafficLightState.Yellow:
                    tl_state = 2.0 / 3.0
                elif state == carla.TrafficLightState.Red:
                    tl_state = 1.0
                else:
                    # Off / Unknown — treat as red.
                    tl_state = 1.0

        # Route completion percentage (always in [0, 1])
        route_pct = self._get_route_completion()

        # Assemble 10-element state vector. prev_steer/prev_throttle removed
        # per Agent-3 audit (POMDP-smell + BC-to-PPO distribution shift).
        state = np.array([
            speed / MAX_SPEED,             # [0] normalized speed in [0, ~1]
            wp_features[0],                # [1] waypoint 1 dx (ego frame, clamped)
            wp_features[1],                # [2] waypoint 1 dy
            wp_features[2],                # [3] waypoint 2 dx
            wp_features[3],                # [4] waypoint 2 dy
            wp_features[4],                # [5] waypoint 3 dx
            wp_features[5],                # [6] waypoint 3 dy
            lane_offset,                   # [7] signed lane offset, normalized [-1, 1]
            tl_state,                      # [8] traffic light state (see encoding above)
            route_pct,                     # [9] route completion fraction [0, 1]
        ], dtype=np.float32)

        # Image: single-channel normalized semantic labels
        if self.image is not None:
            image = self.image
        else:
            image = np.zeros((1, IMG_H, IMG_W), dtype=np.float32)

        return {"image": image, "state": state}

    def _on_image(self, image):
        """Process semantic segmentation camera output into single-channel normalized labels."""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((IMG_H, IMG_W, 4))
        # CARLA BGRA layout: semantic class label is in R channel (index 2)
        labels = array[:, :, 2].astype(np.float32)
        # Normalize to [0, 1] range.  Labels span [0, 27] so divide by 27
        # (not 28) to map the highest class to exactly 1.0.
        labels = labels / (NUM_SEMANTIC_CLASSES - 1)
        # Single channel: (1, H, W)
        self.image = labels[np.newaxis, :, :]

    # ------------------------------------------------------------------
    # TRAFFIC (GAP 6 fix)
    # ------------------------------------------------------------------

    def _spawn_traffic(self):
        """Spawn NPC vehicles and pedestrians using CARLA's Traffic Manager."""
        self._destroy_traffic()

        try:
            # Reuse the cached TM handle created in __init__. Do NOT call
            # get_trafficmanager() again per reset — see CARLA issues #2789
            # and #9172 for why re-registration under a slow-tick window
            # causes silent deadlocks.
            tm = self._tm
            # Re-seed each episode for traffic diversity across resets, but
            # do not toggle synchronous_mode or other state — those were set
            # once at __init__ and should stay stable.
            tm.set_random_device_seed(random.randint(0, 10000))

            # Spawn vehicles
            vehicle_bps = self.blueprint_library.filter("vehicle.*")
            # Filter out bikes/motorcycles for simplicity
            vehicle_bps = [bp for bp in vehicle_bps if int(bp.get_attribute("number_of_wheels")) >= 4]

            spawn_points = self.map.get_spawn_points()
            random.shuffle(spawn_points)

            # Don't spawn at ego's location
            ego_loc = self.vehicle.get_location()
            safe_spawns = [sp for sp in spawn_points if sp.location.distance(ego_loc) > 30.0]

            for i in range(min(self.num_traffic_vehicles, len(safe_spawns))):
                bp = random.choice(vehicle_bps)
                if bp.has_attribute("color"):
                    color = random.choice(bp.get_attribute("color").recommended_values)
                    bp.set_attribute("color", color)
                try:
                    npc = self.world.spawn_actor(bp, safe_spawns[i])
                    npc.set_autopilot(True, tm.get_port())
                    self.traffic_actors.append(npc)
                except RuntimeError:
                    continue

            # Spawn pedestrians
            ped_bps = self.blueprint_library.filter("walker.pedestrian.*")
            walker_controller_bp = self.blueprint_library.find("controller.ai.walker")

            for _ in range(self.num_pedestrians):
                bp = random.choice(ped_bps)
                if bp.has_attribute("is_invincible"):
                    bp.set_attribute("is_invincible", "false")
                spawn_point = carla.Transform()
                loc = self.world.get_random_location_from_navigation()
                if loc is None:
                    continue
                spawn_point.location = loc
                walker = None
                try:
                    walker = self.world.spawn_actor(bp, spawn_point)
                    self.pedestrian_actors.append(walker)
                except RuntimeError:
                    continue
                try:
                    controller = self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker)
                    self.pedestrian_actors.append(controller)
                    controller.start()
                    controller.go_to_location(self.world.get_random_location_from_navigation())
                    controller.set_max_speed(1.0 + random.random() * 1.5)
                except RuntimeError:
                    # Walker spawned but controller failed — walker is already
                    # tracked in self.pedestrian_actors so it will be cleaned up.
                    pass

        except Exception as e:
            print(f"Warning: traffic spawning failed: {e}")

    def _destroy_traffic(self):
        """Clean up all spawned traffic actors.

        Controllers must be stopped and destroyed before their parent walkers,
        so we split the pedestrian list into controllers vs walkers and handle
        them in the correct order.

        After destroying our tracked actors, we also scan the world for any
        orphaned NPCs (e.g. from a partially-failed spawn) and destroy those
        too — this is the safety net against ghost cars.
        """
        # Disable autopilot on traffic vehicles before destroying, so the
        # TrafficManager releases them cleanly.
        for actor in self.traffic_actors:
            try:
                actor.set_autopilot(False)
            except Exception:
                pass

        controllers = []
        walkers = []
        for actor in self.pedestrian_actors:
            try:
                type_id = actor.type_id
            except Exception:
                # Stale handle — skip; the safety-net sweep below will catch it
                continue
            try:
                if type_id == "controller.ai.walker":
                    actor.stop()
                    controllers.append(actor)
                else:
                    walkers.append(actor)
            except Exception:
                # stop() failed but we still need to destroy it
                controllers.append(actor) if "controller" in str(type_id) else walkers.append(actor)

        # Destroy controllers first (children before parents)
        if controllers:
            self._batch_destroy(controllers, label="ped-controllers")

        # Then destroy walkers + traffic vehicles together
        remaining = self.traffic_actors + walkers
        if remaining:
            self._batch_destroy(remaining, label="traffic")

        self.traffic_actors = []
        self.pedestrian_actors = []

        # Safety-net: destroy any NPC actors still in the world that we don't
        # own (e.g. from a crashed previous episode within the same process,
        # or actors spawned but never tracked due to an exception).
        ego_id = self.vehicle.id if self.vehicle is not None else None
        cam_id = self.camera.id if self.camera is not None else None
        col_id = self.collision_sensor.id if self.collision_sensor is not None else None
        owned_ids = {x for x in (ego_id, cam_id, col_id) if x is not None}

        orphans = []
        for pattern in ("vehicle.*", "walker.*", "controller.*"):
            for actor in self.world.get_actors().filter(pattern):
                if actor.id not in owned_ids:
                    if "controller" in actor.type_id:
                        try:
                            actor.stop()
                        except Exception:
                            pass
                    orphans.append(actor)
        if orphans:
            # Silently try to destroy — these are often already gone from
            # the normal cleanup above (stale handles).  Only warn if the
            # batch itself throws, not for individual "not found" errors.
            batch = [carla.command.DestroyActor(a) for a in orphans]
            try:
                self.client.apply_batch_sync(batch)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _init_route_planner(self):
        """Import CARLA's GlobalRoutePlanner, handling potential package name conflicts."""
        import sys
        import importlib

        # Try direct import first (works if CARLA's agents is on path and not shadowed)
        try:
            from agents.navigation.global_route_planner import GlobalRoutePlanner
            # Verify it's actually CARLA's module (has the trace_route method)
            if hasattr(GlobalRoutePlanner, '__init__'):
                return GlobalRoutePlanner(self.map, sampling_resolution=2.0)
        except (ImportError, AttributeError):
            pass

        # Fallback: look for CARLA PythonAPI in common locations
        carla_paths = [
            os.environ.get("CARLA_PYTHONAPI", ""),
            os.path.expanduser("~/carla/PythonAPI/carla"),
            "/opt/carla/PythonAPI/carla",
        ]
        for p in carla_paths:
            if p and os.path.isdir(os.path.join(p, "agents", "navigation")):
                sys.path.insert(0, p)
                # Force reimport
                if "agents" in sys.modules:
                    del sys.modules["agents"]
                if "agents.navigation" in sys.modules:
                    del sys.modules["agents.navigation"]
                if "agents.navigation.global_route_planner" in sys.modules:
                    del sys.modules["agents.navigation.global_route_planner"]
                from agents.navigation.global_route_planner import GlobalRoutePlanner
                return GlobalRoutePlanner(self.map, sampling_resolution=2.0)

        raise ImportError(
            "Could not import CARLA's GlobalRoutePlanner. "
            "Ensure CARLA PythonAPI is installed: "
            "export PYTHONPATH=$PYTHONPATH:/path/to/CARLA/PythonAPI/carla"
        )

    def _get_speed(self):
        """Get vehicle speed in m/s."""
        if self.vehicle is None:
            return 0.0
        vel = self.vehicle.get_velocity()
        return math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

    def _destroy_actors(self):
        """Destroy ego vehicle, sensors, and all traffic.

        In synchronous mode, actor.destroy() queues the command but the server
        only processes it on the next tick.  Without a tick the actor stays in
        the world — this is the root cause of "ghost cars" accumulating across
        episodes.  We use apply_batch_sync for reliability and tick afterward.
        """
        # Stop sensor listeners *before* destroying to prevent callbacks
        # firing on a half-destroyed actor.
        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception:
                pass
        if self.collision_sensor is not None:
            try:
                self.collision_sensor.stop()
            except Exception:
                pass

        self._destroy_traffic()

        # Batch-destroy ego actors (sensors first, then vehicle)
        ego_actors = [a for a in [self.camera, self.collision_sensor, self.vehicle]
                      if a is not None]
        if ego_actors:
            self._batch_destroy(ego_actors, label="ego")

        self.vehicle = None
        self.camera = None
        self.collision_sensor = None

    def _batch_destroy(self, actors, label=""):
        """Batch-destroy a list of actors.

        apply_batch_sync already performs an implicit tick, so we do NOT tick
        again afterward — the extra tick was advancing the simulation and
        letting partially-destroyed actors interact for one more frame.

        Falls back to individual destroy() calls + explicit tick if the batch
        command fails.
        """
        if not actors:
            return

        batch = [carla.command.DestroyActor(a) for a in actors]
        try:
            self.client.apply_batch_sync(batch)
        except Exception as e:
            print(f"[cleanup:{label}] batch destroy failed, falling back: {e}")
            for actor in actors:
                try:
                    actor.destroy()
                except Exception:
                    pass
            # Only tick explicitly when using the individual-destroy fallback,
            # because individual destroy() in sync mode queues commands that
            # require a tick to take effect. seconds=10 prevents indefinite
            # block if the destroy triggers a TM-race window (issue #4 analysis).
            try:
                self.world.tick(seconds=10.0)
            except Exception:
                pass

    def close(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
        self._destroy_actors()
