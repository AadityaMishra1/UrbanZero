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

        # Observation: single-channel semantic seg (normalized) + expanded state vector
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(low=0.0, high=1.0, shape=(1, IMG_H, IMG_W), dtype=np.float32),
            "state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
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
            # Tick to flush destructions (works whether server is sync or async)
            try:
                self.world.tick()
            except Exception:
                pass

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 Hz
        self.world.apply_settings(settings)

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
        self.stagnation_counter = 0

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
        self.stagnation_counter = 0
        self.route_progress = 0.0
        self._prev_seg_t = 0.0
        self._prev_seg_idx = 0
        # Telemetry: count how many reward-side guards trip per episode.
        self._reward_clip_hits = 0
        self._progress_clamp_hits = 0
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
                self.world.tick()
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
                self.world.tick()
            except RuntimeError as e:
                # Server hiccup mid-tick. One retry, then raise.
                print(f"[reset] world.tick() raised {e}; retrying once")
                time.sleep(0.5)
                self.world.tick()
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

        # Defensive tick: a single transient server hiccup shouldn't kill the
        # whole rollout. One reconnect+retry, then surface the failure.
        try:
            self.world.tick()
        except RuntimeError as e:
            print(f"[step] world.tick() raised {e}; reconnecting and retrying once")
            try:
                self.client = carla.Client(CARLA_HOST, self.port)
                self.client.set_timeout(20.0)
                self.world = self.client.get_world()
                self.world.tick()
            except Exception as e2:
                raise RuntimeError(f"world.tick() failed after reconnect: {e2}")

        obs = self._get_obs()
        reward, terminated = self._compute_reward(action)
        self.prev_action = np.array([steer, throttle_brake], dtype=np.float32)

        truncated = self.step_count >= self.max_episode_steps
        if self.stagnation_counter > 150:
            # Stagnation must carry a penalty equal to collision. At -2.0 and
            # 200-step threshold, the agent learned to "drive 50 steps, stop
            # 199 steps" — accumulating +50-100 from driving then eating only
            # -2.0 for stopping. Net positive, so PPO kept the strategy.
            #
            # Fix: threshold 200→150 (7.5s, still gated on red lights and
            # legit queues so real stops are unaffected) and penalty -2.0→-5.0
            # (same as collision). Now stopping and crashing are equally bad,
            # but driving forward gives +1-2/step. The only way to avoid both
            # penalties is to keep moving — which is what we want.
            terminated = True
            reward = -5.0

        # DEBUG: log WHY every episode ends. ROUTE_COMPLETE and REALLY_STUCK
        # also print inline at trigger; this is the unified summary.
        if terminated or truncated:
            speed = self._get_speed()
            rc = self._get_route_completion()
            steps_since_prog = self.step_count - self._last_significant_progress_step
            if len(self.collision_history) > 0:
                reason = "COLLISION"
            elif self.stagnation_counter > 150:
                reason = f"STAGNATION (counter={self.stagnation_counter})"
            elif rc > 0.85 or self.route_index >= len(self.route) - 2:
                reason = "ROUTE_COMPLETE"
            elif steps_since_prog > 600:
                reason = "REALLY_STUCK"
            elif self.step_count >= self.max_episode_steps:
                reason = "MAX_STEPS"
            else:
                reason = "OFF_ROUTE"
            # Skip duplicate print for reasons that print inline at trigger.
            if reason not in ("ROUTE_COMPLETE", "REALLY_STUCK"):
                print(f"[EPISODE END] reason={reason} steps={self.step_count} "
                      f"({self.step_count*0.05:.1f}s) speed={speed:.1f}m/s "
                      f"route={rc*100:.1f}% stag={self.stagnation_counter} "
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
    # REWARD FUNCTION — Roach-style dense shaping for single-env training.
    #
    # Per-step shaping (each clipped/bounded so none can dominate):
    #   r_progress  = 2.0 * progress_delta              [Roach §3.3, CaRL]
    #   r_speed     = 0.3 * (speed/TARGET) * cos(angle) [Roach signed]
    #   r_lateral   = -0.1 * min(|offset|/2.0, 1.0)     [Roach §3.3]
    #   r_smooth    = -0.05 * ||a_t - a_{t-1}||^2       [CAPS, ICRA 2021]
    # Terminals:
    #   r_collision        = -5.0                       [Roach, Toromanoff]
    #   r_route_complete   = +10.0 if parked at goal    [novel: must stop]
    #
    # Design choices: pure CaRL (progress-only) needs 300+ parallel envs
    # (Jaeger et al., CoRL 2025). With 1 env we use Roach's denser
    # additive shaping. Signed alignment (instead of a gate) is what
    # makes circling reward-negative and wrong-way reward-negative —
    # see commit 0bc0dc5 for the math.
    # ------------------------------------------------------------------

    def _compute_reward(self, action):
        speed = self._get_speed()

        # Clipped action (matches what was actually applied in step()).
        steer = float(np.clip(action[0], -1.0, 1.0))
        throttle_brake = float(np.clip(action[1], -1.0, 1.0))

        # 1. Route progress — primary positive signal (CaRL / Roach §3.3).
        # CAPPED at TARGET_SPEED-equivalent distance per step. Without
        # this cap, the 1.5m physical-plausibility clamp meant the agent
        # kept getting more reward up to ~22 m/s. Phase 1 at 1M steps
        # showed the agent converged to 18 m/s (65 km/h) in a
        # 30 km/h urban zone — the reward design was silently paying
        # for overspeed. Capping at TARGET (0.42m/step) makes TARGET
        # the reward-optimal speed; going faster pays the SAME progress
        # while losing speed_reward (triangle falls off above TARGET)
        # AND paying overspeed_penalty above MAX_SPEED.
        progress_delta = self._advance_route_index()
        self.route_progress += progress_delta
        TARGET_PROGRESS_CAP = TARGET_SPEED * 0.05  # = 0.4165 m per tick
        progress_reward = min(progress_delta, TARGET_PROGRESS_CAP) * 2.0

        # 2. Speed reward, signed by route alignment.
        # Triangle-shaped peak at TARGET_SPEED, falls to 0 at MAX_SPEED.
        if speed < TARGET_SPEED:
            speed_reward = 0.3 * (speed / TARGET_SPEED)
        elif speed < MAX_SPEED:
            speed_reward = 0.3 * (1.0 - (speed - TARGET_SPEED) / (MAX_SPEED - TARGET_SPEED))
        else:
            speed_reward = 0.0  # over-speed: no reward, no divide-by-zero

        # 2b. Overspeed penalty above MAX_SPEED.
        # Linear ramp: 0 at MAX (14 m/s), -0.3 at 2*MAX (28 m/s = 100 km/h).
        # Ensures speed above MAX actively costs reward, not just zeros
        # speed_reward. Combined with the progress cap above, the
        # reward-optimal speed under the new design is exactly TARGET_SPEED
        # (30 km/h urban) — at TARGET: +0.83 progress +0.3 speed = 1.13;
        # at MAX: +0.83 progress (capped) +0 speed = 0.83; at 17 m/s:
        # +0.83 progress +0 speed -0.064 overspeed = 0.766.
        overspeed_penalty = -0.3 * max(0.0, (speed - MAX_SPEED) / MAX_SPEED)

        # SIGNED alignment — cos(angle_diff) in [-1, +1]. Wrong-way and
        # perpendicular driving give NEGATIVE r_speed. Figure-8 averages
        # to 0 over a loop. Clean forward driving unchanged.
        alignment = 1.0
        if self.route and self.route_index < len(self.route) - 1:
            wp_loc = self.route[self.route_index].transform.location
            next_wp = self.route[min(self.route_index + 1, len(self.route) - 1)].transform.location
            route_yaw = math.degrees(math.atan2(next_wp.y - wp_loc.y, next_wp.x - wp_loc.x))
            vehicle_yaw = self.vehicle.get_transform().rotation.yaw
            angle_diff = abs(vehicle_yaw - route_yaw) % 360
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            alignment = math.cos(math.radians(angle_diff))
        speed_reward *= alignment

        # 3. Lateral deviation penalty (Roach §3.3, Toromanoff CVPR 2020).
        # Continuous per-step penalty for drifting from lane center.
        # Capped at 2m so a single-lane swerve costs the full 0.1 but a
        # routine lane change can recover quickly. Without this term,
        # the agent had no incentive to stay in lane between waypoints.
        lateral_penalty = 0.0
        if self.vehicle is not None:
            ego_loc_l = self.vehicle.get_location()
            wp_l = self.map.get_waypoint(ego_loc_l, project_to_road=True,
                                         lane_type=carla.LaneType.Driving)
            if wp_l is not None:
                wp_yaw_rad = math.radians(wp_l.transform.rotation.yaw)
                dx_l = ego_loc_l.x - wp_l.transform.location.x
                dy_l = ego_loc_l.y - wp_l.transform.location.y
                signed_offset = math.cos(wp_yaw_rad) * dy_l - math.sin(wp_yaw_rad) * dx_l
                lateral_penalty = -0.1 * min(abs(signed_offset) / 2.0, 1.0)

        # 4. Action smoothness (CAPS — Mysore et al., ICRA 2021).
        # Penalize sudden control changes to reduce steering oscillation
        # and the "shaky camera" the policy was producing. Quadratic so
        # tiny jitter is nearly free; large reversals are expensive.
        d_steer = steer - float(self.prev_action[0])
        d_throttle = throttle_brake - float(self.prev_action[1])
        smoothness_penalty = -0.05 * (d_steer * d_steer + d_throttle * d_throttle)

        # 5. Idle penalty — unconditional, NOT scaled by alignment.
        # Previous -0.2/step penalty was inside speed_reward and got flipped
        # by negative alignment (agent got REWARDED for standing still while
        # facing wrong way). Also too weak: 151 steps × -0.2 = -30, easily
        # offset by ~50 steps of driving reward.
        #
        # Gating: the penalty MUST skip red lights and legit queues, or the
        # agent gets crushed for obeying the law (a 20s red light at
        # -1.0/step = -400 reward, teaching the agent to run red lights or
        # stop far back to avoid the intersection entirely).
        #
        # Computing the gates here requires the at_red_light / blocked_ahead
        # / route_alignment values, so those are computed up-front now
        # (moved up from the stagnation-counter section below).

        # Pre-compute gating signals (shared with stagnation counter).
        at_red_light = False
        if self.vehicle is not None:
            if self.vehicle.is_at_traffic_light():
                tl = self.vehicle.get_traffic_light()
                if tl and tl.get_state() in (carla.TrafficLightState.Red,
                                             carla.TrafficLightState.Yellow):
                    at_red_light = True
        blocked_ahead = self._is_blocked_by_vehicle()
        route_alignment = 1.0
        if self.route and self.route_index < len(self.route) - 1:
            wp_a = self.route[self.route_index].transform.location
            wp_b = self.route[self.route_index + 1].transform.location
            r_yaw = math.degrees(math.atan2(wp_b.y - wp_a.y, wp_b.x - wp_a.x))
            v_yaw = self.vehicle.get_transform().rotation.yaw
            d = abs(v_yaw - r_yaw) % 360
            if d > 180:
                d = 360 - d
            route_alignment = math.cos(math.radians(d))
        legit_queue = blocked_ahead and route_alignment > 0.5
        # Red light is the only truly legitimate reason to be stationary.
        # Being queued behind traffic is NOT free — the previous version's
        # `legit_stop = at_red_light OR legit_queue` over-gated: with 30
        # NPCs around spawn, blocked_ahead almost always fires, so the
        # idle penalty never activated and the agent learned to stand
        # still forever (172k-step clean run, 94% stagnation, 0.009 m/s).
        # Queueing should still cost a little — that's a gradient to find
        # a route around or at least follow the traffic when it moves.
        legit_stop = at_red_light

        # Smooth idle penalty — continuous ramp, no cliff at speed=1.0.
        # Before: step function 0 -> -0.3 -> -1.0 with a discontinuity
        # at speed=1.0 that created a hovering attractor (agent learned
        # to crawl at exactly 1.01 m/s to avoid the penalty cliff).
        # Now: linear ramp from -1.0 at stopped to 0 at 1.5 m/s.
        if legit_stop:
            idle_penalty = 0.0  # obeying red light / queued behind bus
        elif speed >= 1.5:
            idle_penalty = 0.0
        else:
            idle_penalty = -1.0 * (1.5 - speed) / 1.5  # [-1.0 .. 0] linear

        reward = (progress_reward + speed_reward + overspeed_penalty
                  + lateral_penalty + smoothness_penalty + idle_penalty)

        # Stagnation counter — catches stopped/crawling cars.
        # Truncates after 150 steps (7.5s). More permissive gating than
        # the per-step idle penalty: we don't want to truncate the agent
        # while it's correctly queued behind traffic at a green light
        # (that's a 10-30s wait in CARLA traffic density). So the
        # counter skips both red lights AND legit queues; the idle
        # penalty only skips red lights.
        no_progress = progress_delta < 0.01
        if (speed < 0.5 or no_progress) and not at_red_light and not legit_queue:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = max(0, self.stagnation_counter - 1)

        # Cumulative progress anchor for the REALLY_STUCK safety net below.
        # Updates whenever route_progress advances by another meter — any
        # meaningful forward motion keeps the marker fresh, regardless
        # of per-step speed.
        if self.route_progress - self._significant_progress_anchor > 1.0:
            self._significant_progress_anchor = self.route_progress
            self._last_significant_progress_step = self.step_count

        # 4. Collision — small explicit penalty + episode termination.
        terminated = False
        if len(self.collision_history) > 0:
            reward = -5.0
            terminated = True

        # 5. Route completion — must REACH the goal AND PARK there.
        # Real driving means stopping at the destination, not just flying
        # past at speed. Speed gate (<3 m/s = ~11 km/h = true parking-pace)
        # forces deceleration to claim the +10 bonus.
        # Fallback: if agent has been at the goal waypoint for more than
        # 200 steps (10s) without parking, terminate with zero reward
        # so we don't waste training budget on an agent that made it to
        # the destination but can't decelerate. Avoids the "loiter near
        # goal forever" trap where r_progress=0, r_speed=+0.3 keeps
        # ticking until MAX_STEPS.
        if not terminated and self.route and len(self.route) >= 2:
            final_wp = self.route[-1].transform.location
            ego_loc_g = self.vehicle.get_location()
            dist_to_goal = ego_loc_g.distance(final_wp)
            rc = self._get_route_completion()
            at_goal = (dist_to_goal < 5.0 and rc > 0.85) or \
                      self.route_index >= len(self.route) - 2
            if at_goal and speed < 3.0:
                reward = 10.0
                terminated = True
                print(f"[EPISODE END] reason=ROUTE_COMPLETE "
                      f"dist={dist_to_goal:.1f}m rc={rc:.2%} "
                      f"speed={speed:.2f}m/s step={self.step_count}")
            elif at_goal:
                # Track loiter steps; terminate if too long without parking.
                self._at_goal_steps = getattr(self, "_at_goal_steps", 0) + 1
                if self._at_goal_steps > 200:
                    terminated = True
                    reward = 0.0  # neither bonus nor penalty — just stop
                    print(f"[EPISODE END] reason=REACHED_NO_PARK "
                          f"dist={dist_to_goal:.1f}m speed={speed:.2f}m/s "
                          f"step={self.step_count}")
            else:
                self._at_goal_steps = 0

        # 6. Off-route — terminate if too far from planned route.
        # Skip first 20 steps: spawn→route alignment isn't always perfect.
        # Skip when within 3 waypoints of the end: the search window misses
        # the geometric goal once route_index nears len, so the agent can
        # be 5m past the final waypoint (overshoot during turn-in) and
        # spuriously die OFF_ROUTE instead of completing. ROUTE_COMPLETE
        # (above) handles termination in that regime.
        if (not terminated and self.step_count > 20 and self.route
                and self.route_index < len(self.route) - 3):
            ego_loc = self.vehicle.get_location()
            end_idx = min(self.route_index + 50, len(self.route))
            start_idx = max(0, self.route_index - 20)
            dist_to_route = min(
                ego_loc.distance(self.route[i].transform.location)
                for i in range(start_idx, end_idx)
            )
            if dist_to_route > 15.0:
                reward = -5.0
                terminated = True

        # 7. Really-stuck safety net — 30s without 1m of route progress.
        # Kept as a hard backstop for genuinely-wedged states (lodged on
        # a curb, deadlocked against a wall, etc.). With the new reward
        # this should fire rarely; if it dominates the EPISODE END
        # distribution something else is wrong.
        if (not terminated
                and self.step_count - self._last_significant_progress_step > 600):
            terminated = True
            reward = -5.0
            print(f"[EPISODE END] reason=REALLY_STUCK "
                  f"steps_since_progress={self.step_count - self._last_significant_progress_step}")

        # Final defensive clip BEFORE VecNormalize sees the reward.
        # VecNormalize's clip_reward=10.0 only clips the *normalized* reward —
        # an Inf raw reward still poisons its running mean/var permanently.
        if not math.isfinite(reward):
            print(f"[reward-guard] non-finite reward {reward} -> 0.0; terminating")
            reward = 0.0
            terminated = True
        if reward > 10.0 or reward < -10.0:
            self._reward_clip_hits += 1
            reward = max(-10.0, min(10.0, reward))

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

        # Previous actions
        prev_steer = self.prev_action[0]
        prev_throttle = self.prev_action[1]

        # Waypoint encoding: next 3 route waypoints in ego-frame coordinates
        wp_features = np.zeros(6, dtype=np.float32)  # [dx1, dy1, dx2, dy2, dx3, dy3]
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

                # Normalize (divide by 20m to keep in roughly [-1, 1])
                wp_features[i * 2] = dx_ego / 20.0
                wp_features[i * 2 + 1] = dy_ego / 20.0

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

        # Traffic light state (one-hot: none=0, green=1, yellow=2, red=3)
        tl_state = 0.0
        if self.vehicle:
            if self.vehicle.is_at_traffic_light():
                traffic_light = self.vehicle.get_traffic_light()
                if traffic_light:
                    state = traffic_light.get_state()
                    if state == carla.TrafficLightState.Green:
                        tl_state = 1.0
                    elif state == carla.TrafficLightState.Yellow:
                        tl_state = 2.0
                    elif state == carla.TrafficLightState.Red:
                        tl_state = 3.0

        # Route completion percentage
        route_pct = self._get_route_completion()

        # Assemble 12-element state vector
        state = np.array([
            speed / MAX_SPEED,             # [0] normalized speed
            prev_steer,                    # [1] previous steering
            prev_throttle,                 # [2] previous throttle/brake
            wp_features[0],                # [3] waypoint 1 dx (ego frame)
            wp_features[1],                # [4] waypoint 1 dy (ego frame)
            wp_features[2],                # [5] waypoint 2 dx
            wp_features[3],                # [6] waypoint 2 dy
            wp_features[4],               # [7] waypoint 3 dx
            wp_features[5],               # [8] waypoint 3 dy
            lane_offset,                   # [9] distance to lane center (normalized)
            tl_state / 3.0,                # [10] traffic light state (normalized)
            route_pct,                     # [11] route completion fraction
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
            tm = self.client.get_trafficmanager(self.port + 6000 + os.getpid() % 1000)
            tm.set_global_distance_to_leading_vehicle(2.5)
            tm.set_synchronous_mode(True)
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
            # require a tick to take effect.
            try:
                self.world.tick()
            except Exception:
                pass

    def close(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
        self._destroy_actors()
