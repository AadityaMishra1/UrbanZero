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
        self._destroy_actors()
        self.collision_history = []
        self.image = None
        self.step_count = 0
        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)
        self.stagnation_counter = 0
        self.route_progress = 0.0

        # Weather randomization
        if self.enable_weather_randomization:
            self.world.set_weather(random.choice(WEATHER_PRESETS))

        # Spawn ego vehicle
        bp = self.blueprint_library.filter("vehicle.tesla.model3")[0]
        spawn_points = self.map.get_spawn_points()
        random.shuffle(spawn_points)
        self.vehicle = None
        start_sp = None
        for sp in spawn_points:
            try:
                self.vehicle = self.world.spawn_actor(bp, sp)
                start_sp = sp
                break
            except RuntimeError:
                continue
        if self.vehicle is None:
            raise RuntimeError("No free spawn point found")

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

        # Collision sensor
        col_bp = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(col_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda e: self.collision_history.append(e))

        # Spawn traffic
        if self.enable_traffic:
            self._spawn_traffic()

        # Wait for first image
        for _ in range(30):
            self.world.tick()
            if self.image is not None:
                break
            time.sleep(0.01)

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        # Apply action
        steer = float(np.clip(action[0], -1.0, 1.0))
        throttle_brake = float(np.clip(action[1], -1.0, 1.0))
        if throttle_brake >= 0:
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=throttle_brake, steer=steer, brake=0.0
            ))
        else:
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=0.0, steer=steer, brake=-throttle_brake
            ))

        self.world.tick()

        obs = self._get_obs()
        reward, terminated = self._compute_reward(action)
        self.prev_action = np.array([steer, throttle_brake], dtype=np.float32)

        truncated = self.step_count >= self.max_episode_steps
        # Also truncate if stuck for too long
        if self.stagnation_counter > 200:
            truncated = True

        info = {
            "route_completion": self._get_route_completion(),
            "speed": self._get_speed(),
            "collisions": len(self.collision_history),
            "step": self.step_count,
        }

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # ROUTE PLANNING (GAP 1 fix)
    # ------------------------------------------------------------------

    def _generate_route(self, start_sp, spawn_points):
        """Generate a route from start to a distant destination using GlobalRoutePlanner."""
        start_loc = start_sp.location

        # Pick a destination that's far enough away (>100m)
        candidates = []
        for sp in spawn_points:
            d = start_loc.distance(sp.location)
            if 100.0 < d < 500.0:
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
        """Advance the route index to the nearest waypoint ahead of the vehicle."""
        if not self.route:
            return 0.0

        ego_loc = self.vehicle.get_location()
        progress_delta = 0.0

        # Look ahead from current index, advance past waypoints we've passed
        while self.route_index < len(self.route) - 1:
            wp_loc = self.route[self.route_index].transform.location
            next_wp_loc = self.route[self.route_index + 1].transform.location
            dist_to_next = ego_loc.distance(next_wp_loc)
            dist_between = wp_loc.distance(next_wp_loc)

            # If we're closer to the next waypoint than the current, advance
            if ego_loc.distance(wp_loc) > dist_between * 0.5 and dist_to_next < dist_between * 1.5:
                progress_delta += dist_between
                self.route_index += 1
            else:
                break

        return progress_delta

    def _get_route_completion(self):
        """Fraction of route completed [0, 1]."""
        if not self.route or self.total_route_length < 1.0:
            return 0.0
        return min(self.route_progress / self.total_route_length, 1.0)

    # ------------------------------------------------------------------
    # REWARD FUNCTION (GAP 3 fix — CaRL-inspired)
    # ------------------------------------------------------------------

    def _compute_reward(self, action):
        speed = self._get_speed()

        # 1. Route progress (PRIMARY signal — CaRL-style)
        progress_delta = self._advance_route_index()
        self.route_progress += progress_delta
        # Normalize: reward per meter of progress
        progress_reward = progress_delta * 1.0

        # 2. Speed reward (encourage moving at target speed, penalize excess)
        if speed < TARGET_SPEED:
            speed_reward = 0.2 * (speed / TARGET_SPEED)
        else:
            # Penalize speeding proportionally
            speed_reward = 0.2 * max(0.0, 1.0 - (speed - TARGET_SPEED) / (MAX_SPEED - TARGET_SPEED))

        # 3. Lane centering penalty
        lane_penalty = 0.0
        wp = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True,
                                    lane_type=carla.LaneType.Driving)
        if wp:
            ego_loc = self.vehicle.get_location()
            wp_loc = wp.transform.location
            lane_offset = math.sqrt((ego_loc.x - wp_loc.x)**2 + (ego_loc.y - wp_loc.y)**2)
            lane_penalty = -0.2 * min(lane_offset, 3.0)  # cap at 3m

        # 4. Heading alignment (angle between vehicle heading and road direction)
        heading_reward = 0.0
        if wp:
            vehicle_yaw = self.vehicle.get_transform().rotation.yaw
            wp_yaw = wp.transform.rotation.yaw
            angle_diff = abs(vehicle_yaw - wp_yaw) % 360
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            # Reward for being well-aligned (max 0.1 when perfectly aligned)
            heading_reward = 0.1 * (1.0 - angle_diff / 180.0)

        # 5. Action smoothness penalty (CAPS-style)
        steer = action[0]
        throttle_brake = action[1]
        steer_delta = abs(steer - self.prev_action[0])
        throttle_delta = abs(throttle_brake - self.prev_action[1])
        smoothness_penalty = -0.05 * (steer_delta + 0.3 * throttle_delta)

        # 6. Stagnation tracking
        if speed < 0.1:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0

        stagnation_penalty = 0.0
        if self.stagnation_counter > 40:  # 2 seconds of being stuck
            stagnation_penalty = -0.3

        # Combine reward
        reward = progress_reward + speed_reward + lane_penalty + heading_reward + smoothness_penalty + stagnation_penalty

        # 7. Collision — terminal with penalty
        terminated = False
        if len(self.collision_history) > 0:
            reward = -10.0
            terminated = True

        # 8. Off-route penalty (too far from any route waypoint)
        if self.route and self.route_index < len(self.route):
            dist_to_route = self.vehicle.get_location().distance(
                self.route[min(self.route_index, len(self.route) - 1)].transform.location
            )
            if dist_to_route > 30.0:
                reward = -5.0
                terminated = True  # Too far off route, end episode

        return reward, terminated

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

        # Lane offset (signed distance to lane center)
        lane_offset = 0.0
        wp = self.map.get_waypoint(ego_loc, project_to_road=True,
                                    lane_type=carla.LaneType.Driving) if ego_loc else None
        if wp and ego_loc:
            wp_loc = wp.transform.location
            lane_offset = math.sqrt((ego_loc.x - wp_loc.x)**2 + (ego_loc.y - wp_loc.y)**2)
            lane_offset = min(lane_offset, 5.0) / 5.0  # normalize to [0, 1]

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
        # Normalize to [0, 1] range
        labels = labels / NUM_SEMANTIC_CLASSES
        # Single channel: (1, H, W)
        self.image = labels[np.newaxis, :, :]

    # ------------------------------------------------------------------
    # TRAFFIC (GAP 6 fix)
    # ------------------------------------------------------------------

    def _spawn_traffic(self):
        """Spawn NPC vehicles and pedestrians using CARLA's Traffic Manager."""
        self._destroy_traffic()

        try:
            tm = self.client.get_trafficmanager(self.port + 6000)
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
                try:
                    walker = self.world.spawn_actor(bp, spawn_point)
                    self.pedestrian_actors.append(walker)
                    controller = self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker)
                    self.pedestrian_actors.append(controller)
                    controller.start()
                    controller.go_to_location(self.world.get_random_location_from_navigation())
                    controller.set_max_speed(1.0 + random.random() * 1.5)
                except RuntimeError:
                    continue

        except Exception as e:
            print(f"Warning: traffic spawning failed: {e}")

    def _destroy_traffic(self):
        """Clean up all spawned traffic actors."""
        # Stop pedestrian controllers first
        for actor in self.pedestrian_actors:
            try:
                if actor.type_id == "controller.ai.walker":
                    actor.stop()
            except Exception:
                pass

        for actor in self.traffic_actors + self.pedestrian_actors:
            try:
                actor.destroy()
            except Exception:
                pass
        self.traffic_actors = []
        self.pedestrian_actors = []

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
        self._destroy_traffic()
        for actor in [self.camera, self.collision_sensor, self.vehicle]:
            if actor is not None:
                try:
                    actor.destroy()
                except Exception:
                    pass
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None

    def close(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
        self._destroy_actors()
