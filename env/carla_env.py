import carla
import gymnasium as gym
import numpy as np
import os
import time

CARLA_HOST = os.environ.get("CARLA_HOST", "172.25.176.1")
CARLA_PORT = 2000
IMG_W = 84
IMG_H = 84

class CarlaEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Semantic seg image + vehicle state vector [speed, steer, waypoint_angle]
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(low=0, high=255, shape=(3, IMG_H, IMG_W), dtype=np.uint8),
            "state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        })
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.client = carla.Client(CARLA_HOST, CARLA_PORT)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        self.image = None
        self.collision_history = []
        self.step_count = 0

    def reset(self, seed=None, options=None):
        self._destroy_actors()
        self.collision_history = []
        self.image = None
        self.step_count = 0
        bp = self.blueprint_library.filter("vehicle.tesla.model3")[0]
        spawn_points = self.map.get_spawn_points()
        np.random.shuffle(spawn_points)
        self.vehicle = None
        for sp in spawn_points:
            try:
                self.vehicle = self.world.spawn_actor(bp, sp)
                break
            except:
                continue
        if self.vehicle is None:
            raise RuntimeError("No free spawn point found")

        # Semantic segmentation camera
        cam_bp = self.blueprint_library.find("sensor.camera.semantic_segmentation")
        cam_bp.set_attribute("image_size_x", str(IMG_W))
        cam_bp.set_attribute("image_size_y", str(IMG_H))
        cam_bp.set_attribute("fov", "90")
        cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(cam_bp, cam_transform, attach_to=self.vehicle)
        self.camera.listen(self._on_image)

        col_bp = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(col_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda e: self.collision_history.append(e))

        for _ in range(20):
            self.world.tick()
            if self.image is not None:
                break
            time.sleep(0.01)

        self.start_location = self.vehicle.get_location()
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
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
        reward, terminated = self._compute_reward()
        truncated = self.step_count >= 1000
        return obs, reward, terminated, truncated, {}

    def _compute_reward(self):
        vel = self.vehicle.get_velocity()
        speed = (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5

        # Waypoint angle reward — steer toward road direction
        waypoint = self.map.get_waypoint(self.vehicle.get_location())
        vehicle_yaw = self.vehicle.get_transform().rotation.yaw
        waypoint_yaw = waypoint.transform.rotation.yaw
        angle_diff = abs(vehicle_yaw - waypoint_yaw) % 360
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        angle_reward = 1.0 - (angle_diff / 180.0)

        reward = speed * 0.5 + angle_reward * 0.3

        terminated = False
        if len(self.collision_history) > 0:
            reward -= 50.0
            terminated = True
        elif speed < 0.1 and self.step_count > 20:
            reward -= 0.5

        return reward, terminated

    def _get_obs(self):
        vel = self.vehicle.get_velocity() if self.vehicle else None
        speed = (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5 if vel else 0.0
        control = self.vehicle.get_control() if self.vehicle else None
        steer = control.steer if control else 0.0
        waypoint = self.map.get_waypoint(self.vehicle.get_location()) if self.vehicle else None
        vehicle_yaw = self.vehicle.get_transform().rotation.yaw if self.vehicle else 0.0
        waypoint_yaw = waypoint.transform.rotation.yaw if waypoint else 0.0
        angle_diff = (vehicle_yaw - waypoint_yaw) % 360
        if angle_diff > 180:
            angle_diff = angle_diff - 360
        state = np.array([speed / 30.0, steer, angle_diff / 180.0], dtype=np.float32)
        image = self.image if self.image is not None else np.zeros((3, IMG_H, IMG_W), dtype=np.uint8)
        return {"image": image, "state": state}

    def _on_image(self, image):
        # Semantic seg returns class labels in R channel, convert to color-coded RGB
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((IMG_H, IMG_W, 4))
        # Use semantic class index as grayscale across all 3 channels
        labels = array[:, :, 2]  # semantic class in B channel
        rgb = np.stack([labels, labels, labels], axis=0)
        self.image = rgb

    def _destroy_actors(self):
        for actor in [self.camera, self.collision_sensor, self.vehicle]:
            if actor is not None:
                try:
                    actor.destroy()
                except:
                    pass
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None

    def close(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
        self._destroy_actors()
