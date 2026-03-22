"""Bird's-eye spectator camera that follows the ego Tesla Model 3."""
import carla
import os
import time

client = carla.Client(os.environ.get("CARLA_HOST", "172.25.176.1"), 2000)
client.set_timeout(10.0)
world = client.get_world()
spectator = world.get_spectator()

print("Spectator following ego vehicle...")
while True:
    vehicles = world.get_actors().filter("vehicle.tesla.model3")
    if vehicles:
        v = list(vehicles)[0]
        loc = v.get_location()
        spectator.set_transform(carla.Transform(
            carla.Location(x=loc.x, y=loc.y, z=loc.z + 40),
            carla.Rotation(pitch=-90),
        ))
    time.sleep(0.05)
