"""Bird's-eye spectator camera that follows the ego Tesla Model 3.

Crash-resilient: reconnects to CARLA if connection drops, survives
vehicle destroy/respawn between episodes.
"""
import carla
import os
import time

HOST = os.environ.get("CARLA_HOST", "172.25.176.1")
PORT = 2000

def run():
    while True:
        try:
            client = carla.Client(HOST, PORT)
            client.set_timeout(10.0)
            world = client.get_world()
            spectator = world.get_spectator()
            print("Spectator connected, following ego vehicle...")

            while True:
                try:
                    vehicles = world.get_actors().filter("vehicle.tesla.model3")
                    if vehicles:
                        v = list(vehicles)[0]
                        loc = v.get_location()
                        spectator.set_transform(carla.Transform(
                            carla.Location(x=loc.x, y=loc.y, z=loc.z + 40),
                            carla.Rotation(pitch=-90),
                        ))
                except Exception:
                    # Vehicle destroyed between episodes — just wait
                    pass
                time.sleep(0.05)

        except Exception as e:
            print(f"Spectator lost connection: {e}, reconnecting in 3s...")
            time.sleep(3)

if __name__ == "__main__":
    run()
