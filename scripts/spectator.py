"""Bird's-eye spectator camera that follows the ego Tesla Model 3.

NOTE (GitHub issue #4): This external spectator is NO LONGER auto-launched
by scripts/start_training.sh. Each CARLA server window now follows its own
worker's ego via an inline spectator update inside env/carla_env.py's step(),
so with n_envs>=2 you already see every worker without running this script.
Using this script during training creates a SECONDARY CARLA client on the
target sync-mode server, which is explicitly unsupported (CARLA issues
#1996 / #2239) and contributed to the deadlock in issue #4.

Kept for MANUAL offline debugging (e.g., inspecting a dead env after training
has stopped). Accepts URBANZERO_SPECTATOR_PORT (default 2000) so you can
target a specific CARLA instance:
    URBANZERO_SPECTATOR_PORT=3000 python3 scripts/spectator.py

Crash-resilient: reconnects if the connection drops, survives vehicle
destroy/respawn between episodes.
"""
import carla
import os
import time

HOST = os.environ.get("CARLA_HOST", "172.25.176.1")
PORT = int(os.environ.get("URBANZERO_SPECTATOR_PORT",
                           os.environ.get("CARLA_PORT", "2000")))


def run():
    print(f"Spectator target: {HOST}:{PORT}")
    print("WARNING: external spectator during LIVE training can deadlock the "
          "sync-mode server (CARLA issues #1996 / #2239). Use only for "
          "offline debug. The inline spectator in env/carla_env.py's step() "
          "already follows each worker's ego inside that worker's CARLA window.")
    while True:
        try:
            client = carla.Client(HOST, PORT)
            client.set_timeout(10.0)
            world = client.get_world()
            spectator = world.get_spectator()
            print(f"Spectator connected to {HOST}:{PORT}, following ego vehicle...")

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
