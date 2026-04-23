"""Standalone sanity check: are NPCs actually moving in CARLA 0.9.15 sync mode?

Per external reviewer's follow-up question — before burning training
compute on v5, verify the TM/NPC plumbing works in a minimal script
outside the training loop. If NPCs are frozen here, they're frozen in
training.

Tests the reviewer's specific hypothesis: "Traffic Manager Port
Collision. Even if you think they are separate, if one env tries to
talk to the other's TM, or if the tm_port wasn't explicitly passed to
the set_autopilot call, the command just vanishes into the ether."

Usage:
  python3 scripts/sanity_check_npcs.py --port 2000 --n_npcs 10 --ticks 60

  # Test both ports in sequence to isolate port collision:
  python3 scripts/sanity_check_npcs.py --port 2000
  python3 scripts/sanity_check_npcs.py --port 3000

  # Run TWO instances simultaneously (mimics DummyVecEnv sync):
  python3 scripts/sanity_check_npcs.py --port 2000 &
  python3 scripts/sanity_check_npcs.py --port 3000 &
  wait

Exit codes:
  0  — NPCs moved (avg_speed > 0.5 m/s across sampled vehicles)
  1  — NPCs frozen (status=FROZEN after N ticks)
  2  — Setup failure (couldn't connect / spawn / configure)
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify NPC motion in CARLA 0.9.15 sync mode (pre-training sanity check)"
    )
    p.add_argument("--host", type=str,
                   default=os.environ.get("CARLA_HOST", "172.25.176.1"),
                   help="CARLA server host (default: $CARLA_HOST or 172.25.176.1)")
    p.add_argument("--port", type=int, default=2000,
                   help="CARLA RPC port (default: 2000)")
    p.add_argument("--tm_port", type=int, default=None,
                   help="Explicit TM port. If not given, uses port+6000+pid%%1000 "
                        "matching env/carla_env.py's formula.")
    p.add_argument("--n_npcs", type=int, default=10,
                   help="Number of NPCs to spawn (default: 10)")
    p.add_argument("--ticks", type=int, default=60,
                   help="World ticks to run before measuring motion (default: 60 = 3s sim time)")
    p.add_argument("--hybrid_physics", action="store_true", default=False,
                   help="Enable TM hybrid physics mode. Default OFF — issue #12 "
                        "showed NPCs frozen even solo, and hybrid_physics requires "
                        "a hero vehicle within radius or all NPCs go dormant.")
    p.add_argument("--speed_diff", type=float, default=-30.0,
                   help="Global percentage speed difference (-30 = drive 30%% over limit)")
    p.add_argument("--no_sync", action="store_true",
                   help="Skip sync mode (for comparing async vs sync behavior)")
    p.add_argument("--verbose", action="store_true",
                   help="Per-NPC position/velocity logging")
    p.add_argument("--spawn_ego", action="store_true", default=False,
                   help="Spawn an idle ego vehicle before NPCs. Hybrid physics "
                        "requires a hero — without one, ALL NPCs fall into "
                        "dormant state regardless of radius.")
    p.add_argument("--load_map", type=str, default=None,
                   help="If set, force-load this map before the test (e.g. Town01). "
                        "The issue #12 report speculated about Town10HD_Opt being "
                        "the default; load_world('Town01') rules that out.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import carla
    except ImportError:
        print("ERROR: carla Python API not importable. Check CARLA_PYTHONAPI.",
              file=sys.stderr)
        return 2

    pid = os.getpid()
    tm_port = args.tm_port if args.tm_port is not None else args.port + 6000 + pid % 1000

    print(f"[sanity] host={args.host} port={args.port} tm_port={tm_port} "
          f"pid={pid} ticks={args.ticks} n_npcs={args.n_npcs}")

    # Connect
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        if args.load_map:
            print(f"[sanity] loading map '{args.load_map}' (this takes 5-10s) ...")
            world = client.load_world(args.load_map)
        else:
            world = client.get_world()
        settings_initial = world.get_settings()
        # Map-name print per issue #12 lead. The CARLA 0.9.15 default is
        # Town10HD_Opt; if we've been running there all along, TM path
        # generation might be the missing piece.
        try:
            map_name = world.get_map().name
        except Exception:
            map_name = "?"
        print(f"[sanity] connected. map='{map_name}'  "
              f"world.sync (initial) = {settings_initial.synchronous_mode}")
    except Exception as e:
        print(f"[sanity] FAILED to connect: {e}", file=sys.stderr)
        return 2

    try:
        # Apply sync mode (matches env/carla_env.py line 119-121)
        if not args.no_sync:
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)
            print(f"[sanity] world.sync set to True")

        # Create TM (matches env/carla_env.py _spawn_traffic)
        tm = client.get_trafficmanager(tm_port)
        tm.set_global_distance_to_leading_vehicle(2.5)
        tm.set_synchronous_mode(not args.no_sync)
        tm.set_random_device_seed(random.randint(0, 10000))
        if args.hybrid_physics:
            try:
                tm.set_hybrid_physics_mode(True)
                tm.set_hybrid_physics_radius(70.0)
            except AttributeError:
                print("[sanity] NOTE: hybrid_physics API not available")

        print(f"[sanity] TM created on port={tm_port}, sync={not args.no_sync}, "
              f"hybrid_physics={args.hybrid_physics}")
        print(f"[sanity] tm.get_port() reports = {tm.get_port()}")
        if tm.get_port() != tm_port:
            print(f"[sanity] WARNING: tm.get_port()={tm.get_port()} != requested tm_port={tm_port}")
            print(f"[sanity] This is the reviewer-predicted PORT COLLISION bug. "
                  f"set_autopilot() would bind to the wrong TM.")

        # Spawn NPCs on random driving-lane spawn points
        bps = world.get_blueprint_library().filter("vehicle.*")
        bps = [bp for bp in bps if int(bp.get_attribute("number_of_wheels")) >= 4]
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        # Optional ego vehicle spawn (issue #12 hypothesis).
        # TM's hybrid_physics_mode puts NPCs outside the hero's radius into a
        # dormant state. With no hero, ALL NPCs are dormant regardless of
        # radius. The training env has an ego; the original sanity check
        # didn't. Spawn a stationary ego at the first spawn point to test.
        ego = None
        if args.spawn_ego and spawn_points:
            try:
                ego_bp = random.choice(bps)
                ego = world.spawn_actor(ego_bp, spawn_points[0])
                spawn_points = spawn_points[1:]  # reserved for ego
                print(f"[sanity] spawned idle ego id={ego.id} "
                      f"at ({spawn_points[0].location.x:.1f},"
                      f"{spawn_points[0].location.y:.1f}) "
                      f"(hero for hybrid_physics radius check)")
            except Exception as e:
                print(f"[sanity] ego spawn failed: {e}")

        npcs = []
        for sp in spawn_points:
            if len(npcs) >= args.n_npcs:
                break
            bp = random.choice(bps)
            if bp.has_attribute("color"):
                bp.set_attribute("color",
                                 random.choice(bp.get_attribute("color").recommended_values))
            try:
                npc = world.spawn_actor(bp, sp)
                npcs.append(npc)
            except RuntimeError:
                continue
        print(f"[sanity] spawned {len(npcs)} NPCs")

        if not npcs:
            print("[sanity] FAIL: no NPCs spawned")
            return 2

        # Autopilot registration — reviewer flagged this as the critical step
        for npc in npcs:
            try:
                npc.set_autopilot(True, tm.get_port())
            except Exception as e:
                print(f"[sanity] autopilot register failed on {npc.id}: {e}")

        # Apply speed settings
        try:
            tm.global_percentage_speed_difference(args.speed_diff)
            for npc in npcs:
                tm.vehicle_percentage_speed_difference(npc, -20.0)
                tm.auto_lane_change(npc, True)
                tm.ignore_lights_percentage(npc, 0.0)
        except Exception as e:
            print(f"[sanity] per-vehicle settings partial failure: {e}")

        # Tick the world
        print(f"[sanity] ticking world {args.ticks} times "
              f"(~{args.ticks * 0.05:.1f}s sim time) ...")
        per_tick_speeds = []
        for t in range(args.ticks):
            try:
                if args.no_sync:
                    time.sleep(0.05)
                else:
                    world.tick()
            except Exception as e:
                print(f"[sanity] tick {t} failed: {e}")
                break

            # Every 10 ticks, sample velocities
            if t % 10 == 0 or t == args.ticks - 1:
                speeds = []
                for npc in npcs:
                    try:
                        v = npc.get_velocity()
                        s = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
                        speeds.append(s)
                    except Exception:
                        continue
                if speeds:
                    per_tick_speeds.append(speeds)
                    avg = sum(speeds) / len(speeds)
                    mx = max(speeds)
                    moving = sum(1 for s in speeds if s > 0.1)
                    print(f"[sanity] tick {t:3d}: "
                          f"avg_speed={avg:.2f}m/s max_speed={mx:.2f}m/s "
                          f"moving={moving}/{len(speeds)}")

        # Final verdict
        if per_tick_speeds:
            final_speeds = per_tick_speeds[-1]
            final_avg = sum(final_speeds) / len(final_speeds)
            final_max = max(final_speeds)
            moving_count = sum(1 for s in final_speeds if s > 0.1)

            print()
            print("=" * 60)
            print(f"VERDICT after {args.ticks} ticks:")
            print(f"  Final avg NPC speed: {final_avg:.2f} m/s")
            print(f"  Final max NPC speed: {final_max:.2f} m/s")
            print(f"  NPCs moving (>0.1 m/s): {moving_count}/{len(final_speeds)}")
            print("=" * 60)

            if args.verbose:
                print("\nPer-NPC final velocities:")
                for npc, s in zip(npcs, final_speeds):
                    try:
                        loc = npc.get_location()
                        print(f"  NPC {npc.id}: "
                              f"pos=({loc.x:.1f},{loc.y:.1f}) speed={s:.2f}m/s")
                    except Exception:
                        pass

            if final_avg > 0.5:
                print(f"\n[sanity] PASS: NPCs are moving (avg {final_avg:.2f} m/s > 0.5 threshold)")
                exit_code = 0
            else:
                print(f"\n[sanity] FAIL: NPCs are FROZEN (avg {final_avg:.2f} m/s)")
                print("\nDiagnostic for reviewer:")
                print(f"  - TM port requested: {tm_port}")
                print(f"  - TM port reported:  {tm.get_port()}")
                print(f"  - World sync mode: {world.get_settings().synchronous_mode}")
                print(f"  - NPCs spawned:     {len(npcs)}")
                print(f"  - NPCs visible as actors: "
                      f"{len([a for a in world.get_actors() if a.id in [n.id for n in npcs]])}")
                print("\nPossible causes:")
                print("  1. TM not actually issuing commands (reviewer's 'heartbeat' problem)")
                print("  2. set_autopilot port mismatch (reviewer's port collision)")
                print("  3. Hybrid physics putting NPCs in dormant zone")
                print("  4. Version-specific CARLA 0.9.15 bug")
                exit_code = 1
        else:
            print("[sanity] FAIL: no velocity samples collected")
            exit_code = 2

    finally:
        # Cleanup. Issue #12 note: sequential set_autopilot(False) + destroy()
        # on the same actor can cause a C++ abort on "destroyed actor". Use
        # batch destroy via client.apply_batch_sync to avoid the race.
        print("\n[sanity] cleaning up ...")
        try:
            cmds = [carla.command.DestroyActor(a.id) for a in npcs]
            if ego is not None:
                cmds.append(carla.command.DestroyActor(ego.id))
            if cmds:
                client.apply_batch_sync(cmds, False)
        except Exception:
            # Fallback: best-effort per-actor destroy
            for actor in list(npcs) + ([ego] if ego else []):
                try:
                    actor.destroy()
                except Exception:
                    pass
        # Restore async mode so the CARLA server is usable for other tests
        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
            tm.set_synchronous_mode(False)
        except Exception:
            pass
        print("[sanity] done.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
