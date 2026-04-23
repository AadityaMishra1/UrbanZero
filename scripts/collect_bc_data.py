"""
scripts/collect_bc_data.py — Behavior Cloning data collection for UrbanZero.

Drives CARLA using BehaviorAgent("normal") and records (obs, action) tuples
as unstacked single frames. VecFrameStack is NOT applied here — stacking is
done offline at BC training time so both can be tuned independently.

Usage:
    python3 scripts/collect_bc_data.py --port 2000 --n_frames 150000

Output .npz structure:
    images  : (N, 1, 128, 128) float32  — raw single-frame semantic seg
    states  : (N, 10) float32           — env state vector
    actions : (N, 2) float32            — expert [steer, throttle_brake]
    meta    : dict with port/seed/n_frames/date
"""

import os
import sys
import argparse
import time
import datetime
import signal

import numpy as np

# Ensure project root is importable.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# -----------------------------------------------------------------------
# tqdm with graceful fallback
# -----------------------------------------------------------------------
try:
    from tqdm import tqdm as _tqdm_cls
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# -----------------------------------------------------------------------
# CARLA BehaviorAgent import helper
# -----------------------------------------------------------------------

def _import_behavior_agent():
    """Import CARLA's BehaviorAgent, handling PYTHONPATH shadowing.

    Our local agents/ directory can shadow CARLA's agents package.
    We replicate the same fallback strategy used in env/carla_env.py.
    """
    carla_paths = [
        os.environ.get("CARLA_PYTHONAPI", ""),
        "/mnt/c/Users/aadit/ECE-591/CARLA_0.9.15/WindowsNoEditor/PythonAPI/carla",
        os.path.expanduser("~/carla/PythonAPI/carla"),
        "/opt/carla/PythonAPI/carla",
    ]
    # Try direct import first
    try:
        from agents.navigation.behavior_agent import BehaviorAgent
        # Sanity-check: CARLA's BehaviorAgent has run_step()
        if hasattr(BehaviorAgent, "run_step"):
            return BehaviorAgent
    except (ImportError, AttributeError):
        pass

    # Fallback: prepend known CARLA paths
    for p in carla_paths:
        if p and os.path.isdir(os.path.join(p, "agents", "navigation")):
            sys.path.insert(0, p)
            for mod_key in list(sys.modules.keys()):
                if mod_key == "agents" or mod_key.startswith("agents."):
                    del sys.modules[mod_key]
            try:
                from agents.navigation.behavior_agent import BehaviorAgent
                if hasattr(BehaviorAgent, "run_step"):
                    return BehaviorAgent
            except (ImportError, AttributeError):
                continue

    raise ImportError(
        "Could not import CARLA BehaviorAgent. "
        "Ensure CARLA PythonAPI is on PYTHONPATH: "
        "export PYTHONPATH=$PYTHONPATH:/path/to/CARLA/PythonAPI/carla"
    )


# -----------------------------------------------------------------------
# Action conversion: carla.VehicleControl -> env action space
# -----------------------------------------------------------------------

def control_to_action(control: "carla.VehicleControl", rng: np.random.Generator) -> np.ndarray:
    """Convert a CARLA VehicleControl to our env's 2D action space.

    Action space:
        action[0] = steer         in [-1, 1]
        action[1] = throttle_brake in [-1, 1]
            0.0  -> throttle = 0.3 (idle creep, env step() adds +0.3 bias)
           -0.3  -> coast
           <-0.3 -> brake
            >0.0 -> throttle > 0.3

    The idle-creep shift in env.step() is: shifted = action[1] + 0.3.
    Inverting: action[1] = throttle - 0.3  (for pure throttle path)
               action[1] = -brake - 0.3    (for pure brake path)

    Steer noise: Gaussian sigma=0.1 added for robustness (LBC DAgger-style,
    Ross & Bagnell 2010 "A Reduction of Imitation Learning...").
    """
    steer = float(np.clip(control.steer, -1.0, 1.0))

    if control.brake > 0.01:
        # Braking: invert bias, clip to [-1, -0.3] range
        throttle_brake = float(np.clip(-control.brake - 0.3, -1.0, -0.3))
    elif control.throttle > 0.01:
        # Throttle: subtract idle-creep offset so env.step() reconstructs correctly.
        # np.clip lower==-1 upper==1 — the original prompt has a typo (clip(x,1,1));
        # the semantically correct bound is [-1, 1].
        throttle_brake = float(np.clip(control.throttle - 0.3, -1.0, 1.0))
    else:
        # Neither throttle nor brake -> coast (shifted=0 means throttle=0.3,
        # but BehaviorAgent coast intent should map to action[1]=-0.3 so
        # env produces throttle=0, brake=0).
        throttle_brake = -0.3

    # Steer noise for robustness (DAgger-style perturbation)
    steer += float(rng.normal(0.0, 0.1))
    steer = float(np.clip(steer, -1.0, 1.0))

    return np.array([steer, throttle_brake], dtype=np.float32)


# -----------------------------------------------------------------------
# Main collection loop
# -----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect BC data from CARLA BehaviorAgent."
    )
    parser.add_argument("--port", type=int, default=2000,
                        help="CARLA RPC port (default: 2000)")
    parser.add_argument("--n_frames", type=int, default=150_000,
                        help="Number of (obs, action) frames to collect (default: 150000)")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    default_out = os.path.expanduser(f"~/urbanzero/bc_data/bc_data_{ts}.npz")
    parser.add_argument("--output", type=str, default=default_out,
                        help="Output .npz path")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for steer noise (default: 42)")
    return parser.parse_args()


def _make_fresh_agent(BehaviorAgent, env):
    """Construct a BehaviorAgent for env.vehicle and set destination.

    Returns the agent or raises on failure.
    """
    agent = BehaviorAgent(env.vehicle, behavior="normal")
    destination = env.route[-1].transform.location
    agent.set_destination(destination)
    return agent


def main() -> None:
    args = parse_args()
    output_path = os.path.expanduser(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Import CARLA BehaviorAgent
    BehaviorAgent = _import_behavior_agent()
    print(f"[BC-collect] BehaviorAgent imported OK")

    # Import CarlaEnv after path setup
    from env.carla_env import CarlaEnv

    print(f"[BC-collect] Connecting to CARLA on port {args.port} ...")
    env = CarlaEnv(
        port=args.port,
        enable_traffic=True,
        enable_weather_randomization=True,
    )
    print(f"[BC-collect] CarlaEnv ready. Collecting {args.n_frames:,} frames -> {output_path}")

    # Pre-allocate buffers
    images_buf    = np.zeros((args.n_frames, 1, 128, 128), dtype=np.float32)
    states_buf    = np.zeros((args.n_frames, 10),           dtype=np.float32)
    actions_buf   = np.zeros((args.n_frames, 2),            dtype=np.float32)
    # episode_starts_buf[i] = True if frame i is the FIRST frame of an episode
    # (immediately after reset). BC trainer uses this to avoid stacking frames
    # across episode boundaries, which would contaminate the first 3 frames of
    # each episode with stale prior-episode content.
    episode_starts_buf = np.zeros(args.n_frames, dtype=bool)

    n_collected = 0
    n_episodes  = 0
    t_start     = time.time()
    last_save_at = 0  # last frame count at which we flushed to disk

    # Graceful Ctrl-C: flip flag, save partial data, exit
    _interrupted = False
    def _sigint_handler(sig, frame):
        nonlocal _interrupted
        print("\n[BC-collect] Ctrl-C received — will save partial data and exit.")
        _interrupted = True
    signal.signal(signal.SIGINT, _sigint_handler)

    # ------------------------------------------------------------------
    # Episode loop
    # ------------------------------------------------------------------
    obs, _ = env.reset()
    n_episodes += 1
    # Next recorded frame is the first of a new episode
    _new_episode = True

    try:
        agent = _make_fresh_agent(BehaviorAgent, env)
    except Exception as e:
        print(f"[BC-collect] WARNING: initial BehaviorAgent creation failed: {e}")
        agent = None

    while n_collected < args.n_frames and not _interrupted:
        # Record obs BEFORE action (matches supervised-learning convention:
        # the label is "what expert did when it saw this obs").
        images_buf[n_collected]  = obs["image"]
        states_buf[n_collected]  = obs["state"]
        if _new_episode:
            episode_starts_buf[n_collected] = True
            _new_episode = False

        # Compute expert action
        action = None
        if agent is not None:
            try:
                control = agent.run_step()
                action  = control_to_action(control, rng)
            except Exception as e:
                print(f"[BC-collect] WARNING: agent.run_step() raised: {e}. Resetting.")
                agent = None

        if action is None:
            # Fallback: gentle forward action (not stored — reset instead)
            obs, _ = env.reset()
            n_episodes += 1
            _new_episode = True
            try:
                agent = _make_fresh_agent(BehaviorAgent, env)
            except Exception as e2:
                print(f"[BC-collect] WARNING: BehaviorAgent re-init failed: {e2}")
                agent = None
            continue

        actions_buf[n_collected] = action
        n_collected += 1

        # Step the env
        obs, _reward, terminated, truncated, _info = env.step(action)

        # Progress print
        if n_collected % 1000 == 0:
            elapsed = time.time() - t_start
            fps = n_collected / max(elapsed, 1e-6)
            print(f"[BC-collect] {n_collected}/{args.n_frames} frames, "
                  f"FPS={fps:.1f}, episodes={n_episodes}")

        # Periodic incremental save
        if n_collected - last_save_at >= 10_000:
            _save_partial(
                output_path, images_buf, states_buf, actions_buf,
                episode_starts_buf, n_collected, args,
                label="incremental",
            )
            last_save_at = n_collected

        # Episode boundary
        if terminated or truncated:
            obs, _ = env.reset()
            n_episodes += 1
            _new_episode = True
            try:
                agent = _make_fresh_agent(BehaviorAgent, env)
            except Exception as e:
                print(f"[BC-collect] WARNING: BehaviorAgent re-init after reset failed: {e}")
                agent = None

    # Final save
    _save_partial(
        output_path, images_buf, states_buf, actions_buf,
        episode_starts_buf, n_collected, args, label="final",
    )

    elapsed_total = time.time() - t_start
    print(f"[BC-collect] Done. {n_collected:,} frames in "
          f"{elapsed_total/60:.1f} min across {n_episodes} episodes. "
          f"Saved -> {output_path}")

    try:
        env.close()
    except Exception:
        pass


def _save_partial(
    path: str,
    images_buf: np.ndarray,
    states_buf: np.ndarray,
    actions_buf: np.ndarray,
    episode_starts_buf: np.ndarray,
    n: int,
    args: argparse.Namespace,
    label: str = "",
) -> None:
    """Save collected data to .npz, truncated to n frames."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta = {
        "port":     args.port,
        "seed":     args.seed,
        "n_frames": n,
        "date":     datetime.datetime.now().isoformat(),
        "label":    label,
    }
    np.savez_compressed(
        path,
        images=images_buf[:n],
        states=states_buf[:n],
        actions=actions_buf[:n],
        episode_starts=episode_starts_buf[:n],
        # Store meta as a length-1 object array so np.load can recover it
        meta=np.array([meta], dtype=object),
    )
    print(f"[BC-collect] {label} save: {n:,} frames -> {path}")


if __name__ == "__main__":
    main()
