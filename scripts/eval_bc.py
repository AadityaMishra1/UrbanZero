"""Deterministic evaluation of a frozen checkpoint (BC or PPO).

Final deliverable script for the nuclear fallback: after six
consecutive failed PPO finetune attempts, the frozen BC policy from
`bc_pretrain.zip` is the strongest baseline we have. Individual
episodes during failed PPO runs showed the BC policy driving at
30-38% RC before PPO erosion; deterministic eval should confirm
that signal holds across many runs.

Usage:
  python3 scripts/eval_bc.py \
    --model ~/urbanzero/checkpoints/bc_pretrain.zip \
    --episodes 20 \
    --port 2000 \
    --output ~/urbanzero/eval/bc_only_eval.json

Per-run behavior:
  - Loads the SB3 policy from the .zip via PPO.load()
  - Constructs the same vec-env pipeline train.py uses
    (DummyVecEnv -> VecFrameStack -> VecNormalize norm_obs=False)
  - Runs N episodes deterministically (deterministic=True in
    policy.predict, ignoring log_std entirely)
  - Logs per-episode: termination_reason, route_completion,
    episode_length, avg_speed, collision count
  - Aggregates: mean RC, median RC, max RC, %ROUTE_COMPLETE,
    %COLLISION, %OFF_ROUTE, %REALLY_STUCK
  - Writes results to --output JSON

To run on multiple towns, re-run the script with a different
CARLA server that has a different default map, or modify the
CarlaEnv construction below to explicitly load the requested map.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deterministic policy eval")
    p.add_argument("--model", type=str, required=True,
                   help="Path to SB3 .zip policy (bc_pretrain.zip or any PPO checkpoint)")
    p.add_argument("--episodes", type=int, default=20,
                   help="Number of deterministic episodes (default: 20)")
    p.add_argument("--port", type=int, default=2000,
                   help="CARLA RPC port (default: 2000)")
    p.add_argument("--seed", type=int, default=1001,
                   help="RNG seed for route / spawn selection (default: 1001)")
    p.add_argument("--output", type=str,
                   default=os.path.expanduser("~/urbanzero/eval/bc_only_eval.json"),
                   help="Output JSON path")
    p.add_argument("--no-traffic", action="store_true",
                   help="Disable NPC traffic (cleaner for demo)")
    p.add_argument("--max-steps", type=int, default=3000,
                   help="Max steps per episode (default: 3000 = 150s)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_path = os.path.expanduser(args.model)
    output_path = os.path.expanduser(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Lazy imports so --help works without a full CARLA install
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from env.carla_env import CarlaEnv
    from env.safety_wrapper import NaNGuardWrapper
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import (
        DummyVecEnv, VecFrameStack, VecNormalize,
    )

    print(f"[eval] model={model_path}")
    print(f"[eval] episodes={args.episodes}  port={args.port}  seed={args.seed}")

    # Build env pipeline identical to training (so observation shapes match).
    # Single env — deterministic eval doesn't benefit from parallelism.
    def _make_env():
        env = CarlaEnv(
            port=args.port,
            enable_traffic=not args.no_traffic,
            enable_weather_randomization=True,
            max_episode_steps=args.max_steps,
        )
        return NaNGuardWrapper(env)

    vec_env = DummyVecEnv([_make_env])
    vec_env = VecFrameStack(
        vec_env, n_stack=4,
        channels_order={"image": "first", "state": "last"},
    )
    vec_env = VecNormalize(
        vec_env, norm_obs=False, norm_reward=True,
        clip_reward=10.0, clip_obs=10.0,
    )
    # Freeze the VecNormalize running stats — we're evaluating, not training.
    vec_env.training = False
    vec_env.norm_reward = False  # we want raw rewards for reporting

    # Load policy
    model = PPO.load(
        model_path,
        env=vec_env,
        device="cuda" if _cuda_available() else "cpu",
    )
    print(f"[eval] model loaded. policy_std (training value): "
          f"{_get_policy_std(model)}")

    # Run episodes
    episodes = []
    t_start = time.time()

    for ep_idx in range(args.episodes):
        obs = vec_env.reset()
        done = np.array([False])
        total_reward = 0.0
        ep_steps = 0
        ep_speeds = []
        ep_info = {}

        while not done[0]:
            # deterministic=True uses the mean action, ignoring log_std.
            # This is what you want for demo / eval — removes the sampling
            # noise that might push a confident BC policy off the road.
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info_list = vec_env.step(action)
            total_reward += float(reward[0])
            ep_steps += 1
            info = info_list[0]
            ep_speeds.append(info.get("speed", 0.0))
            ep_info = info  # keep last

        reason = ep_info.get("termination_reason", "UNKNOWN")
        rc = ep_info.get("route_completion", 0.0)
        collisions = ep_info.get("collisions", 0)
        avg_speed = float(np.mean(ep_speeds)) if ep_speeds else 0.0

        ep_record = {
            "index": ep_idx,
            "steps": ep_steps,
            "duration_s": ep_steps * 0.05,
            "route_completion": rc,
            "termination_reason": reason,
            "total_reward": total_reward,
            "collisions": collisions,
            "avg_speed_ms": avg_speed,
        }
        episodes.append(ep_record)
        print(f"[eval] ep {ep_idx+1:02d}/{args.episodes}: "
              f"reason={reason:<14} RC={rc*100:5.1f}%  "
              f"steps={ep_steps:>4}  speed={avg_speed:.2f} m/s  "
              f"reward={total_reward:+.1f}")

    vec_env.close()
    elapsed = time.time() - t_start

    # Aggregate
    rcs = [e["route_completion"] for e in episodes]
    reasons = [e["termination_reason"] for e in episodes]
    n = len(episodes)
    aggregate = {
        "n_episodes": n,
        "model_path": model_path,
        "wall_clock_s": elapsed,
        "rc_mean": float(np.mean(rcs)),
        "rc_median": float(np.median(rcs)),
        "rc_max": float(np.max(rcs)),
        "rc_min": float(np.min(rcs)),
        "rc_std": float(np.std(rcs)),
        "pct_route_complete": 100 * sum(1 for r in reasons if r == "ROUTE_COMPLETE") / n,
        "pct_collision":      100 * sum(1 for r in reasons if r == "COLLISION") / n,
        "pct_off_route":      100 * sum(1 for r in reasons if r == "OFF_ROUTE") / n,
        "pct_really_stuck":   100 * sum(1 for r in reasons if r == "REALLY_STUCK") / n,
        "pct_max_steps":      100 * sum(1 for r in reasons if r == "MAX_STEPS") / n,
        "avg_speed_ms":       float(np.mean([e["avg_speed_ms"] for e in episodes])),
    }

    result = {
        "timestamp": datetime.datetime.now().isoformat(),
        "args": vars(args),
        "aggregate": aggregate,
        "episodes": episodes,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print()
    print(f"{'=' * 60}")
    print(f"EVALUATION SUMMARY — {n} episodes, port {args.port}")
    print(f"{'=' * 60}")
    print(f"  RC mean:          {aggregate['rc_mean']*100:.2f}%")
    print(f"  RC median:        {aggregate['rc_median']*100:.2f}%")
    print(f"  RC max:           {aggregate['rc_max']*100:.2f}%")
    print(f"  RC std:           {aggregate['rc_std']*100:.2f}%")
    print(f"  %ROUTE_COMPLETE:  {aggregate['pct_route_complete']:.1f}%")
    print(f"  %COLLISION:       {aggregate['pct_collision']:.1f}%")
    print(f"  %OFF_ROUTE:       {aggregate['pct_off_route']:.1f}%")
    print(f"  %REALLY_STUCK:    {aggregate['pct_really_stuck']:.1f}%")
    print(f"  avg speed:        {aggregate['avg_speed_ms']:.2f} m/s")
    print(f"  wall clock:       {elapsed/60:.1f} min")
    print()
    print(f"Full results -> {output_path}")


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _get_policy_std(model):
    try:
        import torch
        return torch.exp(model.policy.log_std.detach().cpu()).tolist()
    except Exception:
        return "?"


if __name__ == "__main__":
    main()
