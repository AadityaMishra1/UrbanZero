"""
UrbanZero PPO Training Script (v2 — SOTA-informed)

Key improvements over naive baseline:
- SubprocVecEnv with multiple CARLA instances (configurable)
- VecFrameStack for temporal information (4-frame stacking)
- Custom 5-layer CNN with LayerNorm (replaces NatureCNN)
- 10M+ timestep budget (vs. 2M)
- Driving metrics evaluation callback (route completion, driving score)
- Proper hyperparameter tuning for driving tasks

References:
- CaRL (NVIDIA AVG, CoRL 2025): scalable PPO with simple reward
- Roach (ETH Zurich, ICCV 2021): PPO expert agent for CARLA
- CAPS (ICRA 2021): action smoothness regularization
"""

import os
import sys
import argparse

# Ensure project root is importable.
# IMPORTANT: use sys.path.append (not insert) so CARLA's PythonAPI 'agents'
# package takes priority over our local 'agents/' directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from env.carla_env import CarlaEnv
from models.cnn_extractor import DrivingCNN
from models.clamped_policy import ClampedStdPolicy
from eval.evaluator import DrivingMetricsCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="UrbanZero PPO Training")
    parser.add_argument("--n-envs", type=int, default=1,
                        help="Number of parallel CARLA environments (default: 1)")
    parser.add_argument("--base-port", type=int, default=2000,
                        help="Base CARLA port (each env uses base + i*1000)")
    parser.add_argument("--timesteps", type=int, default=10_000_000,
                        help="Total training timesteps (default: 10M)")
    parser.add_argument("--no-traffic", action="store_true",
                        help="Disable traffic spawning")
    parser.add_argument("--no-weather", action="store_true",
                        help="Disable weather randomization")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to model checkpoint to resume from")
    parser.add_argument("--experiment", type=str, default="shaped",
                        help="Experiment name for logs/checkpoints (default: shaped)")
    return parser.parse_args()


def make_env(rank, base_port, enable_traffic, enable_weather):
    """Factory function for creating CARLA environments with unique ports."""
    def _init():
        port = base_port + rank * 1000
        env = CarlaEnv(
            port=port,
            enable_traffic=enable_traffic,
            enable_weather_randomization=enable_weather,
            max_episode_steps=2000,       # 100 seconds at 20Hz
            num_traffic_vehicles=30,
            num_pedestrians=10,
        )
        return env
    return _init


def main():
    args = parse_args()

    LOG_DIR = os.path.expanduser(f"~/urbanzero/logs/{args.experiment}")
    CKPT_DIR = os.path.expanduser(f"~/urbanzero/checkpoints/{args.experiment}")
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    print(f"=== UrbanZero Training ===")
    print(f"  Envs: {args.n_envs}")
    print(f"  Timesteps: {args.timesteps:,}")
    print(f"  Traffic: {not args.no_traffic}")
    print(f"  Weather randomization: {not args.no_weather}")
    print(f"  Experiment: {args.experiment}")
    print(f"  Log dir: {LOG_DIR}")
    print(f"  Checkpoint dir: {CKPT_DIR}")

    # Create vectorized environment
    env_fns = [
        make_env(i, args.base_port, not args.no_traffic, not args.no_weather)
        for i in range(args.n_envs)
    ]

    if args.n_envs == 1:
        env = DummyVecEnv(env_fns)
    else:
        env = SubprocVecEnv(env_fns)

    # Frame stacking: 4 frames for temporal information
    # CRITICAL: must specify channels_order because our image is float32 [0,1],
    # not uint8 [0,255]. Without this, SB3 misclassifies it as non-image and
    # stacks along the wrong axis (width instead of channels).
    env = VecFrameStack(env, n_stack=4, channels_order={"image": "first", "state": "last"})

    # Reward normalization (don't normalize obs — we handle that in the env)
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=max(50000 // args.n_envs, 5000),
        save_path=CKPT_DIR,
        name_prefix="ppo_urbanzero",
        save_vecnormalize=True,
    )
    metrics_cb = DrivingMetricsCallback(
        eval_freq=max(20000 // args.n_envs, 2000),
        verbose=1,
    )
    callbacks = CallbackList([checkpoint_cb, metrics_cb])

    # PPO with custom CNN extractor
    policy_kwargs = dict(
        log_std_init=0.0,
        features_extractor_class=DrivingCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 128], vf=[256, 128]),  # separate actor/critic heads
    )

    # Scale batch parameters with number of envs
    n_steps = 512 if args.n_envs <= 2 else 256
    batch_size = 64 * args.n_envs  # scale with envs

    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = PPO.load(args.resume, env=env, device="cuda" if torch.cuda.is_available() else "cpu")
    else:
        model = PPO(
            ClampedStdPolicy,
            env,
            verbose=1,
            tensorboard_log=LOG_DIR,
            learning_rate=3e-4,
            ent_coef=0.01,              # exploration (prevents std collapse)
            vf_coef=0.5,               # value function loss weight
            max_grad_norm=0.5,          # gradient clipping
            n_steps=n_steps,            # rollout length per env
            batch_size=batch_size,      # mini-batch size
            n_epochs=5,                # PPO epochs per update
            gamma=0.99,                 # discount factor
            gae_lambda=0.95,            # GAE lambda
            clip_range=0.2,             # PPO clipping range
            policy_kwargs=policy_kwargs,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    print(f"Device: {model.device}")
    print(f"Policy architecture:\n{model.policy}")
    print(f"\nStarting training for {args.timesteps:,} timesteps...")

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model + VecNormalize stats
    final_path = os.path.join(CKPT_DIR, "final_model")
    model.save(final_path)
    env.save(os.path.join(CKPT_DIR, "vecnormalize.pkl"))
    print(f"Training complete. Model saved to {final_path}")

    env.close()


if __name__ == "__main__":
    main()
