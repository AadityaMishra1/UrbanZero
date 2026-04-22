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
from env.safety_wrapper import NaNGuardWrapper
from models.cnn_extractor import DrivingCNN
from models.clamped_policy import ClampedStdPolicy
from eval.evaluator import DrivingMetricsCallback
from eval.beacon_callback import BeaconCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
import torch
import time as _time
import random
import numpy as np


class WallClockCheckpointCallback(BaseCallback):
    """Save a checkpoint every N wall-clock minutes, independent of step count.

    This catches the failure mode where SB3's step-based CheckpointCallback
    stops firing (e.g. step counter drift, hung episodes) — the model still
    gets saved on a real-time schedule so you never lose more than N minutes
    of training to a crash.
    """

    def __init__(self, save_path, save_minutes=10, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_interval = save_minutes * 60
        self._last_save_time = _time.time()

    def _on_step(self) -> bool:
        now = _time.time()
        elapsed = now - self._last_save_time
        if elapsed >= self.save_interval:
            path = os.path.join(self.save_path, f"autosave_{self.num_timesteps}_steps")
            self.model.save(path)
            if hasattr(self.training_env, "save"):
                self.training_env.save(
                    os.path.join(self.save_path, "vecnormalize.pkl")
                )
            if self.verbose > 0:
                print(f"\n[autosave @ {self.num_timesteps} steps, "
                      f"{elapsed / 60:.1f}min since last save]")
            self._last_save_time = now
        return True


class EntCoefAnnealCallback(BaseCallback):
    """Linearly anneal PPO.ent_coef from start_value to floor_value over
    anneal_steps env-steps, then hold at floor.

    Cite: Andrychowicz et al. 2021 "What Matters in On-Policy RL" §4.5 —
    entropy coefficient is one of the top-5 impactful choices for continuous
    control. Schedule matters: constant-too-low collapses exploration early
    (the 7M run); constant-too-high caps asymptotic performance; linear decay
    with a nonzero floor balances both. Rajeswaran et al. 2017 §3 warns
    specifically against annealing ent_coef to zero when fine-tuning from
    imitation — the policy decays back toward random.
    """

    def __init__(self, start_value, floor_value, anneal_steps, verbose=0):
        super().__init__(verbose)
        self.start_value = float(start_value)
        self.floor_value = float(floor_value)
        self.anneal_steps = int(anneal_steps)
        self._last_logged_at = -1

    def _on_step(self) -> bool:
        t = self.num_timesteps
        frac = min(1.0, max(0.0, t / self.anneal_steps))
        new_coef = self.start_value + frac * (self.floor_value - self.start_value)
        self.model.ent_coef = new_coef
        # Record to SB3 logger so BeaconCallback can surface the current
        # coefficient. SB3 does not auto-log ent_coef for PPO; if we don't
        # record it here the beacon's "ent_coef" field is always None.
        # record() is cheap (no-op until next dump); safe to call every step.
        self.model.logger.record("train/ent_coef", float(new_coef))
        if self.verbose and (t - self._last_logged_at) >= 100_000:
            print(f"[ent_coef_sched] t={t} ent_coef={new_coef:.4f}")
            self._last_logged_at = t
        return True


class RollingBestCallback(BaseCallback):
    """Save `best_by_rc.zip` + `best_vecnormalize.pkl` whenever rolling
    route_completion exceeds prior max.

    Addresses Agent-4 infra finding: the 7M run peaked at 7.99% RC at Phase
    2 step 1.14M and regressed to 5.95% by the end — we would have shipped
    the regressed model. This callback retains the peak independent of the
    final checkpoint.

    Uses a rolling-window deque matching BeaconCallback's sizing so we
    don't ship based on a single lucky episode.
    """

    def __init__(self, save_path, window=50, min_episodes=20, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.window = window
        self.min_episodes = min_episodes
        self.best_rc = 0.0
        from collections import deque
        self._ep_rcs = deque(maxlen=window)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos") or []
        if dones is None:
            return True
        for i, done in enumerate(dones):
            if done and i < len(infos):
                self._ep_rcs.append(float(infos[i].get("route_completion", 0.0)))
        if len(self._ep_rcs) < self.min_episodes:
            return True
        rolling_rc = sum(self._ep_rcs) / len(self._ep_rcs)
        if rolling_rc > self.best_rc:
            self.best_rc = rolling_rc
            os.makedirs(self.save_path, exist_ok=True)
            path = os.path.join(self.save_path, "best_by_rc")
            self.model.save(path)
            if hasattr(self.training_env, "save"):
                self.training_env.save(
                    os.path.join(self.save_path, "best_vecnormalize.pkl")
                )
            if self.verbose:
                print(f"\n[rolling-best] new best rolling RC = {rolling_rc:.4f} "
                      f"@ t={self.num_timesteps}, saved to {path}.zip")
        return True


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


def make_env(rank, base_port, enable_traffic, enable_weather, base_seed):
    """Factory function for creating CARLA environments with unique ports.

    Each worker gets a distinct RNG seed (base_seed + rank) so that under
    SubprocVecEnv workers pick different spawn points / weather / traffic
    patterns rather than cloning the main process's seeded RNG state.
    Without this, domain randomization collapses to n_envs copies of the
    same episode sequence.
    """
    def _init():
        import random as _random
        worker_seed = base_seed + rank
        _random.seed(worker_seed)
        np.random.seed(worker_seed)
        port = base_port + rank * 1000
        env = CarlaEnv(
            port=port,
            enable_traffic=enable_traffic,
            enable_weather_randomization=enable_weather,
            max_episode_steps=3000,       # 150 seconds at 20Hz: enough room
                                          # for slow early agents to complete
                                          # 200-800m routes without timing out
            # Traffic density reduced from 30 -> 15. 30 NPCs on a 200-800m
            # route with only ~100 spawn points in a typical CARLA town is
            # dense urban traffic — too intimidating for a from-scratch
            # policy that hasn't yet learned basic locomotion. 15 gives the
            # agent room to make mistakes without constant collision risk.
            num_traffic_vehicles=15,
            num_pedestrians=10,
        )
        # Boundary NaN/Inf guard — turns silent corruption into a logged
        # forced-terminal so VecNormalize never absorbs an Inf and PPO
        # never sees a NaN obs.
        env = NaNGuardWrapper(env)
        return env
    return _init


def main():
    args = parse_args()

    # Seed everything reproducibly so a "weird run" is at least the same
    # weird run twice. Per-env CARLA RNG (TM, weather, spawn) still varies
    # because it derives from random.randint() calls inside CarlaEnv.reset().
    seed = int(os.environ.get("URBANZERO_SEED", "42"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    LOG_DIR = os.path.expanduser(f"~/urbanzero/logs/{args.experiment}")
    CKPT_DIR = os.path.expanduser(f"~/urbanzero/checkpoints/{args.experiment}")
    BEACON_PATH = os.path.expanduser("~/urbanzero/beacon.json")
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    # Entropy coefficient schedule constants — hoisted above callback
    # construction because the EntCoefAnnealCallback constructor reads them.
    # Andrychowicz 2021 benchmarks 0.003-0.03 as the viable band for
    # continuous control from scratch; 0.001 (prior run) is below that
    # and caused the documented entropy collapse (std: 0.367 -> 0.230,
    # entropy_loss: -0.831 -> +0.115). The 0.01 floor (not 0) is critical
    # per the same paper — dropping ent_coef to 0 late in training is the
    # Rajeswaran et al. 2017 collapse-trigger we're avoiding.
    ENT_COEF_START = 0.02
    ENT_COEF_FLOOR = 0.01

    print(f"=== UrbanZero Training ===")
    print(f"  Envs: {args.n_envs}")
    print(f"  Timesteps: {args.timesteps:,}")
    print(f"  Traffic: {not args.no_traffic}")
    print(f"  Weather randomization: {not args.no_weather}")
    print(f"  Experiment: {args.experiment}")
    print(f"  Log dir: {LOG_DIR}")
    print(f"  Checkpoint dir: {CKPT_DIR}")
    print(f"  Ent-coef schedule: {ENT_COEF_START} -> {ENT_COEF_FLOOR} (floor) over 10M steps")

    # Create vectorized environment
    env_fns = [
        make_env(i, args.base_port, not args.no_traffic, not args.no_weather, seed)
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

    # Reward normalization (don't normalize obs — we handle that in the env).
    # When resuming, load saved VecNormalize stats so the agent sees the same
    # reward distribution it was trained on — otherwise GAE advantages become
    # invalid and training destabilizes.
    if args.resume:
        vecnorm_path = os.path.join(os.path.dirname(args.resume), "vecnormalize.pkl")
        if os.path.exists(vecnorm_path):
            env = VecNormalize.load(vecnorm_path, env)
            env.training = True
            print(f"  Loaded VecNormalize stats from {vecnorm_path}")
        else:
            print(f"  Warning: {vecnorm_path} not found, using fresh VecNormalize")
            env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0, clip_obs=10.0)
    else:
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
    # Wall-clock checkpoint: saves every 10 minutes regardless of step count,
    # so a crash never loses more than 10 minutes of training even if the
    # step-based CheckpointCallback stops firing.
    wallclock_cb = WallClockCheckpointCallback(
        save_path=CKPT_DIR, save_minutes=10, verbose=1,
    )
    # Progress beacon: writes ~/urbanzero/beacon.json every 30s. The watchdog
    # uses its mtime to detect a hung trainer; the user uses it to glance
    # at training health without attaching to tmux.
    beacon_cb = BeaconCallback(
        beacon_path=BEACON_PATH,
        experiment=args.experiment,
        carla_port=args.base_port,
        write_seconds=30,
        verbose=1,
    )
    # Entropy coefficient anneal: 0.02 -> 0.01 (floor) over 10M steps.
    ent_coef_cb = EntCoefAnnealCallback(
        start_value=ENT_COEF_START,
        floor_value=ENT_COEF_FLOOR,
        anneal_steps=10_000_000,
        verbose=1,
    )
    # Rolling-best: save the model whenever rolling route_completion hits
    # a new high. Final eval / demo runs from best_by_rc.zip, not from the
    # last autosave.
    rolling_best_cb = RollingBestCallback(
        save_path=CKPT_DIR, window=50, min_episodes=20, verbose=1,
    )
    callbacks = CallbackList([
        checkpoint_cb, metrics_cb, wallclock_cb, beacon_cb,
        ent_coef_cb, rolling_best_cb,
    ])

    # PPO with custom CNN extractor.
    # log_std_init = -0.5 gives std ~= 0.6 at init — Andrychowicz et al. 2021
    # "What Matters in On-Policy RL" (ICLR) found the 0.5-0.7 std range
    # is where PPO's exploration-vs-exploitation works for continuous control
    # from scratch. Prior -1.0 combined with the [-2, -0.7] clamp capped
    # std at 0.5 and drove it to the lower floor (0.14) over 7M steps.
    # New ClampedStdPolicy only caps from above (std <= 1.0).
    policy_kwargs = dict(
        log_std_init=-0.5,
        features_extractor_class=DrivingCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
    )

    # Scale batch parameters with number of envs
    n_steps = 512 if args.n_envs <= 2 else 256
    batch_size = 64 * args.n_envs  # scale with envs

    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = PPO.load(
            args.resume,
            env=env,
            device="cuda" if torch.cuda.is_available() else "cpu",
            # Override checkpoint's saved hyperparameters with current values.
            # Without this, PPO.load() silently uses the old LR/epochs from the
            # checkpoint, ignoring the tuned values above.
            learning_rate=3e-4,
            n_epochs=4,
            ent_coef=ENT_COEF_START,
            clip_range=0.2,
            max_grad_norm=0.5,
        )
    else:
        model = PPO(
            ClampedStdPolicy,
            env,
            verbose=1,
            tensorboard_log=LOG_DIR,
            learning_rate=3e-4,         # Schulman et al. 2017 PPO default;
                                        # prior 5e-5 was a post-hoc stabilizer
                                        # for a broken reward, no longer needed
            ent_coef=ENT_COEF_START,    # initialized high, annealed by callback
            vf_coef=0.5,                # value function loss weight
            max_grad_norm=0.5,          # gradient clipping
            n_steps=n_steps,            # rollout length per env
            batch_size=batch_size,      # mini-batch size
            n_epochs=4,                 # PPO default; fewer epochs meant
                                        # less gradient per rollout
            gamma=0.99,                 # discount factor
            gae_lambda=0.95,            # GAE lambda
            clip_range=0.2,             # standard PPO clip range
            policy_kwargs=policy_kwargs,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    print(f"Device: {model.device}")
    print(f"Policy architecture:\n{model.policy}")
    print(f"\nStarting training for {args.timesteps:,} timesteps...")

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except Exception as e:
        # Emergency save — don't lose weights that are sitting in GPU memory.
        print(f"\n[CRASH] Training failed at {model.num_timesteps} steps: {e}")
        try:
            emergency_path = os.path.join(CKPT_DIR, f"emergency_{model.num_timesteps}_steps")
            model.save(emergency_path)
            env.save(os.path.join(CKPT_DIR, "vecnormalize.pkl"))
            print(f"[CRASH] Emergency checkpoint saved to {emergency_path}")
        except Exception as save_err:
            print(f"[CRASH] Emergency save also failed: {save_err}")
        raise

    # Save final model + VecNormalize stats
    final_path = os.path.join(CKPT_DIR, "final_model")
    model.save(final_path)
    env.save(os.path.join(CKPT_DIR, "vecnormalize.pkl"))
    print(f"Training complete. Model saved to {final_path}")

    env.close()


if __name__ == "__main__":
    main()
