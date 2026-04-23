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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack
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

    # BC-finetune hyperparameter override. When starting from a BC prior
    # (URBANZERO_BC_WEIGHTS set), PPO needs radically smaller updates than
    # from-scratch training or it will destabilize the prior via excessive
    # approx_kl and clip_fraction. Observed: at lr=3e-4 / ent_coef=0.02
    # on a BC model with std=0.22, approx_kl hit 0.079 (5x healthy), clip_
    # fraction 0.34 (2x healthy). Roach (Zhang 2021 §3.2) uses lr=1e-5 for
    # BC finetune; we use lr=1e-4 as a compromise for our 5M-step budget.
    # ent_coef also cut: 0.005 start -> 0.001 floor stays within Andrychowicz
    # 2021's viable band's lower edge (0.003-0.03), preserving BC's
    # learned log_std rather than forcing wider exploration.
    _is_bc_finetune = bool(os.environ.get("URBANZERO_BC_WEIGHTS", "").strip())
    if _is_bc_finetune:
        # v4 diagnosis (after v1/v2/v3 all failed): the core issue wasn't
        # σ-amplification (v3 fixed that — KL looked perfect). The issue is
        # that our REWARD was designed for pure-RL to break the sit-still
        # attractor. It has idle_cost=-0.15/step that actively PUNISHES
        # BC's expert behavior (stopping at red lights, slowing behind
        # traffic). PPO pushes the policy AWAY from what BC taught; the
        # entropy bonus compounds by widening σ until the policy is random.
        #
        # v4 fix is to make the REWARD BC-compatible (not PPO-hyperparams):
        #   env side (via URBANZERO_IDLE_COST_COEF=0 + REALLY_STUCK_STEPS=
        #   3000): stop punishing correct stopping; BC prior prevents the
        #   sit-still attractor on its own.
        #   policy side: keep BC's tight σ=0.22 (revert v3's widening);
        #   drop ent_coef to near-zero so entropy bonus doesn't drift σ
        #   upward. With BC-compatible reward, gradients are small and
        #   directional, so tight σ is fine.
        #   PPO hparams: keep v3's lr=1e-4, n_epochs=1, clip_range=0.1
        #   as additional safety; these aren't the root cause but they
        #   don't hurt.
        # v5 hyperparameters from external reviewer's corrections:
        #   lr=5e-5 (v4 was 1e-4; reviewer: "nudge, not overwrite the BC policy")
        #   ent_coef=1e-3 constant (v4 was 1e-4; reviewer recommends 0.001 floor)
        #   batch_size handled at PPO construction (increased from 64 to 128)
        LR_BC_FINETUNE = 5e-5
        ENT_COEF_START = 1e-3
        ENT_COEF_FLOOR = 1e-3
        N_EPOCHS_BC = 1
        CLIP_RANGE_BC = 0.1
        WIDEN_LOG_STD_TO = None   # BC's σ=0.22 is fine once reward doesn't fight it
        print(f"  [BC-finetune v5] lr={LR_BC_FINETUNE}, n_epochs={N_EPOCHS_BC}, "
              f"clip_range={CLIP_RANGE_BC}, "
              f"ent_coef={ENT_COEF_START} (constant, no schedule), "
              f"widen_log_std={'disabled' if WIDEN_LOG_STD_TO is None else WIDEN_LOG_STD_TO}")
        print(f"  [BC-finetune v5] IMPORTANT: requires env vars at launch:")
        print(f"                   URBANZERO_IDLE_COST_COEF=0")
        print(f"                   URBANZERO_REALLY_STUCK_STEPS=3000")
        print(f"                   URBANZERO_COLLISION_COEF=0.01  (smooth collision)")
    else:
        LR_BC_FINETUNE = 3e-4  # from-scratch default
        N_EPOCHS_BC = 3
        CLIP_RANGE_BC = 0.2
        WIDEN_LOG_STD_TO = None

    print(f"=== UrbanZero Training ===")
    print(f"  Envs: {args.n_envs}")
    print(f"  Timesteps: {args.timesteps:,}")
    print(f"  Traffic: {not args.no_traffic}")
    print(f"  Weather randomization: {not args.no_weather}")
    print(f"  Experiment: {args.experiment}")
    print(f"  Log dir: {LOG_DIR}")
    print(f"  Checkpoint dir: {CKPT_DIR}")
    print(f"  Ent-coef schedule: {ENT_COEF_START} -> {ENT_COEF_FLOOR} (floor) over 10M steps")

    # Create vectorized environment.
    #
    # DummyVecEnv, not SubprocVecEnv — see GitHub issues #3/#4/#6 in this
    # repo. SB3's SubprocVecEnv has no worker-death recovery: when a
    # worker process dies (e.g. BrokenPipeError in the step pipe),
    # SubprocVecEnv.step_wait() calls remote.recv() with no timeout,
    # and the trainer hangs forever on `unix_stream_read_generic`. This
    # is a documented SB3 limitation, not a CARLA issue — even a
    # successful 3-iteration run at 147 FPS was killed by a subsequent
    # worker crash we couldn't catch.
    #
    # DummyVecEnv runs all envs serially in the main process: no pipes,
    # no IPC, no BrokenPipeError possible. The envs still connect to
    # their own CARLA servers on their own ports; the only difference
    # is that env.step() calls are sequential instead of parallel.
    # At 2 envs the wall-clock cost is ~30-40% throughput (147 FPS -> ~90-110
    # FPS expected) — well above the 70 FPS PASS gate. Reliability
    # dominates throughput at this deadline.
    #
    # If we ever need the throughput, wrap SubprocVecEnv with a custom
    # class that catches BrokenPipeError/EOFError on step_wait() and
    # restarts the dead worker. Out of scope for the Saturday deadline.
    env_fns = [
        make_env(i, args.base_port, not args.no_traffic, not args.no_weather, seed)
        for i in range(args.n_envs)
    ]
    env = DummyVecEnv(env_fns)
    print(f"  VecEnv: DummyVecEnv (n_envs={args.n_envs}, serial stepping)")

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
    # v5 per external reviewer: "Small batches in CARLA lead to high variance
    # gradients." Increase BC-finetune batch size to 128 (pure-RL unchanged).
    if _is_bc_finetune:
        batch_size = 128
    else:
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
            learning_rate=LR_BC_FINETUNE,
            n_epochs=N_EPOCHS_BC,
            ent_coef=ENT_COEF_START,
            clip_range=CLIP_RANGE_BC,
            max_grad_norm=0.5,
        )
    else:
        # BC warmstart path — only activates when URBANZERO_BC_WEIGHTS is set
        # AND no --resume checkpoint was supplied (i.e., this is a fresh start
        # that wants a BC prior).  If --resume is set, that always wins.
        _bc_weights = os.environ.get("URBANZERO_BC_WEIGHTS", "").strip()
        if _bc_weights:
            _bc_weights = os.path.expanduser(_bc_weights)
            print(f"[BC-warmstart] loading weights from {_bc_weights}")
            model = PPO.load(
                _bc_weights,
                env=env,
                device="cuda" if torch.cuda.is_available() else "cpu",
                # Same hyperparameter override as the resume path — ensures
                # PPO does not silently inherit stale values from the BC zip.
                learning_rate=LR_BC_FINETUNE,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=N_EPOCHS_BC,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=ENT_COEF_START,
                vf_coef=0.5,
                clip_range=CLIP_RANGE_BC,
                max_grad_norm=0.5,
            )
            # H2 fix (Issue #7): BC converged to std=[0.22, 0.21] which
            # amplifies PPO's log-prob-ratio gradients by ~20x vs std=1.0.
            # Widen log_std without touching the mean network — keeps BC's
            # good actor (MAE=0.05) while letting PPO updates not blow up.
            # Target: std=0.5 → log_std=log(0.5)≈-0.69 (Andrychowicz 2021
            # §4.5 viable band [0.3, 0.7] — within the existing upper clamp
            # at log(0.7)≈-0.357 in models/clamped_policy.py).
            if WIDEN_LOG_STD_TO is not None:
                with torch.no_grad():
                    old_ls = model.policy.log_std.data.clone()
                    model.policy.log_std.data.fill_(WIDEN_LOG_STD_TO)
                    new_std = torch.exp(model.policy.log_std.data)
                print(f"[BC-warmstart] widened log_std {old_ls.tolist()} -> "
                      f"{model.policy.log_std.data.tolist()} "
                      f"(std now {new_std.tolist()})")
            # Attempt to load sibling VecNormalize stats.  The BC trainer
            # saves identity stats (no real rollouts), so this is mainly
            # for structural completeness; reward normalization will adapt
            # quickly from real rollouts.
            _bc_vecnorm_path = _bc_weights.replace(".zip", "") + "_vecnormalize.pkl"
            if os.path.exists(_bc_vecnorm_path):
                # env is currently a VecNormalize wrapping VecFrameStack.
                # We load the saved stats into it, preserving the env object.
                # Guard each field with hasattr because training env uses
                # norm_obs=False (reward-only VecNormalize) so obs_rms does
                # not exist; copying blindly raises AttributeError at launch.
                _vn_loaded = VecNormalize.load(_bc_vecnorm_path, env.venv)
                if hasattr(env, "obs_rms") and hasattr(_vn_loaded, "obs_rms"):
                    env.obs_rms = _vn_loaded.obs_rms
                if hasattr(env, "ret_rms") and hasattr(_vn_loaded, "ret_rms"):
                    env.ret_rms = _vn_loaded.ret_rms
                env.training = True
                print(f"[BC-warmstart] VecNormalize stats restored from {_bc_vecnorm_path}")
            else:
                print(f"[BC-warmstart] {_bc_vecnorm_path} not found — using initial stats")
        else:
            model = PPO(
                ClampedStdPolicy,
                env,
                verbose=1,
                tensorboard_log=LOG_DIR,
                learning_rate=LR_BC_FINETUNE,  # Schulman 2017 PPO default (3e-4)
                                            # from scratch; lowered to 1e-4
                                            # when BC warmstart is active (see
                                            # LR_BC_FINETUNE definition above).
                ent_coef=ENT_COEF_START,    # initialized high, annealed by callback
                vf_coef=0.5,                # value function loss weight
                max_grad_norm=0.5,          # gradient clipping
                n_steps=n_steps,            # rollout length per env
                batch_size=batch_size,      # mini-batch size
                n_epochs=3,                 # Compromise between prior 2 (proven
                                            # at 7M steps without deadlock) and
                                            # the PPO-default 10. Going to 4 in
                                            # v2 doubled the tick-gap between
                                            # rollouts and triggered CARLA issue
                                            # #9172 (TrafficManagerLocal race) —
                                            # see GitHub issue #4 root-cause.
                                            # 3 keeps the gap close to the 7M
                                            # run's proven-safe cadence while
                                            # giving slightly more gradient use.
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
