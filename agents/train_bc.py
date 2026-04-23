"""
agents/train_bc.py — Gaussian NLL Behavior Cloning trainer for UrbanZero.

Trains the SB3 ClampedStdPolicy (actor + features extractor only) on expert
demonstrations collected by scripts/collect_bc_data.py.  Produces an SB3-
compatible .zip that agents/train.py can load via PPO.load() and a sibling
_vecnormalize.pkl so the resume path finds its expected file.

Usage:
    python3 agents/train_bc.py \\
        --data ~/urbanzero/bc_data/bc_data_<ts>.npz \\
        --output ~/urbanzero/checkpoints/bc_pretrain.zip

Architecture notes:
- We construct a real SB3 PPO so the policy architecture (DrivingCNN +
  ClampedStdPolicy) is byte-for-byte identical to what train.py creates.
- Only the features extractor + actor MLP + action_net + log_std are
  optimized. The value network (mlp_extractor.forward_critic, value_net)
  is left at its random initialization so PPO can fit a critic from scratch
  with real rollout returns, avoiding a BC-trained critic that has never
  seen real reward signals.
- log_std is a shared Parameter on ClampedStdPolicy. We include it in the
  optimizer and add a prior pulling it toward log(0.3) ~ -1.204 — small
  initial std is a reasonable starting point for imitation (expert
  trajectories have low action variance by design).
"""

import os
import sys
import argparse
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Ensure project root is importable before SB3 imports.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack

from models.cnn_extractor import DrivingCNN
from models.clamped_policy import ClampedStdPolicy


# -----------------------------------------------------------------------
# Constants — must match train.py and CarlaEnv exactly
# -----------------------------------------------------------------------

IMG_H = 128
IMG_W = 128
STATE_DIM = 10
ACTION_DIM = 2
N_STACK_DEFAULT = 4

LOG_STD_PRIOR = math.log(0.3)   # target for log_std regularization


# -----------------------------------------------------------------------
# Dummy gym env — provides correct observation/action spaces to SB3
# -----------------------------------------------------------------------

class _DummyDrivingEnv(gym.Env):
    """Minimal env that exposes the same spaces as CarlaEnv (unstacked).

    DummyVecEnv + VecFrameStack will expand the image channel axis and
    flatten the state to (state_dim * n_stack,), matching the runtime
    observation shape seen by the policy during PPO training.
    """

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(
                low=0.0, high=1.0, shape=(1, IMG_H, IMG_W), dtype=np.float32
            ),
            "state": gym.spaces.Box(
                low=-1.5, high=1.5, shape=(STATE_DIM,), dtype=np.float32
            ),
        })
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        obs = {
            "image": np.zeros((1, IMG_H, IMG_W), dtype=np.float32),
            "state": np.zeros(STATE_DIM, dtype=np.float32),
        }
        return obs, {}

    def step(self, action):
        obs = {
            "image": np.zeros((1, IMG_H, IMG_W), dtype=np.float32),
            "state": np.zeros(STATE_DIM, dtype=np.float32),
        }
        return obs, 0.0, False, False, {}


# -----------------------------------------------------------------------
# Offline frame stacking
# -----------------------------------------------------------------------

def _stack_frames(
    images: np.ndarray,
    states: np.ndarray,
    n_stack: int,
    episode_starts: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply offline VecFrameStack semantics to raw single-frame buffers.

    VecFrameStack with channels_order={"image": "first", "state": "last"}:
      - image : stack last n_stack frames along channel axis (axis 1)
                result shape: (N, n_stack, H, W)
      - state : stack last n_stack frames along last axis, then flatten
                result shape: (N, STATE_DIM * n_stack)  — matches DrivingCNN

    Episode boundary handling: if `episode_starts` is provided (bool array
    of length N where True marks the first frame of an episode), the
    stacking walks back ONLY within the current episode. For frames within
    n_stack-1 of an episode start, we frame-repeat the first frame of the
    episode (not of the previous episode). Without this guard the first
    ~3 frames of every episode would contain stale prior-episode content,
    contaminating ~3% of the dataset at ~100-step episode length.

    Args:
        images : (N, 1, H, W) float32
        states : (N, STATE_DIM) float32
        n_stack: number of frames to stack
        episode_starts : (N,) bool, True at first frame of each episode.
                         If None, fall back to treating the whole buffer
                         as one episode (legacy behavior).

    Returns:
        images_stacked : (N, n_stack, H, W) float32
        states_stacked : (N, STATE_DIM * n_stack) float32
    """
    N = len(images)
    assert len(states) == N, "images and states must have same length"
    if episode_starts is not None:
        assert len(episode_starts) == N, "episode_starts must match length"

    H, W = images.shape[2], images.shape[3]
    images_stacked = np.zeros((N, n_stack, H, W), dtype=np.float32)
    states_stacked = np.zeros((N, STATE_DIM * n_stack), dtype=np.float32)

    # Precompute episode-start index for each frame: the largest j <= i
    # where episode_starts[j] is True. This is the earliest frame we may
    # walk back to when building the stack for frame i.
    if episode_starts is not None:
        ep_start_idx = np.zeros(N, dtype=np.int64)
        last = 0
        for i in range(N):
            if episode_starts[i]:
                last = i
            ep_start_idx[i] = last
    else:
        ep_start_idx = None

    for i in range(N):
        lo = ep_start_idx[i] if ep_start_idx is not None else 0
        for k in range(n_stack):
            # k=0 is the oldest frame, k=n_stack-1 is the current frame
            src_idx = i - (n_stack - 1 - k)
            # Clamp to episode start, then to 0 — never cross an episode
            # boundary; instead frame-repeat the first frame of the
            # current episode.
            src_idx = max(src_idx, lo, 0)
            images_stacked[i, k] = images[src_idx, 0]  # strip the channel-1 dim
            states_stacked[i, k * STATE_DIM : (k + 1) * STATE_DIM] = states[src_idx]

    return images_stacked, states_stacked


# -----------------------------------------------------------------------
# Policy parameter groups: actor-only (freeze value head)
# -----------------------------------------------------------------------

def _actor_params(policy: nn.Module) -> list:
    """Return parameters for the features extractor, actor MLP, action_net,
    and log_std.  The value MLP (mlp_extractor's critic path) and value_net
    linear layer are intentionally excluded so they are trained from scratch
    during PPO fine-tuning on real rewards.

    SB3 ActorCriticPolicy (2.x) attribute layout:
        policy.features_extractor       — shared DrivingCNN (or pi_features_extractor
                                          if separate extractors are used)
        policy.mlp_extractor            — MlpExtractor with forward_actor / forward_critic
            .policy_net                 — actor MLP Sequential
            .value_net                  — critic MLP Sequential  <- excluded
        policy.action_net               — final linear steer/throttle
        policy.value_net                — final linear V(s)       <- excluded
        policy.log_std                  — shared log-std Parameter
    """
    # Collect value-only parameter ids to exclude
    value_param_ids: set = set()

    # mlp_extractor.value_net (critic MLP layers)
    if hasattr(policy, "mlp_extractor") and hasattr(policy.mlp_extractor, "value_net"):
        for p in policy.mlp_extractor.value_net.parameters():
            value_param_ids.add(id(p))

    # policy.value_net (final V(s) linear layer in SB3 2.x)
    if hasattr(policy, "value_net"):
        for p in policy.value_net.parameters():
            value_param_ids.add(id(p))

    actor_params = [
        p for p in policy.parameters()
        if p.requires_grad and id(p) not in value_param_ids
    ]
    return actor_params


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Behavior Cloning trainer for UrbanZero (Gaussian NLL)."
    )
    parser.add_argument("--data", type=str, required=True,
                        help="Path to .npz file from collect_bc_data.py")
    default_out = os.path.expanduser("~/urbanzero/checkpoints/bc_pretrain.zip")
    parser.add_argument("--output", type=str, default=default_out,
                        help="Output SB3 .zip path "
                             "(default: ~/urbanzero/checkpoints/bc_pretrain.zip)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Training epochs (default: 30)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Mini-batch size (default: 256)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Adam learning rate (default: 3e-4)")
    _default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=_default_device,
                        help=f"Torch device (default: {_default_device})")
    parser.add_argument("--n_stack", type=int, default=N_STACK_DEFAULT,
                        help=f"Frame stack depth — must match VecFrameStack in train.py "
                             f"(default: {N_STACK_DEFAULT})")
    return parser.parse_args()


# -----------------------------------------------------------------------
# Main training routine
# -----------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    data_path   = os.path.expanduser(args.data)
    output_path = os.path.expanduser(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    device = torch.device(args.device)
    print(f"[BC-train] device={device}  data={data_path}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("[BC-train] Loading .npz ...")
    npz = np.load(data_path, allow_pickle=True)
    raw_images  = npz["images"]    # (N, 1, 128, 128) float32
    raw_states  = npz["states"]    # (N, 10) float32
    raw_actions = npz["actions"]   # (N, 2) float32
    # episode_starts may be missing in legacy .npz files written before
    # the 2026-04-23 collector update; fall back to None (legacy stacking).
    if "episode_starts" in npz.files:
        raw_episode_starts = npz["episode_starts"].astype(bool)
        n_eps = int(raw_episode_starts.sum())
        print(f"[BC-train] episode_starts present: {n_eps} episodes")
    else:
        raw_episode_starts = None
        print("[BC-train] WARNING: .npz has no episode_starts; "
              "stacking will contaminate first 3 frames of each episode "
              "(legacy behavior)")
    N = len(raw_images)
    print(f"[BC-train] {N:,} frames loaded. Applying {args.n_stack}-frame stack ...")

    # ------------------------------------------------------------------
    # 2. Offline frame stacking
    # ------------------------------------------------------------------
    images_stacked, states_stacked = _stack_frames(
        raw_images, raw_states, args.n_stack, raw_episode_starts
    )
    # images_stacked : (N, n_stack, H, W)
    # states_stacked : (N, STATE_DIM * n_stack)
    print(f"[BC-train] Stacked shapes: images={images_stacked.shape}, "
          f"states={states_stacked.shape}, actions={raw_actions.shape}")

    # Convert to tensors
    t_images  = torch.from_numpy(images_stacked).float()    # (N, n_stack, H, W)
    t_states  = torch.from_numpy(states_stacked).float()    # (N, STATE_DIM*n_stack)
    t_actions = torch.from_numpy(raw_actions).float()       # (N, 2)

    dataset = TensorDataset(t_images, t_states, t_actions)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,       # avoid multiprocessing complexity
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    n_batches = len(loader)
    print(f"[BC-train] DataLoader: {n_batches} batches/epoch, "
          f"batch_size={args.batch_size}")

    # ------------------------------------------------------------------
    # 3. Build SB3 PPO with the exact architecture train.py uses
    # ------------------------------------------------------------------
    print("[BC-train] Building SB3 PPO model (exact train.py architecture) ...")

    dummy_env = DummyVecEnv([lambda: _DummyDrivingEnv()])
    dummy_env = VecFrameStack(
        dummy_env,
        n_stack=args.n_stack,
        channels_order={"image": "first", "state": "last"},
    )
    # VecNormalize with norm_obs=False so obs spaces pass through unchanged
    dummy_env = VecNormalize(dummy_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    policy_kwargs = dict(
        log_std_init=-0.5,
        features_extractor_class=DrivingCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
    )

    model = PPO(
        policy=ClampedStdPolicy,
        env=dummy_env,
        policy_kwargs=policy_kwargs,
        device=args.device,
        learning_rate=args.lr,
        verbose=0,
    )
    policy = model.policy.to(device)
    policy.train()

    print(f"[BC-train] Policy architecture:\n{policy}")

    # ------------------------------------------------------------------
    # 4. Optimizer: actor params only (value head frozen)
    # ------------------------------------------------------------------
    actor_params = _actor_params(policy)
    n_actor_params = sum(p.numel() for p in actor_params)
    n_total_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[BC-train] Optimizing {n_actor_params:,}/{n_total_params:,} params "
          f"(features extractor + actor MLP + action_net + log_std; value head frozen)")

    optimizer = torch.optim.Adam(actor_params, lr=args.lr)

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    print(f"[BC-train] Starting {args.epochs} epochs ...")

    for epoch in range(1, args.epochs + 1):
        epoch_nll      = 0.0
        epoch_mae      = 0.0
        n_batches_done = 0

        for batch_images, batch_states, batch_actions in loader:
            batch_images   = batch_images.to(device)    # (B, n_stack, H, W)
            batch_states   = batch_states.to(device)    # (B, STATE_DIM*n_stack)
            batch_actions  = batch_actions.to(device)   # (B, 2)

            obs_dict = {"image": batch_images, "state": batch_states}

            optimizer.zero_grad()

            # Forward through actor pathway (SB3 ActorCriticPolicy internals).
            # In SB3 2.3+ with separate features extractors, use
            # pi_features_extractor; fall back to features_extractor for
            # older versions that share a single extractor.
            if hasattr(policy, "pi_features_extractor"):
                features = policy.extract_features(
                    obs_dict, policy.pi_features_extractor
                )
            else:
                features = policy.extract_features(obs_dict)

            latent_pi    = policy.mlp_extractor.forward_actor(features)
            mean_actions = policy.action_net(latent_pi)  # (B, 2)

            # log_std is a shared Parameter: shape (action_dim,) in SB3 DiagGaussianDist
            log_std = policy.log_std   # (2,) Parameter
            std     = log_std.exp()    # (2,)

            # Gaussian NLL: 0.5 * sum_dim[(e/std)^2] + sum_dim[log_std]
            # + tiny log_std prior pulling toward log(0.3) for good init std
            residual = batch_actions - mean_actions     # (B, 2)
            nll = (
                0.5 * (residual / std).pow(2).sum(dim=-1).mean()
                + log_std.sum()
            )
            log_std_prior = 0.01 * (log_std - LOG_STD_PRIOR).pow(2).sum()
            loss = nll + log_std_prior

            loss.backward()
            # Clip gradients to match train.py's max_grad_norm
            nn.utils.clip_grad_norm_(actor_params, max_norm=0.5)
            optimizer.step()

            # Apply ClampedStdPolicy's upper bound after each optimizer step
            if hasattr(policy, "_clamp"):
                policy._clamp()

            with torch.no_grad():
                mae = residual.abs().mean().item()

            epoch_nll      += nll.item()
            epoch_mae      += mae
            n_batches_done += 1

        avg_nll = epoch_nll / max(n_batches_done, 1)
        avg_mae = epoch_mae / max(n_batches_done, 1)
        print(
            f"[BC-train] epoch {epoch:03d}/{args.epochs}  "
            f"NLL={avg_nll:.4f}  MAE={avg_mae:.4f}  "
            f"log_std=[{policy.log_std.data[0]:.3f}, {policy.log_std.data[1]:.3f}]  "
            f"std=[{policy.log_std.data.exp()[0]:.3f}, {policy.log_std.data.exp()[1]:.3f}]"
        )

    # ------------------------------------------------------------------
    # 6. Save SB3 .zip
    # ------------------------------------------------------------------
    policy.set_training_mode(False)
    model.save(output_path)
    print(f"[BC-train] Model saved -> {output_path}")

    # ------------------------------------------------------------------
    # 7. Save sibling VecNormalize .pkl
    # The pkl contains identity running stats (no real rollouts done here).
    # train.py's BC warmstart path looks for:
    #   <URBANZERO_BC_WEIGHTS without .zip>_vecnormalize.pkl
    # We save it there and also save a copy as vecnormalize.pkl in the
    # same directory so that the standard resume path also finds it.
    # NOTE: VecNormalize.save() internally uses Python's pickle module
    # to serialize running mean/variance stats — this is the standard SB3
    # mechanism and is safe as long as the .pkl originates from this script.
    # ------------------------------------------------------------------
    base_path = output_path
    if base_path.endswith(".zip"):
        base_path = base_path[:-4]
    vecnorm_sibling = base_path + "_vecnormalize.pkl"
    dummy_env.save(vecnorm_sibling)
    print(f"[BC-train] VecNormalize stats saved -> {vecnorm_sibling}")

    # Also save a copy next to the .zip as vecnormalize.pkl for standard
    # resume compatibility.
    vecnorm_dir_copy = os.path.join(os.path.dirname(output_path), "vecnormalize.pkl")
    if not os.path.exists(vecnorm_dir_copy):
        dummy_env.save(vecnorm_dir_copy)
        print(f"[BC-train] Also saved identity VecNormalize -> {vecnorm_dir_copy}")

    print(f"[BC-train] Done. To use as BC warmstart:\n"
          f"  export URBANZERO_BC_WEIGHTS={output_path}\n"
          f"  python3 agents/train.py")

    dummy_env.close()


if __name__ == "__main__":
    main()
