import os
import sys
sys.path.insert(0, os.path.expanduser("~/urbanzero"))

from env.carla_env import CarlaEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
import torch

LOG_DIR = os.path.expanduser("~/urbanzero/logs/naive")
CKPT_DIR = os.path.expanduser("~/urbanzero/checkpoints/naive")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

def make_env():
    return CarlaEnv()

env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

checkpoint_cb = CheckpointCallback(
    save_freq=5000,
    save_path=CKPT_DIR,
    name_prefix="ppo_urbanzero"
)

model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=3e-4,
    vf_coef=0.25,
    max_grad_norm=0.5,
    n_steps=512,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", model.device)
print("Starting training...")
model.learn(total_timesteps=2_000_000, callback=checkpoint_cb)
model.save(os.path.expanduser("~/urbanzero/checkpoints/naive/final_model"))
print("Done.")
env.close()
