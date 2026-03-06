# UrbanZero

Tabula rasa reinforcement learning for urban autonomous driving in CARLA.

## Overview
A PPO agent learns to navigate a city from scratch using only a reward signal — no expert demonstrations, no hardcoded rules. Uses semantic segmentation + vehicle state as observation space.

## Stack
- CARLA 0.9.15
- ROS2 Humble
- Stable-Baselines3 (PPO)
- Python 3.10, PyTorch (CUDA)

## Structure
- `env/` — Gymnasium wrapper for CARLA
- `agents/` — PPO training script
- `rewards/` — reward function variants
- `eval/` — generalization evaluation
- `logs/` — TensorBoard training logs

## Training
```bash
source ~/urbanzero_env/bin/activate
python3 agents/train.py
```
