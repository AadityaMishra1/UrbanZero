# UrbanZero

Tabula rasa reinforcement learning for urban autonomous driving in CARLA.

A PPO agent learns to navigate a city from scratch using only a reward signal — no expert demonstrations, no hardcoded rules. The architecture is informed by state-of-the-art approaches from CaRL (NVIDIA, CoRL 2025), Think2Drive (ECCV 2024), and Roach (ETH Zurich, ICCV 2021).

## Stack

- **CARLA 0.9.15** — urban driving simulator
- **Stable-Baselines3** — PPO implementation
- **Python 3.10**, PyTorch (CUDA)
- **ROS2 Humble** (planned bridge integration)

## Project Structure

```
UrbanZero/
├── env/
│   └── carla_env.py          # Gymnasium wrapper for CARLA with route planning,
│                              # shaped reward, traffic, weather randomization
├── agents/
│   └── train.py              # PPO training script with SubprocVecEnv,
│                              # frame stacking, custom CNN, eval metrics
├── models/
│   └── cnn_extractor.py      # 5-layer CNN with LayerNorm (replaces NatureCNN)
├── eval/
│   └── evaluator.py          # CARLA leaderboard-style metrics callback
├── rewards/                   # (reward function variants — future)
└── UrbanZero.pdf             # Project proposal
```

## Prerequisites

### 1. CARLA Simulator

Download and run CARLA 0.9.15:

```bash
# Download from https://github.com/carla-simulator/carla/releases/tag/0.9.15
# Extract and run the server:
./CarlaUE4.sh -prefernvidia -quality-level=Low
```

For multi-env training, launch multiple CARLA instances on different ports:

```bash
./CarlaUE4.sh -carla-rpc-port=2000 &
./CarlaUE4.sh -carla-rpc-port=3000 &
./CarlaUE4.sh -carla-rpc-port=4000 &
./CarlaUE4.sh -carla-rpc-port=5000 &
```

### 2. CARLA PythonAPI

The environment uses CARLA's `GlobalRoutePlanner` for navigation. The CARLA PythonAPI `agents` package must be importable:

```bash
# Option A: Install the CARLA egg (comes with CARLA download)
pip install /path/to/CARLA/PythonAPI/carla/dist/carla-0.9.15-py3.10-linux-x86_64.egg

# Option B: Add to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/CARLA/PythonAPI/carla

# Option C: Set the env var (the code checks this as fallback)
export CARLA_PYTHONAPI=/path/to/CARLA/PythonAPI/carla
```

**Important:** CARLA's PythonAPI has its own `agents/` package (for `agents.navigation.global_route_planner`). Our project also has an `agents/` directory. The code handles this conflict automatically, but if you get an `ImportError` about `GlobalRoutePlanner`, make sure the CARLA PythonAPI path is set correctly using one of the options above.

### 3. Python Dependencies

```bash
pip install carla gymnasium stable-baselines3[extra] torch numpy tensorboard
```

### 4. CARLA Host

By default the env connects to `172.25.176.1:2000`. Override with environment variables if CARLA is on a different host:

```bash
export CARLA_HOST=localhost   # if CARLA runs on the same machine
export CARLA_PORT=2000        # default port
```

## Training

### Quick Start (single environment)

```bash
# Basic training — no traffic, for initial debugging
python agents/train.py --n-envs 1 --no-traffic --timesteps 2000000 --experiment debug

# Full training with traffic + weather randomization
python agents/train.py --n-envs 1 --timesteps 10000000 --experiment shaped
```

### Parallel Training (recommended)

For serious training, run multiple CARLA instances and use `SubprocVecEnv`:

```bash
# Start 4 CARLA servers on ports 2000, 3000, 4000, 5000 first, then:
python agents/train.py --n-envs 4 --timesteps 10000000 --experiment parallel
```

### Resume from Checkpoint

```bash
python agents/train.py --resume ~/urbanzero/checkpoints/shaped/ppo_urbanzero_5000000_steps.zip
```

### All CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--n-envs` | 1 | Number of parallel CARLA environments |
| `--base-port` | 2000 | Base CARLA port (env _i_ uses `base + i*1000`) |
| `--timesteps` | 10,000,000 | Total training timesteps |
| `--no-traffic` | false | Disable NPC vehicle/pedestrian spawning |
| `--no-weather` | false | Disable weather randomization per episode |
| `--resume` | None | Path to checkpoint `.zip` to resume from |
| `--experiment` | shaped | Experiment name (determines log/checkpoint subdirectory) |

### Monitor Training

```bash
tensorboard --logdir ~/urbanzero/logs/
```

Key metrics to watch in TensorBoard:

| Metric | What It Means | Healthy Range |
|--------|--------------|---------------|
| `driving/route_completion` | Fraction of route completed per episode | Should climb toward 0.5+ after 2-3M steps |
| `driving/driving_score` | Route completion x (1 - collision rate) | Gold standard metric; >0.3 is good progress |
| `driving/collision_rate` | Fraction of episodes ending in collision | Should decrease over training |
| `driving/avg_speed_ms` | Mean speed in m/s | 3-8 m/s is reasonable urban driving |
| `rollout/ep_rew_mean` | Mean episode reward (SB3 built-in) | Should trend upward |
| `rollout/ep_len_mean` | Mean episode length | Should increase (longer = surviving longer) |

## What to Expect

### Training Progression

Training follows a rough progression through emergent behaviors:

**Steps 0 – 500K: Exploration phase**
- Agent mostly drives erratically or crashes immediately
- Episode lengths are very short (10-50 steps)
- Route completion near 0%
- Collision rate near 100%
- This is normal — the agent is exploring the action space

**Steps 500K – 2M: Basic locomotion**
- Agent learns to apply throttle and avoid immediate walls
- Starts to follow road geometry (steering toward waypoints)
- Episodes get longer (100-500 steps)
- Route completion starts climbing (5-15%)
- Steering may still be jerky/oscillating

**Steps 2M – 5M: Lane following**
- Agent learns to stay in lane and follow the road
- Action smoothness improves (less steering oscillation)
- Route completion 15-40%
- Collision rate drops below 50%
- May still run red lights and struggle with turns

**Steps 5M – 10M: Traffic awareness**
- Agent starts reacting to other vehicles (braking, yielding)
- Traffic light compliance begins to emerge
- Route completion 30-60%
- Driving becomes visually recognizable as "driving"

**Beyond 10M: Refinement**
- Better intersection handling, smoother control
- With 4+ parallel envs and 20M+ steps, expect 50-70% route completion

### What's Different from the Naive Baseline

The naive baseline (v1) had these fatal flaws:
- **No route/destination** — the agent had no concept of where to go
- **Speed reward only** — driving fast in circles was a valid optimum
- **84x84 image, 3 identical channels** — wasted 2/3 of visual capacity
- **3-element state vector** — no waypoint lookahead, no lane info
- **Single frame** — couldn't infer velocity or motion from vision
- **NatureCNN** — Atari-era architecture, insufficient for driving
- **2M steps, 1 env** — 10-100x too few samples for PPO

The v2 rewrite fixes all of these. The agent now receives a planned route, gets rewarded for progressing along it, sees future waypoints in its state vector, has temporal context via frame stacking, and uses a properly sized CNN.

### Common Issues

| Problem | Cause | Fix |
|---------|-------|-----|
| `ImportError: GlobalRoutePlanner` | CARLA PythonAPI not on path | `export CARLA_PYTHONAPI=/path/to/CARLA/PythonAPI/carla` |
| `RuntimeError: No free spawn point` | Too many actors for the map | Reduce `--num-traffic-vehicles` or restart CARLA |
| Agent drives off-road immediately | Not enough training steps | Train for at least 2M steps before evaluating |
| Reward is flat/not improving | Check TensorBoard for entropy collapse | Ensure `ent_coef=0.01` is set (it is by default) |
| CARLA connection timeout | CARLA server not running | Start CARLA first, verify host/port with env vars |
| GPU OOM with 4 parallel envs | CARLA rendering uses VRAM | Use `--no-rendering` flag when launching CARLA servers |

## Architecture Details

### Observation Space

| Component | Shape | Description |
|-----------|-------|-------------|
| **Image** | `(4, 128, 128)` | 4 stacked frames of normalized semantic segmentation labels (float32, [0,1]) |
| **State** | `(48,)` | 4 stacked frames of 12-element vector: speed, prev actions, 3 ego-frame waypoints (dx/dy), lane offset, traffic light state, route completion |

### Action Space

| Action | Range | Description |
|--------|-------|-------------|
| Steering | [-1, 1] | Full left to full right |
| Throttle/Brake | [-1, 1] | Negative = brake, positive = throttle |

### Reward Function

The reward follows CaRL's principle of route progress as the primary signal:

| Component | Weight | Description |
|-----------|--------|-------------|
| **Route progress** | 1.0 per meter | Primary signal — meters advanced along planned route |
| **Speed reward** | 0.2 | Encourages target speed (~30 km/h), penalizes speeding |
| **Heading alignment** | 0.1 | Bonus for facing the road direction |
| **Lane centering** | -0.2 | Penalty proportional to distance from lane center |
| **Action smoothness** | -0.05 | CAPS-style penalty on steering/throttle delta |
| **Stagnation** | -0.3 | Penalty after 2+ seconds of zero speed |
| **Collision** | -10.0 | Terminal penalty, episode ends |
| **Off-route** | -5.0 | Terminal if >30m from route |

### CNN Architecture

5-layer CNN with LayerNorm (inspired by CaRL, NVIDIA AVG):

```
Image (4, 128, 128) → Conv2d(4,32,5,s2) → LN → ReLU
                     → Conv2d(32,64,3,s2) → LN → ReLU
                     → Conv2d(64,128,3,s2) → LN → ReLU
                     → Conv2d(128,128,3,s2) → LN → ReLU
                     → Conv2d(128,256,3,s2) → LN → ReLU
                     → Flatten → FC(4096, 256)

State (48,) → FC(48, 64) → ReLU → FC(64, 64) → ReLU

Fusion: Cat(256, 64) → FC(320, 256) → ReLU → Actor/Critic heads
```

## References

1. CaRL — Learning Scalable Planning Policies with Simple Rewards (NVIDIA AVG, CoRL 2025)
2. Think2Drive — Efficient RL by Thinking with Latent World Model (ECCV 2024)
3. CarDreamer — Open-Source World Model Platform for Driving (2024)
4. Roach — End-to-End Urban Driving by Imitating a RL Coach (ICCV 2021)
5. CAPS — Regularizing Action Policies for Smooth Control (ICRA 2021)
6. RAD — Reinforcement Learning with Augmented Data (NeurIPS 2020)
7. Toromanoff et al. — End-to-End Model-Free RL for Urban Driving Using Implicit Affordances (CVPR 2020)
8. Schulman et al. — Proximal Policy Optimization Algorithms (2017)
9. Dosovitskiy et al. — CARLA: An Open Urban Driving Simulator (CoRL 2017)
10. Kiran et al. — Deep RL for Autonomous Driving: A Survey (IEEE TITS 2022)
