# BC Pipeline Status Report

**Date:** 2026-04-23 02:03–02:32 UTC-4
**Branch:** `claude/setup-av-training-VetPV`
**Tip:** `2413ad6`

## PHASE 1: BC Collection — COMPLETE

Two parallel collectors, one per CARLA port. BehaviorAgent("normal") expert.

| Port | Frames | Episodes | Wall time | FPS | Seed | File size |
|------|--------|----------|-----------|-----|------|-----------|
| 2000 | 50,000 | 88 | 6.6 min | 139 | 77 | 71 MB |
| 3000 | 50,000 | 82 | 6.8 min | 132 | 78 | 72 MB |

**Total: 100,000 frames across 170 episodes.**

Note: Required `pip install shapely` — BehaviorAgent depends on it and it wasn't in the venv.

**NPC status: STILL FROZEN (Issue #9).** TM logs show `hybrid_physics=True radius=70m` and `speed_diff=-30%` but NPCs remain stationary. BC data was collected with static NPC traffic only. Expert ego data is still valid for teaching lane-following + steering.

## PHASE 2: BC Training — COMPLETE

Trained on port 2000 data only (50k frames). Using both files (100k) caused OOM — the 4-frame stacking creates a (100000, 4, 128, 128) float32 array that exceeds 30GB RAM.

**Architecture:** DrivingCNN (5-layer CNN + state MLP + fusion) → MLP → 2D Gaussian action head.
**Params optimized:** 2,258,404 / 2,357,221 (value head frozen).
**Loss:** Gaussian NLL. **Optimizer:** Adam, lr=3e-4, batch=256, 195 batches/epoch.

### Epoch-by-epoch training log

```
epoch 001/20  NLL=-0.8438  MAE=0.1951  log_std=[-0.561, -0.565]  std=[0.571, 0.568]
epoch 002/20  NLL=-0.9994  MAE=0.1633  log_std=[-0.618, -0.624]  std=[0.539, 0.536]
epoch 003/20  NLL=-1.1167  MAE=0.1506  log_std=[-0.673, -0.680]  std=[0.510, 0.507]
epoch 004/20  NLL=-1.2357  MAE=0.1351  log_std=[-0.725, -0.733]  std=[0.484, 0.480]
epoch 005/20  NLL=-1.3632  MAE=0.1180  log_std=[-0.774, -0.787]  std=[0.461, 0.455]
epoch 006/20  NLL=-1.4829  MAE=0.1034  log_std=[-0.821, -0.840]  std=[0.440, 0.432]
epoch 007/20  NLL=-1.6025  MAE=0.0911  log_std=[-0.869, -0.894]  std=[0.419, 0.409]
epoch 008/20  NLL=-1.7196  MAE=0.0830  log_std=[-0.919, -0.952]  std=[0.399, 0.386]
epoch 009/20  NLL=-1.8379  MAE=0.0765  log_std=[-0.971, -1.011]  std=[0.379, 0.364]
epoch 010/20  NLL=-1.9502  MAE=0.0725  log_std=[-1.022, -1.068]  std=[0.360, 0.344]
epoch 011/20  NLL=-2.0588  MAE=0.0691  log_std=[-1.074, -1.127]  std=[0.342, 0.324]
epoch 012/20  NLL=-2.1672  MAE=0.0662  log_std=[-1.125, -1.183]  std=[0.325, 0.306]
epoch 013/20  NLL=-2.2710  MAE=0.0634  log_std=[-1.175, -1.238]  std=[0.309, 0.290]
epoch 014/20  NLL=-2.3742  MAE=0.0613  log_std=[-1.228, -1.294]  std=[0.293, 0.274]
epoch 015/20  NLL=-2.4709  MAE=0.0598  log_std=[-1.275, -1.345]  std=[0.279, 0.261]
epoch 016/20  NLL=-2.5662  MAE=0.0578  log_std=[-1.321, -1.394]  std=[0.267, 0.248]
epoch 017/20  NLL=-2.6622  MAE=0.0552  log_std=[-1.369, -1.445]  std=[0.254, 0.236]
epoch 018/20  NLL=-2.7517  MAE=0.0541  log_std=[-1.413, -1.491]  std=[0.243, 0.225]
epoch 019/20  NLL=-2.8439  MAE=0.0515  log_std=[-1.458, -1.540]  std=[0.233, 0.214]
epoch 020/20  NLL=-2.9321  MAE=0.0503  log_std=[-1.503, -1.587]  std=[0.223, 0.205]
```

**Final: NLL=-2.9321, MAE=0.0503, std=[0.223, 0.205].**
MAE well under 0.3 RED flag. Loss monotonically decreased every epoch.

**Output:** `checkpoints/bc_pretrain.zip` (10 MB)

## PHASE 3: PPO Finetune with BC Warmstart — RUNNING

### Launch issues encountered and resolved

1. **`start_training.sh` does not forward `URBANZERO_BC_WEIGHTS` env var** into the tmux session. The ACTIVATE string on line 43 only exports PYTHONPATH and URBANZERO_SEED. Workaround: launched tmux manually with explicit `export URBANZERO_BC_WEIGHTS=...` in the command string.

2. **`env.obs_rms` AttributeError on BC warmstart** — `train.py` line 419 tries to restore VecNormalize obs stats from `bc_pretrain_vecnormalize.pkl`, but the training env uses `norm_obs=False` (reward-only VecNormalize), so `obs_rms` doesn't exist. Workaround: moved `bc_pretrain_vecnormalize.pkl` aside so `train.py` takes the "not found" else branch at line 423. **Remote Claude should fix the code: either guard `env.obs_rms` assignment with `hasattr`, or don't save obs_rms in train_bc.py when norm_obs=False.**

### BC warmstart confirmation

```
[BC-warmstart] loading weights from /home/aadityamishra/urbanzero/checkpoints/bc_pretrain.zip
[BC-warmstart] /home/aadityamishra/urbanzero/checkpoints/bc_pretrain_vecnormalize.pkl not found — using initial stats
```

### Beacon at 28k steps (~4 min into finetune)

```json
{
    "experiment": "bc_ppo_finetune",
    "timesteps": 28144,
    "fps": 116.24,
    "rolling_ep_return": -8.662,
    "rolling_ep_len": 222.0,
    "rolling_route_completion": 0.0556,
    "rolling_collision_rate": 0.68,
    "rolling_avg_speed_ms": 7.599,
    "termination_reasons": {"OFF_ROUTE": 15, "COLLISION": 34, "REALLY_STUCK": 1},
    "policy_std": 0.2222,
    "approx_kl": 0.07924,
    "clip_fraction": 0.3418,
    "entropy_loss": 0.1717,
    "explained_variance": 0.6898,
    "ent_coef": 0.01997,
    "cumulative_reward_clip_hits": 0,
    "total_episodes": 141
}
```

### Early observations (28k steps, 141 episodes)

**BC prior IS showing:**
- `policy_std=0.222` — tight from step 0 (pure RL starts at 0.6-1.0). BC taught confident actions.
- `avg_speed=7.6 m/s` — agent drives fast from step 0 (pure RL started at 0-3 m/s and took 100k+ steps to move)
- `explained_variance=0.69` — value function explains 69% of return variance (pure RL was at 0.36)
- `REALLY_STUCK: 1/50` — virtually no sitting still (pure RL had 8-14/50)

**Concerns:**
- `rolling_RC=5.56%` — still in the ~5% band. BC taught throttle but steering hasn't translated yet.
- `collision_rate=0.68` — very high. Agent drives fast into things. BC may have taught speed but not obstacle avoidance (NPCs were frozen during collection).
- `approx_kl=0.079` — HIGH. PPO is making large policy updates. Risk of catastrophic forgetting of BC prior. May need KL penalty or lower lr for finetuning.
- `clip_fraction=0.34` — also very high (healthy range is 0.05-0.15). Same concern.

**Some promising episodes in the data:**
```
COLLISION steps=285 (14.2s) speed=2.1m/s route=30.0% progress=49.1m/164m
COLLISION steps=498 (24.9s) speed=1.1m/s route=38.4% progress=124.2m/324m
COLLISION steps=137 (6.9s) speed=3.2m/s route=36.1% progress=53.1m/147m
COLLISION steps=353 (17.7s) speed=1.7m/s route=26.1% progress=108.1m/414m
COLLISION steps=188 (9.4s) speed=2.4m/s route=20.9% progress=58.8m/281m
COLLISION steps=169 (8.5s) speed=1.6m/s route=19.9% progress=32.2m/162m
```

Multiple episodes hitting 20-38% RC — significantly better than any pure RL run. But they all end in COLLISION. The agent is following the route better (BC steering prior) but crashing into frozen NPCs.

### Episode data (last 40 episodes)

```
COLLISION steps=42 (2.1s) speed=2.1m/s route=5.6% progress=7.7m/138m
COLLISION steps=109 (5.5s) speed=2.5m/s route=4.9% progress=12.5m/254m
COLLISION steps=256 (12.8s) speed=1.9m/s route=1.2% progress=4.1m/344m
COLLISION steps=124 (6.2s) speed=3.5m/s route=3.7% progress=9.3m/249m
COLLISION steps=173 (8.7s) speed=1.3m/s route=1.0% progress=3.9m/384m
COLLISION steps=141 (7.1s) speed=3.3m/s route=6.3% progress=22.1m/351m
COLLISION steps=196 (9.8s) speed=1.4m/s route=10.2% progress=31.1m/304m
COLLISION steps=227 (11.4s) speed=2.5m/s route=3.7% progress=12.5m/338m
OFF_ROUTE steps=178 (8.9s) speed=8.7m/s route=2.3% progress=11.3m/489m
COLLISION steps=277 (13.9s) speed=2.4m/s route=1.5% progress=4.6m/303m
COLLISION steps=254 (12.7s) speed=1.3m/s route=0.0% progress=0.0m/351m
OFF_ROUTE steps=368 (18.4s) speed=7.7m/s route=3.5% progress=10.4m/297m
COLLISION steps=128 (6.4s) speed=3.9m/s route=3.4% progress=14.3m/416m
COLLISION steps=146 (7.3s) speed=6.9m/s route=1.4% progress=4.5m/317m
COLLISION steps=285 (14.2s) speed=2.1m/s route=30.0% progress=49.1m/164m
COLLISION steps=97 (4.9s) speed=2.7m/s route=3.3% progress=10.7m/318m
COLLISION steps=169 (8.5s) speed=1.6m/s route=19.9% progress=32.2m/162m
OFF_ROUTE steps=329 (16.4s) speed=9.8m/s route=0.0% progress=0.0m/351m
COLLISION steps=188 (9.4s) speed=2.4m/s route=20.9% progress=58.8m/281m
OFF_ROUTE steps=174 (8.7s) speed=8.3m/s route=5.0% progress=12.1m/244m
OFF_ROUTE steps=201 (10.1s) speed=8.5m/s route=2.8% progress=10.0m/363m
COLLISION steps=137 (6.9s) speed=3.2m/s route=36.1% progress=53.1m/147m
OFF_ROUTE steps=164 (8.2s) speed=10.3m/s route=10.5% progress=28.3m/270m
COLLISION steps=173 (8.7s) speed=1.5m/s route=0.0% progress=0.0m/302m
COLLISION steps=208 (10.4s) speed=1.8m/s route=4.8% progress=8.7m/181m
COLLISION steps=178 (8.9s) speed=2.2m/s route=10.2% progress=17.0m/167m
COLLISION steps=350 (17.5s) speed=1.7m/s route=14.5% progress=23.6m/163m
COLLISION steps=102 (5.1s) speed=2.1m/s route=0.8% progress=3.5m/436m
OFF_ROUTE steps=390 (19.5s) speed=6.3m/s route=7.3% progress=21.5m/295m
COLLISION steps=160 (8.0s) speed=1.0m/s route=9.8% progress=43.1m/439m
COLLISION steps=296 (14.8s) speed=3.9m/s route=1.1% progress=3.6m/343m
OFF_ROUTE steps=413 (20.7s) speed=5.7m/s route=3.1% progress=8.7m/278m
OFF_ROUTE steps=232 (11.6s) speed=5.1m/s route=4.2% progress=10.6m/253m
OFF_ROUTE steps=750 (37.5s) speed=5.0m/s route=0.0% progress=0.0m/413m
COLLISION steps=99 (5.0s) speed=2.2m/s route=3.3% progress=10.7m/318m
COLLISION steps=498 (24.9s) speed=1.1m/s route=38.4% progress=124.2m/324m
OFF_ROUTE steps=515 (25.8s) speed=6.4m/s route=5.2% progress=27.2m/527m
COLLISION steps=353 (17.7s) speed=1.7m/s route=26.1% progress=108.1m/414m
COLLISION steps=296 (14.8s) speed=1.9m/s route=4.7% progress=15.9m/338m
OFF_ROUTE steps=945 (47.2s) speed=4.8m/s route=2.5% progress=10.4m/420m
```

## Issues for Remote Claude

1. **BUG: `train.py` line 419 `env.obs_rms` AttributeError** — when BC vecnormalize has obs stats but training env uses `norm_obs=False`. Guard with `hasattr` or skip obs_rms restore.

2. **BUG: `start_training.sh` line 43 doesn't forward `URBANZERO_BC_WEIGHTS`** — add `export URBANZERO_BC_WEIGHTS='${URBANZERO_BC_WEIGHTS:-}'` to the ACTIVATE string.

3. **Missing dep: `shapely`** — add to requirements or document for BehaviorAgent.

4. **OOM on 100k frames**: `_stack_frames()` in `train_bc.py` allocates (N, 4, 128, 128) float32 in one shot. For 100k frames that's ~25GB. Either chunk the stacking or support memory-mapped arrays.

5. **HIGH approx_kl=0.079 and clip_fraction=0.34** — PPO is aggressively updating away from the BC prior. Consider adding KL penalty term or reducing lr (1e-4 instead of 3e-4) for the finetune phase to preserve the BC prior longer.

6. **NPCs still frozen (Issue #9)** — not fixed by the TM changes. All collisions are with static obstacles.
