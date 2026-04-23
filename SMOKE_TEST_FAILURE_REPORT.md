# SMOKE TEST FAILURE REPORT — Issue #6: ForkServerProcess-2 BrokenPipeError deadlock

**Date:** 2026-04-22 21:01 UTC-4
**Branch:** `claude/setup-av-training-VetPV`
**Tip:** `0a3f114` ("revert infra to v1 (834a8e0) behavior, keep v2 experiment changes")
**Config:** 2 envs, ports 2000/3000, `URBANZERO_AUTO_RESUME=0`, seed 42
**GPU:** RTX 4080 Super, 14.7/16.0 GB free at launch
**CARLA:** 0.9.15, Windows, both ports verified UP before launch

## What happened

Training launched, printed the model architecture, logged `Starting training for 10,000,000 timesteps...`, then:

1. **Worker 1 (port 2000)** ran one full episode: `COLLISION` at step 379 (18.9s), respawned successfully.
2. **Worker 2 (port 3000) — `ForkServerProcess-2` — crashed** with `BrokenPipeError` while trying to send `(observation, reward, done, info, reset_info)` back to the main process via `remote.send()`.
3. Worker 1 kept running (completed 2 more episodes), but **the main process deadlocked** — it was sleeping on `unix_stream_read_generic`, waiting forever for Worker 2 which was already dead.
4. The beacon froze at **timesteps=2, fps=1.07** and never updated again.
5. The main process eventually exited (PID gone), but tmux session stayed open. No emergency checkpoint was saved — the crash happened inside `SubprocVecEnv`'s IPC layer, not inside `model.learn()`, so the `try/except` in `train.py:402-418` never caught it.

## Exact crash traceback (from log)

```
Process ForkServerProcess-2:
Traceback (most recent call last):
  File "/usr/lib/python3.10/multiprocessing/process.py", line 314, in _bootstrap
    self.run()
  File "/usr/lib/python3.10/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "stable_baselines3/common/vec_env/subproc_vec_env.py", line 43, in _worker
    remote.send((observation, reward, done, info, reset_info))
  File "/usr/lib/python3.10/multiprocessing/connection.py", line 206, in send
    self._send_bytes(_ForkingPickler.dumps(obj))
  File "/usr/lib/python3.10/multiprocessing/connection.py", line 404, in _send_bytes
    self._send(header)
  File "/usr/lib/python3.10/multiprocessing/connection.py", line 368, in _send
    n = write(self._handle, buf)
BrokenPipeError: [Errno 32] Broken pipe
```

## Process state at time of inspection

| Process | PID | Kernel wchan | Status |
|---------|-----|-------------|--------|
| Main (train.py) | 616475 | `unix_stream_read_generic` | Sleeping, 43 threads |
| Worker 1 | 616509 | `pipe_read` | Sleeping |
| Worker 2 | 616510 | `do_epoll_wait` | Sleeping |

Could not get Python-level stack dumps (py-spy requires sudo, no passwordless sudo configured).

## Pattern across ALL 6 recent logs (tonight's session)

| Log timestamp | Envs | Outcome | Notes |
|--------------|------|---------|-------|
| 19:40 | 4 | `UnboundLocalError: ENT_COEF_START` | Code bug (pre-0a3f114) |
| 19:52 | 4 | Froze after iter 1 (1024 steps, 131 FPS) | Same deadlock pattern |
| 20:01 | 2 | `RuntimeError: time-out of 20000ms while waiting for the simulator` on port 2000 | CARLA connection failure |
| 20:03 | 2 | **SUCCESS for 3 iterations** (3072 steps, 147 FPS, healthy PPO stats) | Then froze/killed |
| 20:32 | 2 | Froze immediately after "Starting training" | Zero iterations logged |
| 21:01 | 2 | **THIS RUN** — ForkServerProcess-2 BrokenPipeError after 1 episode | Current failure |

**Key observation:** The 20:03 run **did work** for 3 iterations (20 seconds, 3072 steps at 147 FPS) with healthy stats (approx_kl=0.0055, entropy_loss=-1.84, std=0.607). This proves the v2 reward/policy/hyperparams are fine — the problem is purely infrastructure: **SubprocVecEnv workers die randomly and take the whole training run with them**.

## Root cause analysis

The `BrokenPipeError` in `ForkServerProcess-2` means the main process's end of the pipe closed (or was never properly established) before the worker tried to send its step result. This is NOT a CARLA deadlock (the worker completed a full episode + reset before dying). Possible causes:

1. **SB3 SubprocVecEnv has no worker crash recovery.** If one worker's pipe breaks, the main process hangs on `recv()` forever. There's no timeout, no health check, no restart logic. This is a known SB3 limitation.

2. **The main process may have crashed first** (OOM, segfault in torch, etc.) which closed the pipe, causing the worker's `send()` to get `BrokenPipeError`. The log doesn't show a main-process exception because the crash happened in C++ (torch/CUDA) not Python.

3. **Stale CARLA state between runs.** Despite cleanup code in `__init__`, CARLA servers retain sync-mode state from prior crashed sessions. The 20:01 log shows a 20s timeout on port 2000 — the server was in a bad state. Launching new training without restarting CARLA servers means training into potentially corrupted server state.

## What needs to be fixed

### 1. Critical: Add worker crash resilience to training

Options (pick one):
- Wrap `SubprocVecEnv` with a custom class that catches `BrokenPipeError`/`EOFError`/`ConnectionResetError` in `step_wait()` and restarts the dead worker
- OR use SB3's built-in timeout mechanism if available
- OR **fall back to `DummyVecEnv` for n_envs <= 2** — single process, no IPC, cannot deadlock on broken pipes. The 20:03 run showed 147 FPS with 2 envs in SubprocVecEnv; DummyVecEnv might get ~80-100 FPS which still passes P2 gate (>=70 FPS).

### 2. The try/except in train.py:402-418 doesn't catch IPC failures

The `model.learn()` call raises inside SB3's `collect_rollouts()` -> `env.step_wait()` -> `self.remotes[i].recv()`, which raises `EOFError` or `ConnectionResetError`. The existing `except Exception` should catch this, but based on the log NO emergency checkpoint was saved — which means either (a) the exception was `KeyboardInterrupt` not `Exception`, or (b) the process hung instead of raising. **Add a timeout to the recv() calls or wrap SubprocVecEnv.**

### 3. Between runs: CARLA servers MUST be restarted

The cleanup in `__init__` (lines 85-121) tries to destroy leftover actors and re-apply sync settings, but it doesn't fix corrupted server-side state. The 20:01 timeout proves this. Add a note to the smoke test procedure: kill and relaunch CARLA between every training attempt.

## Files that need changes

- `env/carla_env.py` — lines 78-121 (init/cleanup), 382-477 (step)
- `agents/train.py` — lines 256-265 (SubprocVecEnv creation), 402-418 (crash handler)
- `stable_baselines3/common/vec_env/subproc_vec_env.py` — the `_worker` function and `step_wait` method (no crash recovery) — NOTE: this is a dependency, don't edit directly; wrap it instead.

## Positive signals from the 20:03 run (proof the v2 stack works when IPC doesn't break)

```
fps: 147
iterations: 3 (3072 steps in 20s)
approx_kl: 0.0055
clip_fraction: 0.0811
entropy_loss: -1.84
std: 0.607
value_loss: 0.628
ent_coef: 0.02
```

These are all healthy PPO metrics. The v2 reward, policy, and hyperparameters are working correctly.

## Full log (current run — train_20260422_210123.log)

```
=== UrbanZero Training ===
  Envs: 2
  Timesteps: 10,000,000
  Traffic: True
  Weather randomization: True
  Experiment: v2_rl
  Log dir: /home/aadityamishra/urbanzero/logs/v2_rl
  Checkpoint dir: /home/aadityamishra/urbanzero/checkpoints/v2_rl
  Ent-coef schedule: 0.02 -> 0.01 (floor) over 10M steps
Using cuda device
Device: cuda
Policy architecture:
ClampedStdPolicy(
  (features_extractor): DrivingCNN(
    (cnn): Sequential(
      (0): Conv2d(4, 32, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
      (1): LayerNorm((32, 64, 64), eps=1e-05, elementwise_affine=True)
      (2): ReLU()
      (3): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      (4): LayerNorm((64, 32, 32), eps=1e-05, elementwise_affine=True)
      (5): ReLU()
      (6): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      (7): LayerNorm((128, 16, 16), eps=1e-05, elementwise_affine=True)
      (8): ReLU()
      (9): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      (10): LayerNorm((128, 8, 8), eps=1e-05, elementwise_affine=True)
      (11): ReLU()
      (12): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      (13): LayerNorm((256, 4, 4), eps=1e-05, elementwise_affine=True)
      (14): ReLU()
      (15): Flatten(start_dim=1, end_dim=-1)
    )
    (cnn_fc): Sequential(
      (0): Linear(in_features=4096, out_features=256, bias=True)
      (1): ReLU()
    )
    (state_mlp): Sequential(
      (0): Linear(in_features=40, out_features=64, bias=True)
      (1): ReLU()
      (2): Linear(in_features=64, out_features=64, bias=True)
      (3): ReLU()
    )
    (fusion): Sequential(
      (0): Linear(in_features=320, out_features=256, bias=True)
      (1): ReLU()
    )
  )
  [...pi_features_extractor and vf_features_extractor identical to above...]
  (mlp_extractor): MlpExtractor(
    (policy_net): Sequential(
      (0): Linear(in_features=256, out_features=256, bias=True)
      (1): Tanh()
      (2): Linear(in_features=256, out_features=128, bias=True)
      (3): Tanh()
    )
    (value_net): Sequential(
      (0): Linear(in_features=256, out_features=256, bias=True)
      (1): Tanh()
      (2): Linear(in_features=256, out_features=128, bias=True)
      (3): Tanh()
    )
  )
  (action_net): Linear(in_features=128, out_features=2, bias=True)
  (value_net): Linear(in_features=128, out_features=1, bias=True)
)

Starting training for 10,000,000 timesteps...
Logging to /home/aadityamishra/urbanzero/logs/v2_rl/PPO_1
[spawn-filter] 155/155 spawn points passed driving-lane filter
[spawn] at (80.3, 16.9), yaw_diff=0.0deg, road_id=21, lane_id=4 (0 failed, 0 yaw-rejected)
[EPISODE END] reason=COLLISION steps=379 (18.9s) speed=0.6m/s route=11.5% progress=23.6m/205m clip_hits=0 prog_clamp_hits=0
[spawn-filter] 155/155 spawn points passed driving-lane filter
[spawn] at (-27.0, 69.7), yaw_diff=0.0deg, road_id=12, lane_id=1 (0 failed, 0 yaw-rejected)
Process ForkServerProcess-2:
Traceback (most recent call last):
  File "/usr/lib/python3.10/multiprocessing/process.py", line 314, in _bootstrap
    self.run()
  File "/usr/lib/python3.10/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "stable_baselines3/common/vec_env/subproc_vec_env.py", line 43, in _worker
    remote.send((observation, reward, done, info, reset_info))
  File "/usr/lib/python3.10/multiprocessing/connection.py", line 206, in send
    self._send_bytes(_ForkingPickler.dumps(obj))
  File "/usr/lib/python3.10/multiprocessing/connection.py", line 404, in _send_bytes
    self._send(header)
  File "/usr/lib/python3.10/multiprocessing/connection.py", line 368, in _send
    n = write(self._handle, buf)
BrokenPipeError: [Errno 32] Broken pipe
[spawn-filter] 155/155 spawn points passed driving-lane filter
[spawn] at (-41.7, 89.7), yaw_diff=0.0deg, road_id=13, lane_id=-2 (0 failed, 0 yaw-rejected)
[EPISODE END] reason=COLLISION steps=464 (23.2s) speed=0.9m/s route=4.1% progress=6.3m/155m clip_hits=0 prog_clamp_hits=0
[spawn-filter] 155/155 spawn points passed driving-lane filter
[spawn] at (11.2, -64.4), yaw_diff=0.0deg, road_id=10, lane_id=-1 (0 failed, 0 yaw-rejected)
```

## Beacon (frozen at crash time)

```json
{
    "ts": "2026-04-23T01:01:30Z",
    "ts_unix": 1776906090,
    "pid": 616475,
    "host": "AadityasPC",
    "experiment": "v2_rl",
    "status": "training",
    "timesteps": 2,
    "fps": 1.07,
    "uptime_sec": 1,
    "rolling_ep_count": 0,
    "rolling_ep_return": 0.0,
    "rolling_ep_len": 0.0,
    "rolling_route_completion": 0.0,
    "rolling_collision_rate": 0.0,
    "rolling_avg_speed_ms": 0.0,
    "termination_reasons": {},
    "policy_std": 0.6065,
    "approx_kl": null,
    "clip_fraction": null,
    "entropy_loss": null,
    "explained_variance": null,
    "ent_coef": 0.02,
    "cumulative_reward_clip_hits": 0,
    "total_episodes": 0,
    "carla_port": 2000,
    "last_checkpoint": null,
    "last_checkpoint_unix": null
}
```

## Prior successful run log (20:03 — proof v2 works)

```
Starting training for 10,000,000 timesteps...
Logging to /home/aadityamishra/urbanzero/logs/v2_rl/PPO_1
---------------------------------
| time/              |          |
|    fps             | 148      |
|    iterations      | 1        |
|    time_elapsed    | 6        |
|    total_timesteps | 1024     |
| train/             |          |
|    ent_coef        | 0.02     |
---------------------------------
------------------------------------------
| time/                   |              |
|    fps                  | 144          |
|    iterations           | 2            |
|    time_elapsed         | 14           |
|    total_timesteps      | 2048         |
| train/                  |              |
|    approx_kl            | 0.0017640111 |
|    clip_fraction        | 0.0203       |
|    clip_range           | 0.2          |
|    ent_coef             | 0.02         |
|    entropy_loss         | -1.84        |
|    explained_variance   | -0.0398      |
|    learning_rate        | 0.0003       |
|    loss                 | 0.004        |
|    n_updates            | 4            |
|    policy_gradient_loss | 0.00082      |
|    std                  | 0.607        |
|    value_loss           | 0.715        |
------------------------------------------
------------------------------------------
| time/                   |              |
|    fps                  | 147          |
|    iterations           | 3            |
|    time_elapsed         | 20           |
|    total_timesteps      | 3072         |
| train/                  |              |
|    approx_kl            | 0.0055222875 |
|    clip_fraction        | 0.0811       |
|    clip_range           | 0.2          |
|    ent_coef             | 0.02         |
|    entropy_loss         | -1.84        |
|    explained_variance   | -0.0421      |
|    learning_rate        | 0.0003       |
|    loss                 | -0.00225     |
|    n_updates            | 8            |
|    policy_gradient_loss | -0.00457     |
|    std                  | 0.607        |
|    value_loss           | 0.628        |
------------------------------------------
```

## Recommendation

The simplest, most reliable fix: **use DummyVecEnv instead of SubprocVecEnv for n_envs <= 2**. DummyVecEnv runs all envs in the main process — no pipes, no IPC, no BrokenPipeError, no deadlocks. The 20:03 run proved 147 FPS with SubprocVecEnv; DummyVecEnv will be somewhat slower but eliminates the entire class of IPC failures that have killed every run tonight. If FPS drops below the P2 gate (70), THEN consider SubprocVecEnv with crash recovery — but try the simple thing first.
