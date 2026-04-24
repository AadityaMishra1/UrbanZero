# UrbanZero

Reinforcement learning for urban autonomous driving in CARLA 0.9.15.
NCSU CSC/ECE 591 Spring 2026 final project.

**Author:** Aaditya Mishra (`amishr26@ncsu.edu`)
**Branch with full history:** `claude/setup-av-training-VetPV`
**Final deliverable result:** BC policy at **7.42% mean route completion**, **32.93% max** across 20 deterministic episodes (see `eval/bc_final_20260424_0059.json`).

---

## TL;DR for any future agent / collaborator picking this up

This project went through **6 documented training-run failures** before
identifying the actual environmental root cause via standalone
diagnostic. The deliverable is the **frozen Behavior Cloning policy**
(`checkpoints/bc_pretrain.zip` on the training PC) plus the
**scientific failure analysis** in `PROJECT_NOTES.md`.

Critical context, in priority order:
1. **`PROJECT_NOTES.md`** — the scientific memory. 50KB+ of
   per-run failure analysis with citations, pre-registered
   falsification criteria, paper references. Read this first.
2. **`eval/bc_final_20260424_0059.json`** — the final result.
3. **`DIAGNOSIS_FOR_REVIEW.md`** — self-contained technical
   briefing prepared for an external ML/RL reviewer; includes
   reward function, hyperparameters, all 6 run beacons, paper
   citations.
4. **GitHub Issues #1-#13** — per-iteration data + diagnostics.
   Issue #13 (closed) documents the root cause discovery.
5. **`PC_CLAUDE_FINAL_BC_EVAL.md`** + others — the paste blocks
   used to direct the PC-side agent that ran training/eval.

The remaining work to ship the course deliverable is in the
"What's Outstanding" section below.

---

## What This Project Is

The original goal (`UrbanZero.pdf`): train a tabula-rasa PPO agent
in CARLA to drive 200-800m urban routes from scratch using only
camera + state input and a reward signal. No expert demos, no
hardcoded rules.

This goal **was not achieved**. The project pivoted to Behavior
Cloning warmstart after pure-RL failed across 3 documented runs.
BC+PPO finetune then failed across 3 more runs due to reward-vs-BC
conflict and entropy-gradient dominance. Final deliverable is
**frozen BC evaluated deterministically**.

The **scientific contribution** is the failure analysis: 6 documented
failure modes, each with pre-registered diagnostic criteria, and
the eventual root-cause identification (`tm.set_hybrid_physics_mode
(True)` was freezing 70% of NPCs across all training runs, contaminating
every reward signal).

## Final Result

```
Frozen BC policy (bc_pretrain.zip), 20 deterministic episodes:

  RC mean:        7.42%
  RC median:      3.64%
  RC max:        32.93%   ← best individual episode
  ROUTE_COMPLETE: 0%
  COLLISION:      75%     ← distribution shift signature
  OFF_ROUTE:      25%
  Avg speed:      5.37 m/s
```

Comparison: pure-RL plateaued at ~5% mean RC across 3 runs. BC
delivers +2.4 percentage points over pure-RL ceiling, but does not
complete routes. The COLLISION rate is the documented Codevilla 2019
distribution-shift failure: BC was trained on 100k frames where NPCs
were frozen (issue #13 root cause undiscovered at collection time);
when evaluated against dynamic NPCs, BC has no learned concept of
moving obstacles.

## Tech Stack

- **CARLA 0.9.15** — urban driving simulator (Windows host)
- **stable-baselines3 2.x** — PPO + custom policy
- **PyTorch (CUDA)** — RTX 4080 Super, 16 GB VRAM
- **ROS 2 Humble** — `ros/urbanzero_node.py` publishes vehicle
  state and control to ROS topics during training
- **Python 3.10**, WSL2 (Ubuntu 24.04) running the trainer
- Hardware: Ryzen 7 9800X3D + RTX 4080 Super + Windows 11 host

## Repository Structure

```
UrbanZero/
├── env/
│   ├── carla_env.py          # Gym env wrapper for CARLA, reward function,
│   │                          # route planning, traffic spawning
│   └── safety_wrapper.py     # NaN-guard around the env
├── agents/
│   ├── train.py              # PPO trainer with BC warmstart path
│   └── train_bc.py           # Gaussian NLL BC trainer (multi-file input)
├── models/
│   ├── clamped_policy.py     # PPO policy with log_std upper clamp
│   └── cnn_extractor.py      # 5-layer CNN + state MLP fusion
├── eval/
│   ├── beacon_callback.py    # Writes ~/urbanzero/beacon.json every step
│   ├── evaluator.py          # Per-episode metrics
│   └── bc_final_20260424_0059.json   # ★ FINAL RESULT
├── ros/
│   └── urbanzero_node.py     # ROS 2 node — publishes state/control topics
├── scripts/
│   ├── collect_bc_data.py    # BehaviorAgent rollout → .npz
│   ├── eval_bc.py            # Deterministic eval (the script that produced
│   │                          # the final result)
│   ├── sanity_check_npcs.py  # Standalone NPC motion diagnostic
│   │                          # (the tool that found the root cause)
│   ├── start_training.sh     # Tmux-based training launcher
│   ├── watchdog.sh           # Restarts trainer on beacon staleness
│   ├── preflight.py          # Pre-launch infra checks
│   └── spectator.py          # Optional CARLA spectator viewer
├── run.sh                    # Single-pane launcher (training + ROS + tensorboard)
├── PROJECT_NOTES.md          # ★ Scientific memory — every failure documented
├── DIAGNOSIS_FOR_REVIEW.md   # Self-contained writeup for external review
├── PC_CLAUDE_*.md            # Iterative paste blocks used to direct PC-side
│                              # agent during training/eval (historical record)
└── README.md                 # this file
```

## Reproducing the Final Result

1. CARLA 0.9.15 server running on Windows host, port 2000
   ```bat
   cd C:\Users\<user>\ECE-591\CARLA_0.9.15\WindowsNoEditor
   .\CarlaUE4.exe -carla-rpc-port=2000
   ```

2. WSL2/Ubuntu side, with venv activated:
   ```bash
   export PYTHONPATH=$PYTHONPATH:/mnt/c/.../CARLA_0.9.15/WindowsNoEditor/PythonAPI/carla
   cd ~/UrbanZero
   python3 scripts/eval_bc.py \
     --model ~/urbanzero/checkpoints/bc_pretrain.zip \
     --episodes 20 --port 2000 --seed 1001 \
     --output ~/urbanzero/eval/bc_replay.json
   ```

3. Output JSON has the same shape as `eval/bc_final_20260424_0059.json`.

For full training reproduction (data collection → BC training → eval),
see the run history in `git log` and the `PC_CLAUDE_*.md` paste blocks.

## ROS Integration

`ros/urbanzero_node.py` runs as a ROS 2 node alongside training
(launched in `run.sh` Pane 2). Publishes 4 topics at 20 Hz:

| Topic | Type | Description |
|---|---|---|
| `/urbanzero/camera/semantic` | `sensor_msgs/Image` | Semantic-seg camera frames |
| `/urbanzero/vehicle/control` | `geometry_msgs/Twist` | linear.x = throttle-brake, angular.z = steer |
| `/urbanzero/vehicle/speed` | `std_msgs/Float32` | Ego speed in m/s |
| `/urbanzero/status` | `std_msgs/String` | Human-readable status line |

Launch separately:
```bash
source /opt/ros/humble/setup.bash
python3 ros/urbanzero_node.py
ros2 topic echo /urbanzero/status
```

---

# COURSE GRADING RUBRIC (CSC/ECE 591 Spring 2026)

Embedded here so any future agent picking up this repo knows exactly
what the deliverable is being graded against.

## Project (30% of final grade)

| Milestone | Points |
|---|---|
| Proposal | 5 |
| Status Report | 5 |
| **Final Paper** | **10** |
| **Final Video** | **10** |

**Format requirements (from syllabus):**
- Final paper: **6-8 page, 2-column IEEE or ACM format** (need LaTeX)
- Final video: **1-3 minutes, 3-minute hard max**
- Both due **Apr 26 11:59pm AOE** (= Apr 27 ~7:59am EDT)
- Submit paper: <https://docs.google.com/forms/d/e/1FAIpQLSdofyQAXAuFc6k4HEDeC-ZiKk4rLVgwYqHm9JTZ5nEvqLjRoQ/viewform>
- Submit video: <https://docs.google.com/forms/d/e/1FAIpQLSeZhx701WOPGoqqZKvjHZVOB6Ta-tN5zo8kdezFKbLMooxGgw/viewform>

## Final Paper Rubric (10 points)

| Item | Points |
|---|---|
| Abstract conveys the project | 0.667 |
| Introduction motivates the problem and identifies contributions | 1 |
| Relevant citations and references | 0.667 |
| Use of Figures to convey key concepts / results | 0.5 |
| Technical approach is explained clearly | 0.5 |
| Claims supported by evidence | 0.667 |
| **Code / Artifacts demonstrate effort** | **3.5** |
| Related Work | 0.332 |
| **Lessons learned documents the journey** | **1** |
| Polish, Spelling, Grammar | 0.5 |
| Figure / Graph Axis labels | 0.332 |
| Conclusion summarizes work | 0.332 |

For multi-person teams: each person identifies unique contributions.
This project is solo (one author).

## Final Video Rubric (10 points)

| Item | Points |
|---|---|
| Introduction motivates the problem and identifies contributions | 2 |
| Presentation is ≤ 3 minute time limit | 2 |
| **Lessons learned and documentation of the journey** | **3** |
| Polish, Spelling, Grammar | 1 |
| Conclusion summarizes work | 2 |

## Course Hard Requirements

From `csc591-software-for-robots/project/README.md`:

- **Running ROS** — every project must run ROS (ROS2/ROS1/MicroROS).
  ✅ Satisfied via `ros/urbanzero_node.py`.
- **Sensing and Actuating** — every project must sense and actuate.
  ✅ Satisfied: semantic-seg camera input, throttle+steer output.
- **Domain pick** — must pick a domain. ✅ Driving (explicitly listed
  in suggested domains).
- **GitHub repo with README listing key deliverables** — ✅ this file.

---

## What's Outstanding (TODO before submission)

For an agent picking up this repo to ship the deliverables:

| Task | Effort | Dependency |
|---|---|---|
| **Paper draft (6-8 page IEEE LaTeX)** | 4-6h | source: `PROJECT_NOTES.md` + `eval/bc_final_20260424_0059.json` |
| **3-4 figures** | 1-2h | matplotlib from beacon JSON in issues #7-#13 + final eval JSON |
| **3-min video** | 2-3h | OBS / QuickTime screen capture + voiceover |
| **Final repo polish** | 30 min | clean up old `PC_CLAUDE_*.md` if desired |

### Suggested figures (with data sources)

1. **6-run failure timeline** — RC trajectory across all runs, marking
   the diagnosed failure mode for each. Data: per-run beacons in issues
   #7-#11.
2. **Per-run beacon metrics** — RC, policy_std, approx_kl trajectories
   across the 6 runs to show the documented patterns.
3. **Final BC eval distribution** — histogram of per-episode RC from
   `eval/bc_final_20260424_0059.json`.
4. **Reward function decomposition** — bar chart showing per-step
   max reward magnitudes per term (progress, carrot, idle, shaping,
   terminals).

### Paper section outline (mapped to rubric)

- **Abstract** (0.667 pt) — frame as failure-driven scientific result
- **Introduction + contributions** (1 pt) — three contributions:
  full pipeline, documented failure modes, root cause via standalone
  diagnostic
- **Related Work** (0.332 pt) — CaRL, Roach, LBC, LAV, Ng 1999,
  Codevilla 2019, Andrychowicz 2021 — all in `PROJECT_NOTES.md §2`
- **Technical Approach** (0.5 pt) — env, reward, BC pipeline,
  diagnostic methodology
- **Results / Evidence** (0.667 + 0.5 + 0.332 pt) — final BC eval table,
  per-run failure data, root cause discovery
- **Lessons Learned** (1 pt) — directly from `PROJECT_NOTES.md §11`
- **Conclusion** (0.332 pt) — what worked, what didn't, what's next

### Video script outline (mapped to rubric)

- 0:00-0:30 — Hook + problem motivation + contributions (2 pts)
- 0:30-1:30 — 6-failure timeline + root cause discovery (3 pts —
  the strongest content)
- 1:30-2:30 — Final BC result + comparison to pure-RL (2 pts conclusion)
- 2:30-3:00 — Future work and one-line takeaway

Hard cap at 3 minutes. Practice timing.

---

## How to Pick This Up Without Context (for ClaudeCoWork)

If you are an AI agent or collaborator pulling this repo fresh:

1. **Read `PROJECT_NOTES.md` end to end** (~30 min) — this is the
   scientific memory. Every commit, every failure mode, every fix
   attempt is documented with rationale and citations.
2. **Read `eval/bc_final_20260424_0059.json`** — this is the final
   result. The aggregate block has the headline numbers.
3. **Look at GitHub issues #7-#13** for raw beacon data per training
   run. Issue #13 (closed) explains the root cause discovery.
4. **Do NOT re-attempt training**. Six runs failed. The hybrid_physics
   bug is fixed but BC was trained on contaminated data; PPO finetune
   under the current pipeline was definitively shown unstable. Effort
   should go to writeup, not more experiments.
5. **The deliverable is the paper + video** per the rubric above.
   Source material for both is in `PROJECT_NOTES.md` and the eval JSON.
6. **No co-author tags from AI assistants on commits to main.**
   Author commits as the user only.

## Submission Checklist

- [ ] Paper PDF (6-8 page IEEE/ACM 2-column) uploaded via Google Form
- [ ] Video (≤3 min) uploaded via Google Form
- [ ] Repo public / accessible to grader
- [ ] README current and rubric-aligned (this file)
- [ ] Deadline: Apr 26 11:59pm AOE (Apr 27 7:59am EDT)
