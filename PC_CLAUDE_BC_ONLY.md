# PC-Side Claude — BC-Only Plan (skip Run-4, go straight to BC)

**Context:** User elected to skip Run-4 pure-RL and go straight to BC
warmstart. Tabula-rasa constraint was explicitly relaxed in
PROJECT_NOTES.md §0. Three prior RL-from-scratch runs plateaued at
~5% RC; BC gives the policy an expert prior that ALREADY drives, then
PPO finetunes on top under the existing reward.

**Three-phase plan:**
1. **Parallel collection** — both CARLA servers collect BC data
   simultaneously. ~50k frames per port = 100k total. Wall clock
   ~45-90min (vs 90-180min serial).
2. **BC training** — Gaussian NLL on concatenated data. ~30min-2h
   on RTX 4080S.
3. **PPO finetune** — 2 envs, BC weights loaded via
   `URBANZERO_BC_WEIGHTS`. ~5M steps = ~11.5h.

**Total wall clock: 12-16h.** Deadline is Saturday EOD 2026-04-25
(~62h remaining). Comfortable margin for eval + demo + report.

**Why this is better than Run-4:**
- BehaviorAgent drives lanes and follows routes by construction, so
  the BC prior already solves "stay in right lane" and "steer toward
  waypoint" that pure-RL struggled to learn.
- Single path, no concurrent coordination, no decision gates every
  15 minutes.
- p(ship) rises from 0.62 (split path) to ~0.75 (single BC path
  succeeds) because we stop spending time on pure-RL uncertainty.

**Risks (honest):**
- BC pipeline has NEVER been run end-to-end. First-run bugs possible
  in collector, trainer, or PPO warmstart load path.
- BehaviorAgent in CARLA 0.9.15 is sometimes unstable (hangs on
  lane-change in dense traffic, slow on intersections). Mitigation:
  the collector has try/except around `run_step()` and auto-resets
  on failure.
- BC MSE/NLL on multimodal expert actions can regress to the mean
  (Codevilla 2019). Using Gaussian NLL (not pure MSE) helps.

---

## Copy-paste everything below into PC-side Claude Code as your next message

```text
Plan change: the user wants to skip Run-4 pure-RL and go straight to
BC warmstart. Tabula-rasa constraint was relaxed per PROJECT_NOTES.md
§0. Execute the three-phase BC plan.

=== HARD RULES ===
1. DO NOT edit Python/shell files. All code authority is remote Claude.
2. DO NOT commit. DO NOT push.
3. Report progress after each phase. DO NOT launch the next phase
   without confirming the prior phase completed cleanly.

=== STEP 1: kill all running jobs ===

  tmux kill-session -t urbanzero 2>/dev/null
  tmux kill-session -t wd 2>/dev/null
  tmux kill-session -t bc 2>/dev/null
  tmux kill-session -t spec0 2>/dev/null
  tmux kill-session -t spec1 2>/dev/null
  pkill -f "agents/train.py" 2>/dev/null
  pkill -f "scripts/spectator.py" 2>/dev/null
  pkill -f "scripts/watchdog.sh" 2>/dev/null
  pkill -f "scripts/collect_bc_data.py" 2>/dev/null
  pkill -f "agents/train_bc.py" 2>/dev/null
  sleep 3
  ps aux | grep -E 'train|watchdog|spectator|collect_bc' | grep -v grep
  # expect empty

=== STEP 2: pull latest tip ===

  cd ~/UrbanZero
  git fetch origin
  git checkout claude/setup-av-training-VetPV
  git pull
  git log --oneline -3

Top commit subject should reference "BC-only" or "parallel collection".

Verify all BC pipeline files exist:
  ls -la scripts/collect_bc_data.py agents/train_bc.py \
         scripts/run_bc_pipeline.sh
  # all three should be present

Verify train_bc.py supports multiple .npz inputs:
  grep -n 'nargs="\\+"' agents/train_bc.py
  # expect one match (the --data arg)

Verify the NPC fix landed (BC collection uses the same env and
benefits from moving NPCs for more diverse data):
  grep -n "global_percentage_speed_difference" env/carla_env.py
  # expect one match inside _spawn_traffic

=== STEP 3: verify both CARLA instances up ===

  for p in 2000 3000; do
    if timeout 3 bash -c ">/dev/tcp/172.25.176.1/$p" 2>/dev/null; then
      echo "port $p: UP"
    else
      echo "port $p: DOWN"
    fi
  done

If DOWN, paste relaunch block to user.

=== STEP 4: archive prior training artifacts (optional, safe) ===

  ts=$(date +%s)
  if [ -d ~/urbanzero/logs/v2_rl ]; then
    mv ~/urbanzero/logs/v2_rl ~/urbanzero/logs/v2_rl.preBC-${ts}
    mkdir -p ~/urbanzero/logs/v2_rl
  fi
  mkdir -p ~/urbanzero/bc_data
  mkdir -p ~/urbanzero/checkpoints

=== PHASE 1: PARALLEL BC COLLECTION ===

Launch two collectors simultaneously, one per CARLA port. Each
collects ~50k frames. Different seeds for diversity.

  ts=$(date +%Y%m%d_%H%M%S)
  tmux new -d -s bc0 \
    "source ~/urbanzero_env/bin/activate && \
     cd ~/UrbanZero && \
     python3 scripts/collect_bc_data.py \
       --port 2000 \
       --n_frames 50000 \
       --seed 77 \
       --output ~/urbanzero/bc_data/bc_port2000_${ts}.npz \
     2>&1 | tee ~/urbanzero/logs/bc_collect_port2000_${ts}.log"

  tmux new -d -s bc1 \
    "source ~/urbanzero_env/bin/activate && \
     cd ~/UrbanZero && \
     python3 scripts/collect_bc_data.py \
       --port 3000 \
       --n_frames 50000 \
       --seed 78 \
       --output ~/urbanzero/bc_data/bc_port3000_${ts}.npz \
     2>&1 | tee ~/urbanzero/logs/bc_collect_port3000_${ts}.log"

  tmux ls
  # expect: bc0, bc1

Save the ts variable — you need it to reference the .npz files later:
  echo "COLLECTION_TS=${ts}" > ~/urbanzero/bc_data/.latest_ts
  # For later: ts=$(cat ~/urbanzero/bc_data/.latest_ts | cut -d= -f2)

=== PHASE 1 T+5min check ===

Verify both panes are making progress:
  tmux capture-pane -t bc0 -p | tail -10
  tmux capture-pane -t bc1 -p | tail -10

Expected output in each:
  [BC-collect] 1000/50000 frames, FPS=X.X, episodes=Y
  [BC-collect] 2000/50000 frames, FPS=X.X, episodes=Y

If either pane shows an exception or is stalled with 0 frames:
  - Grep the log for the error:
      tail -30 ~/urbanzero/logs/bc_collect_port*_${ts}.log
  - Likely causes: BehaviorAgent import failure (missing CARLA
    PythonAPI path), CARLA server connection issue.
  - STOP and report to the user with the error. Do not improvise.

ALSO eyeball the CARLA viewports briefly. During BC collection with
the fixed TM code, NPCs should be moving. If NPCs are still frozen in
the viewport during BC collection, the TM fix didn't take — report
to remote Claude but DO NOT kill the collection (the ego's expert
driving is still useful data even with frozen NPCs).

=== PHASE 1 completion check (wait ~45-90min) ===

Poll for completion:
  while true; do
    p0=$(tmux capture-pane -t bc0 -p 2>/dev/null | grep -c "Done\|^\[BC-collect\].*/50000.*frames" | tail -1)
    p1=$(tmux capture-pane -t bc1 -p 2>/dev/null | grep -c "Done\|^\[BC-collect\].*/50000.*frames" | tail -1)
    echo "bc0 progress: $(tmux capture-pane -t bc0 -p 2>/dev/null | grep -E '^\[BC-collect\]' | tail -1)"
    echo "bc1 progress: $(tmux capture-pane -t bc1 -p 2>/dev/null | grep -E '^\[BC-collect\]' | tail -1)"
    if ! tmux has-session -t bc0 2>/dev/null && ! tmux has-session -t bc1 2>/dev/null; then
      echo "Both collectors done."
      break
    fi
    sleep 60
  done

(Or simpler: just check `tmux ls` every 10 minutes until both bc0
and bc1 sessions are gone.)

When both done:
  ls -lh ~/urbanzero/bc_data/
  # expect two .npz files ~500MB-1GB each

=== PHASE 1 report to user ===

Paste:
  "BC collection complete. Files:
   $(ls -lh ~/urbanzero/bc_data/bc_port*.npz)
   Total frames: ~100k across ~1000-2000 episodes.
   Launching BC training."

=== PHASE 2: BC TRAINING ===

Concatenates both .npz files with episode-boundary preservation.

  ts=$(cat ~/urbanzero/bc_data/.latest_ts 2>/dev/null | cut -d= -f2)
  if [ -z "$ts" ]; then
    # Fallback: use glob
    NPZ_FILES="$(ls ~/urbanzero/bc_data/bc_port*.npz | tr '\n' ' ')"
  else
    NPZ_FILES="$HOME/urbanzero/bc_data/bc_port2000_${ts}.npz $HOME/urbanzero/bc_data/bc_port3000_${ts}.npz"
  fi
  echo "Training on: $NPZ_FILES"

  tmux new -d -s bctrain \
    "source ~/urbanzero_env/bin/activate && \
     cd ~/UrbanZero && \
     python3 agents/train_bc.py \
       --data $NPZ_FILES \
       --output ~/urbanzero/checkpoints/bc_pretrain.zip \
       --epochs 20 \
       --batch_size 256 \
       --lr 3e-4 \
     2>&1 | tee ~/urbanzero/logs/bc_train_$(date +%Y%m%d_%H%M).log"

=== PHASE 2 T+2min check ===

  tmux capture-pane -t bctrain -p | tail -30

Expected output:
  [BC-train] device=cuda:0  data=[...]
  [BC-train] Loading /home/.../bc_port2000_...npz ...
  [BC-train] Loading /home/.../bc_port3000_...npz ...
  [BC-train] episode_starts present: NNN episodes
  [BC-train] 100000 frames loaded. Applying 4-frame stack ...
  [BC-train] Stacked shapes: images=(100000, 4, 128, 128), states=...
  [BC-train] Epoch 1/20  loss=... mean_err=... log_std=[...]

If it fails at "Loading .npz" or during architecture construction,
report the full error — this is the untested-path risk.

=== PHASE 2 completion check (wait ~30min-2h) ===

Poll:
  while tmux has-session -t bctrain 2>/dev/null; do
    echo "bctrain: $(tmux capture-pane -t bctrain -p | grep -E 'Epoch|loss=' | tail -1)"
    sleep 120
  done
  echo "BC training done."

Verify outputs exist:
  ls -la ~/urbanzero/checkpoints/bc_pretrain.zip \
         ~/urbanzero/checkpoints/bc_pretrain_vecnormalize.pkl
  # both must exist — the .zip AND the sibling vecnormalize.pkl

Report to user with final-epoch loss and mean_err:
  "BC training complete. Final loss: X.XX, mean_err: Y.YY.
   bc_pretrain.zip saved. Launching PPO finetune."

=== PHASE 3: PPO FINETUNE WITH BC WEIGHTS ===

  URBANZERO_EXP=bc_ppo_finetune \
  URBANZERO_N_ENVS=2 \
  URBANZERO_BASE_PORT=2000 \
  URBANZERO_TIMESTEPS=5000000 \
  URBANZERO_AUTO_RESUME=0 \
  URBANZERO_SEED=911 \
  URBANZERO_BC_WEIGHTS=$HOME/urbanzero/checkpoints/bc_pretrain.zip \
    bash scripts/start_training.sh

  tmux a -t urbanzero
  # Wait for the first rollout line. Then Ctrl-b d to detach.

Verify BC weights actually loaded by grepping the log:
  tail -100 $(ls -t ~/urbanzero/logs/bc_ppo_finetune/train_*.log | head -1) | \
    grep -i "BC-warmstart\|BC weights\|PPO.load\|vecnormalize"
  # expect "[BC-warmstart] loaded weights from ..." line

If the "[BC-warmstart]" log line is missing, the env var didn't take.
STOP and report — likely a path issue or the agents/train.py patch
didn't trigger.

Also launch the watchdog:
  tmux new -d -s wd 'bash scripts/watchdog.sh'

=== PHASE 3 EARLY CHECKS ===

**T+5min**: beacon sanity check. The BC prior should start showing
immediately — an agent that has seen expert driving should behave
VERY differently from pure-RL init at step 0.

Expected at ~T+5min (~36k steps):
  - avg_speed > 3 m/s from step 0 (BC taught it to throttle)
  - policy_std ∈ [0.30, 0.70] (the BC-trained log_std starts small)
  - COLLISION% < 40 (BC taught lane-following, avoids immediate crashes)
  - OFF_ROUTE% < 25 (BC taught route-following)
  - rolling_RC > 8% (BC's expert demos average ~70% RC on BehaviorAgent)

Beacon snapshot command:
  cat ~/urbanzero/beacon.json | python3 -c "
  import json,sys
  b=json.load(sys.stdin)
  tr=b.get('termination_reasons',{})
  tot=sum(tr.values()) or 1
  print(f'ts={b[\"timesteps\"]} fps={b[\"fps\"]:.1f} '
        f'speed={b[\"rolling_avg_speed_ms\"]:.3f} '
        f'ep_len={b[\"rolling_ep_len\"]:.0f} '
        f'RC={b[\"rolling_route_completion\"]:.3%} '
        f'std={b[\"policy_std\"]:.3f} '
        f'stuck%={100*tr.get(\"REALLY_STUCK\",0)/tot:.0f} '
        f'coll%={100*tr.get(\"COLLISION\",0)/tot:.0f} '
        f'offr%={100*tr.get(\"OFF_ROUTE\",0)/tot:.0f}')
  "

If T+5min RC is < 5% (same as pure-RL start), the BC prior did NOT
transfer correctly. Likely cause: VecNormalize stats mismatch, or
PPO.load didn't actually load the BC weights. STOP and report.

**T+30min**: rolling_RC ≥ 15%. If RC is climbing, PPO is improving
the BC prior under the current reward.

**T+3h**: rolling_RC ≥ 25%. At 5M-step budget and ~3h of 122 FPS
training, the agent should be in the 20-40% RC band.

**T+11h** (end of 5M budget): rolling_RC ≥ 30%. Ship
best_by_rc.zip (NOT final_model.zip — see §8 rule 7).

=== RED FLAGS (any one → STOP and report) ===

  [RED-1] BC collection exits with 0 frames (agent crashed immediately)
  [RED-2] BC train loss goes NaN or explodes (>100 within 1 epoch)
  [RED-3] BC train final mean_err > 0.3 (BC didn't learn expert actions)
  [RED-4] PPO finetune T+5min RC < 5% (BC prior didn't transfer)
  [RED-5] "[BC-warmstart]" log line missing after launch
  [RED-6] cumulative_reward_clip_hits > 0 during finetune
  [RED-7] NaN-GUARD or reward-guard fires at any phase

=== GREEN FLAGS (report as milestones) ===

  [GREEN-1] First ROUTE_COMPLETE episode in finetune log
  [GREEN-2] rolling_RC > 25% during finetune
  [GREEN-3] rolling_RC > 40% during finetune
  [GREEN-4] best_by_rc.zip updated N times (list count at end)

=== WHEN FINETUNE FINISHES ===

At timesteps ≥ 5_000_000 the training loop exits cleanly. Report:
  - Total wall-clock hours (collection + train + finetune)
  - BC final loss and mean_err
  - PPO finetune: final RC, peak RC + timestep, termination dist
  - Which checkpoint to ship: best_by_rc.zip

Kill watchdog:
  tmux kill-session -t wd

Do NOT start eval/demo yet. Remote Claude will write the eval prompt
once you confirm the finetune finished cleanly.

=== CONFIG SUMMARY ===

- Tip: claude/setup-av-training-VetPV HEAD (post-BC-only commit)
- Phase 1: 2x parallel BC collect, 50k frames each, seeds 77+78
- Phase 2: BC train, 20 epochs, Gaussian NLL, batch=256
- Phase 3: PPO finetune, N_ENVS=2, seed 911, 5M steps, BC warmstart
- All reward changes from fix-2 + Run-4 stay in effect
  (POTENTIAL_K=0.015, log_std clamp ≤ 0.7, idle_cost, persistent carrot)

The total path is 12-16h wall clock. Keep ~48h buffer for eval +
demo + report. Report phase transitions and red/green flags.
```

---

## Notes for the user (not to paste)

**What shipping looks like with BC path**:

By Saturday midday:
- `best_by_rc.zip` from the PPO finetune
- Expected final rolling RC: 25-45% (BehaviorAgent's demonstrations
  average ~70% on Town01; PPO fine-tunes from there under our reward)
- The agent WILL drive in lanes and follow the route from step 0 of
  the finetune — BehaviorAgent does this by construction
- Demo video showing one successful route-completion

**What the report will say**:
- Pure-RL result: three documented failure modes at ~5% RC each,
  with per-run analysis (you have all of this in PROJECT_NOTES)
- BC warmstart pivot rationale: tabula-rasa constraint relaxed per
  user direction, Roach 2021 / LBC 2019 / LAV 2022 precedent
- Final results: BC + PPO finetune learning curve, eval on three
  towns, termination distribution, ablation (pure-RL vs BC+PPO)

**Risk**:
- The BC pipeline has NEVER been run E2E. First-try bugs likely in
  collector (BehaviorAgent import, action-space conversion), trainer
  (VecFrameStack semantics, policy weight shapes), or PPO warmstart
  load (VecNormalize stitch). If any of these hit, we debug —
  expected 1-3h total debug time budgeted.

**If it all goes wrong**: worst case is we deliver the pure-RL
result from Run-3 (5% RC) as an "honest failure" report. That's
still a legitimate course deliverable — ECE 591 is about the
scientific process, not just peak numbers.
