# PC-Side Claude — Run-4 + Parallel BC Collection

**Context:** Run-3 at tip `ff7a1e1` failed at 104k steps with same
~5% RC plateau as Runs 1 and 2. NPCs visibly frozen in both CARLA
viewports despite `tm.set_hybrid_physics_mode(True)`. Remote Claude
delegated the next-step decision to a fresh no-context subagent.

**Subagent's decision:** kill Run-3; launch Run-4 with TWO fixes;
in PARALLEL run BC data collection on the second CARLA server. Stop
serializing fallbacks — hedge the deadline with both paths running
simultaneously.

**Fixes pushed in the new tip (see `git log --oneline -1` after pull):**

1. `env/carla_env.py::POTENTIAL_K`: 0.03 → 0.015. Max |F|/step
   drops to ~0.0105 so progress_reward dominates shaping 2:1.
   Rationale: at K=0.03 the shaping magnitude equaled progress_reward,
   meaning Φ could subsidize any motion that reduced dist-to-lookahead
   — including perpendicular approach from off-route spawns. Subtle
   Ng-compliant echo of the 7M-run circling attractor.

2. `env/carla_env.py::_spawn_traffic`: NPC motion fix for frozen
   traffic. Three components:
   - `tm.global_percentage_speed_difference(-30.0)` — CARLA default
     is +30 (drive 30% BELOW limit). In zero-speed zones that's 0
     desired speed. Negative forces NPCs ABOVE limit.
   - Per-vehicle `tm.vehicle_percentage_speed_difference(-20)`,
     `auto_lane_change(True)`, `ignore_lights_percentage(0)`
   - **Commit `self.world.tick()`** after the spawn loop — in sync
     mode `set_autopilot()` is async; without a commit tick the
     TM's vehicle-registration table races the first env.step()
     and NPCs can stay unregistered for the whole episode.

**Run-4 config:** N_ENVS=1 (not 2). The single-env run uses port
2000 only so port 3000 is free for parallel BC data collection.
Single env at ~60 FPS is slower than 2×60=120, but Run-4 is the
**decision-gate run** — gates fire at T+10/45/90min. If it passes,
we scale up; if it fails, BC pipeline is already warm.

---

## Copy-paste everything below into PC-side Claude Code as your next message

```text
The remote Claude has pushed Run-4 fixes after consulting a fresh
no-context subagent. Kill Run-3 and launch Run-4 + parallel BC.

=== HARD RULES (unchanged) ===
1. DO NOT edit Python/shell files. All code authority is remote Claude.
2. DO NOT commit. DO NOT push.
3. DO NOT resume from any prior checkpoint.
4. DO NOT add new reward shaping terms. Remote Claude explicitly
   ruled out cos-heading shaping (7M-run trap).

=== STEP 1: kill Run-3 ===

  tmux kill-session -t urbanzero 2>/dev/null
  tmux kill-session -t wd 2>/dev/null
  tmux kill-session -t spec0 2>/dev/null
  tmux kill-session -t spec1 2>/dev/null
  pkill -f "agents/train.py" 2>/dev/null
  pkill -f "scripts/spectator.py" 2>/dev/null
  pkill -f "scripts/watchdog.sh" 2>/dev/null
  pkill -f "scripts/collect_bc_data.py" 2>/dev/null
  sleep 3
  ps aux | grep -E 'train\.py|watchdog\.sh|spectator\.py|collect_bc' | grep -v grep
  # expect empty

=== STEP 2: pull the Run-4 fixes ===

  cd ~/UrbanZero
  git fetch origin
  git checkout claude/setup-av-training-VetPV
  git pull
  git log --oneline -3

Expected top commit subject includes "Run-4" or "POTENTIAL_K"
or "NPC motion fix". If top commit is still dfbcf9e, STOP and
re-pull after 30 seconds.

Verify the two fixes landed:
  grep -n "POTENTIAL_K = 0.015" env/carla_env.py
  # expect one match

  grep -n "global_percentage_speed_difference" env/carla_env.py
  # expect one match inside _spawn_traffic

  grep -n "committed via tick" env/carla_env.py
  # expect one match (the new [TM] print line)

If ANY grep fails, STOP and tell the user the pull didn't take.

=== STEP 3: archive Run-3's artifacts ===

  ts=$(date +%s)
  if [ -d ~/urbanzero/logs/v2_rl ]; then
    mv ~/urbanzero/logs/v2_rl ~/urbanzero/logs/v2_rl.run3-${ts}
    mkdir -p ~/urbanzero/logs/v2_rl
  fi
  if [ -d ~/urbanzero/checkpoints/v2_rl ]; then
    mv ~/urbanzero/checkpoints/v2_rl ~/urbanzero/checkpoints/v2_rl.run3-${ts}
    mkdir -p ~/urbanzero/checkpoints/v2_rl
  fi
  if [ -f ~/urbanzero/beacon.json ]; then
    cp ~/urbanzero/beacon.json ~/urbanzero/beacon.run3-${ts}.json
  fi

=== STEP 4: verify both CARLA instances ===

  for p in 2000 3000; do
    if timeout 3 bash -c ">/dev/tcp/172.25.176.1/$p" 2>/dev/null; then
      echo "port $p: UP"
    else
      echo "port $p: DOWN"
    fi
  done

If DOWN, paste to user the same relaunch block as prior runs.
Both servers are needed — Run-4 uses 2000, BC collection uses 3000.

=== STEP 5: PANE A — launch Run-4 pure-RL (port 2000 only) ===

  URBANZERO_EXP=run4_k015 \
  URBANZERO_N_ENVS=1 \
  URBANZERO_BASE_PORT=2000 \
  URBANZERO_TIMESTEPS=10000000 \
  URBANZERO_AUTO_RESUME=0 \
  URBANZERO_SEED=311 \
    bash scripts/start_training.sh

Attach briefly to confirm:
  tmux a -t urbanzero
  # Wait for [TM] line + first rollout. Ctrl-b d to detach.

=== STEP 6: PANE B — launch BC data collection (port 3000) IN PARALLEL ===

  tmux new -d -s bc \
    'source ~/urbanzero_env/bin/activate && \
     cd ~/UrbanZero && \
     python3 scripts/collect_bc_data.py \
       --port 3000 \
       --n_frames 100000 \
       --seed 77 \
       --output ~/urbanzero/bc_data/bc_data_$(date +%s).npz \
     2>&1 | tee ~/urbanzero/logs/bc_collect_$(date +%Y%m%d_%H%M).log'

  tmux ls
  # expect: urbanzero (Run-4), bc (BC collection). Watchdog next.

=== STEP 7: launch watchdog ===

  tmux new -d -s wd 'bash scripts/watchdog.sh'
  tmux ls
  # expect: urbanzero, bc, wd

=== STEP 8: EARLY-CHECK T+5min — NPC motion diagnostic ===

Verify the NPC fix is working:

  tail -300 $(ls -t ~/urbanzero/logs/v2_rl/train_*.log | head -1) | \
    grep '\[TM\]' | head -6

Expected (per spawn, two lines per reset):
  [TM] port=8XXX sync_mode=True requested, world.sync=True hybrid_physics=True radius=70m
  [TM] spawned 30 NPCs, speed_diff=-30% global, committed via tick

If the SECOND line ("committed via tick") is MISSING, the new code
didn't land. STOP and check the git pull.

Ask the user:
--- paste to user ---
"Please look at the CARLA window on port 2000 for ~30 seconds. Are
the non-ego NPC vehicles visibly moving (driving down streets,
turning at intersections)? Yes/no. This is the NPC fix verification."
--- end paste ---

If user says NPCs still frozen: STOP Run-4, DO NOT kill BC
collection. Report to remote Claude. We pivot fully to BC.

=== STEP 9: BC collection progress check T+5min ===

  tmux capture-pane -t bc -p | tail -20

Expected: lines like "[BC-collect] N/100000 frames, FPS=...". If
the BC pane is crashed/stopped, investigate (likely CARLA server or
BehaviorAgent import issue). DO NOT kill Run-4 for BC issues — the
two are independent.

=== STEP 10: HARD GATES for Run-4 (report at each) ===

**T+10min** (~72k env-steps @ 120 FPS × 1 env):
  - policy_std ∈ [0.45, 0.70]  (clamp working, not pinned)
  - avg_speed > 1.5 m/s  (idle_cost working)
  - cumulative_reward_clip_hits == 0  (no reward overflow)
  - User viewport eyeball: NPCs moving
  If ANY fails → KILL Run-4, report to remote Claude.

**T+45min** (~300k steps):
  - rolling_RC ≥ 6%  (above Run-3's plateau)
  - COLLISION% < 50  (below Run-3's contaminated 50%)
  - OFF_ROUTE% < 35  (below Run-3's 41%)
  Report the full beacon snapshot.

**T+90min** (~600k steps) — DECISION GATE:
  - rolling_RC < 8%: KILL Run-4, promote BC Stage-3 PPO to primary
    (chain: BC pipeline already has data collected + trainer built;
    remote Claude will write Stage-3 launch prompt).
  - rolling_RC ≥ 8%: Run-4 continues. BC becomes insurance only.

**T+4h** (~1.7M steps):
  - rolling_RC ≥ 15%: on-track; continue to full 10M.
  - rolling_RC < 15% but ≥ 8%: borderline; let it run to 3M gate.

=== STEP 11: beacon snapshot command ===

At each gate:

  cat ~/urbanzero/beacon.json | python3 -c "
  import json,sys
  b=json.load(sys.stdin)
  tr=b.get('termination_reasons',{})
  tot=sum(tr.values()) or 1
  stuck_pct=100*tr.get('REALLY_STUCK',0)/tot
  coll_pct=100*tr.get('COLLISION',0)/tot
  offr_pct=100*tr.get('OFF_ROUTE',0)/tot
  print(f'ts={b[\"timesteps\"]} fps={b[\"fps\"]:.1f} '
        f'speed={b[\"rolling_avg_speed_ms\"]:.3f} '
        f'ep_len={b[\"rolling_ep_len\"]:.0f} '
        f'stuck%={stuck_pct:.1f} coll%={coll_pct:.1f} offr%={offr_pct:.1f} '
        f'RC={b[\"rolling_route_completion\"]:.3%} '
        f'best_RC={b.get(\"rolling_best_rc\", \"n/a\")} '
        f'std={b[\"policy_std\"]:.3f} '
        f'approx_kl={b.get(\"approx_kl\")} '
        f'ev={b.get(\"explained_variance\")} '
        f'clip_hits={b.get(\"cumulative_reward_clip_hits\", 0)}')
  "

=== STEP 12: when BC collection finishes ===

When `tmux capture-pane -t bc -p | tail -5` shows "Done. N frames"
(typically ~1.5h):
  - Confirm .npz is saved: `ls -lh ~/urbanzero/bc_data/`
  - Immediately launch BC training in the same pane:

    tmux send-keys -t bc 'python3 agents/train_bc.py \
      --data ~/urbanzero/bc_data/bc_data_*.npz \
      --output ~/urbanzero/checkpoints/bc_pretrain.zip \
      --epochs 20 2>&1 | tee ~/urbanzero/logs/bc_train_$(date +%Y%m%d_%H%M).log' Enter

  Training time: ~2h on RTX 4080S. After it finishes, the
  `bc_pretrain.zip` + sibling `_vecnormalize.pkl` are ready for
  PPO warmstart.

=== STEP 13: if Run-4 fails T+90min gate, trigger BC PPO finetune ===

If remote Claude confirms the kill decision:

  tmux kill-session -t urbanzero
  # wait for BC training to finish if it's still running

  URBANZERO_EXP=bc_pft \
  URBANZERO_N_ENVS=2 \
  URBANZERO_BASE_PORT=2000 \
  URBANZERO_TIMESTEPS=5000000 \
  URBANZERO_AUTO_RESUME=0 \
  URBANZERO_SEED=911 \
  URBANZERO_BC_WEIGHTS=~/urbanzero/checkpoints/bc_pretrain.zip \
    bash scripts/start_training.sh

  (The `URBANZERO_BC_WEIGHTS` env var triggers the BC-warmstart path
  in agents/train.py added in commit dfbcf9e.)

=== RED FLAGS ===

  [RED-1] NPCs still frozen at T+5min viewport check → pivot to BC
  [RED-2] [TM] "committed via tick" line absent → pull didn't take
  [RED-3] BC collection pane dead → investigate but don't kill Run-4
  [RED-4] policy_std outside [0.45, 0.70] at T+10min → clamp broken
  [RED-5] cumulative_reward_clip_hits > 0 → shaping magnitude wrong
  [RED-6] Run-4 RC < 8% at T+90min → BC takes over

=== CONFIG SUMMARY ===

- Tip: claude/setup-av-training-VetPV HEAD (post Run-4 commits)
- Run-4: N_ENVS=1, port 2000, seed 311, TIMESTEPS=10M
- BC collect: port 3000, seed 77, n_frames=100k
- POTENTIAL_K = 0.015 (was 0.03)
- log_std clamp ≤ 0.7 (unchanged)
- idle_cost = -0.15 * max(0, 1 - speed/1.0) (unchanged)
- Carrot = 0.005 * min(speed, TARGET)/TARGET (un-annealed, unchanged)
- progress_reward = 0.05 * progress_delta (unchanged)
- Terminals ±50 (unchanged)
- TM: sync + hybrid_physics + speed_diff -30% + commit tick (NEW)

DO NOT add new reward terms. On any red flag, report with full
context; remote Claude decides.
```

---

## Notes for the user (context, not to paste)

**Why the subagent made this call (summary):**

The subagent (given zero context from me) diagnosed three compounding
issues, not one:

1. **Frozen NPCs contaminated the reward signal** — ~50% of Run-3's
   COLLISION terminals were against stationary NPCs, meaning those
   terminals were essentially uninformative to the learning signal.
   Fix the NPCs → ~30% of those −50 penalties become properly
   attributable to learnable behaviors.

2. **Shaping magnitude was too strong** — at POTENTIAL_K=0.03, max
   shaping/step equaled max progress_reward/step. The shaping was
   REDUCING distance to lookahead regardless of approach angle,
   which subtly rewarded perpendicular approach. Halving K restores
   progress as the dominant directional signal.

3. **Fallback paths should run in parallel, not serial** — waiting
   for Run-3 to "fail a gate" then pivoting to BC burned 5h of
   deadline each time. Running both simultaneously means whichever
   works, ships.

**The one thing the subagent REFUSED to add:** a cosine-heading
shaping term, even in potential-based form. Its reasoning: every one
of the 7M-run's seven documented pathologies started with "we added
one more shaping term to fix behavior X." If POTENTIAL_K=0.015 is
still too weak to produce >8% RC at T+90min, the next move is
`progress_reward 0.05→0.10` (same directional signal, stronger
magnitude, zero new terms) + BC path taking over.

**p(ship): 0.62** = 1 - (1-0.30)(1-0.55). Primary risk: if the NPC
fix doesn't actually unstick NPCs (e.g., CARLA #9172 TM race persists
beyond the commit tick), Run-4's COLLISION rate stays 50% and RC
stays ~5%. In that case BC path runs regardless, so the deadline is
still hedged.
