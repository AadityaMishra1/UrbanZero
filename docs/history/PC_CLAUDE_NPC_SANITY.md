# PC-Side Claude — NPC Sanity Check (run BEFORE v5)

Per external reviewer's final question. Verify NPCs move in a
minimal standalone script before burning v5 compute.

## Copy-paste to PC-side Claude

```text
Reviewer requested a standalone NPC sanity check before v5 launch.
Run the new script on BOTH ports and report results. This is
non-negotiable — if NPCs don't move here, they won't move in v5.

=== STEP 1: kill anything running ===

  tmux kill-session -t urbanzero 2>/dev/null
  tmux kill-session -t wd 2>/dev/null
  tmux kill-session -t bc0 2>/dev/null
  tmux kill-session -t bc1 2>/dev/null
  pkill -f "agents/train.py" 2>/dev/null
  pkill -f "scripts/watchdog.sh" 2>/dev/null
  pkill -f "scripts/collect_bc_data.py" 2>/dev/null
  sleep 3

=== STEP 2: pull the sanity check ===

  cd ~/UrbanZero
  git fetch origin
  git pull
  ls -la scripts/sanity_check_npcs.py
  # ~11 KB, should be executable

=== STEP 3: verify CARLA up ===

  for p in 2000 3000; do
    timeout 3 bash -c ">/dev/tcp/172.25.176.1/$p" 2>/dev/null \
      && echo "port $p: UP" || echo "port $p: DOWN"
  done

=== STEP 4: TEST A — port 2000 alone ===

  source ~/urbanzero_env/bin/activate
  cd ~/UrbanZero
  python3 scripts/sanity_check_npcs.py --port 2000 --n_npcs 10 --ticks 60 --verbose

Expected (if NPCs work):
  [sanity] PASS: NPCs are moving (avg X.XX m/s > 0.5 threshold)
  exit code 0

Expected (if frozen — matches training-run report):
  [sanity] FAIL: NPCs are FROZEN (avg 0.00 m/s)
  exit code 1
  + detailed diagnostic printout

Report the full output. Save exit code:
  echo "Port 2000 exit code: $?"

=== STEP 5: TEST B — port 3000 alone ===

  python3 scripts/sanity_check_npcs.py --port 3000 --n_npcs 10 --ticks 60 --verbose
  echo "Port 3000 exit code: $?"

=== STEP 6: TEST C — both ports SIMULTANEOUSLY ===

This is the key test for the reviewer's TM port collision theory.
Two clients ticking two worlds at the same time is what DummyVecEnv
does during training.

  python3 scripts/sanity_check_npcs.py --port 2000 --n_npcs 10 --ticks 60 \
    > /tmp/sanity_2000.log 2>&1 &
  PID_2000=$!
  python3 scripts/sanity_check_npcs.py --port 3000 --n_npcs 10 --ticks 60 \
    > /tmp/sanity_3000.log 2>&1 &
  PID_3000=$!
  wait $PID_2000
  E_2000=$?
  wait $PID_3000
  E_3000=$?
  echo "=== Port 2000 (concurrent): exit $E_2000 ==="
  cat /tmp/sanity_2000.log
  echo
  echo "=== Port 3000 (concurrent): exit $E_3000 ==="
  cat /tmp/sanity_3000.log

=== STEP 7: interpret and report ===

Report to the user in this EXACT format:

  Test A (port 2000 alone):    [PASS/FAIL] avg_speed=X.XX m/s
  Test B (port 3000 alone):    [PASS/FAIL] avg_speed=X.XX m/s
  Test C concurrent, port 2000: [PASS/FAIL] avg_speed=X.XX m/s
  Test C concurrent, port 3000: [PASS/FAIL] avg_speed=X.XX m/s

Plus paste the "Diagnostic for reviewer" block from any FAIL output.

=== DECISION MATRIX ===

ALL 4 PASS (NPCs move in solo + concurrent):
  → TM plumbing is fine. The in-training diagnostic must have been
    misreading or CARLA was in a different state. Proceed to v5.

SOLO PASS, CONCURRENT FAIL:
  → REVIEWER'S HYPOTHESIS CONFIRMED. TM port collision under
    concurrent clients. Remote Claude will push a fix (likely an
    explicit tm_port argument scheme that guarantees non-collision,
    or process-level isolation).

ALL 4 FAIL (NPCs frozen even solo):
  → TM plumbing itself is broken, not a concurrency issue.
    Possible causes: wrong map, CARLA version mismatch,
    hybrid_physics side-effect. Remote Claude investigates based
    on the diagnostic printout.

SOLO FAIL but A passes and B fails (or vice versa):
  → Port-specific issue. One of the CARLA servers is broken.
    User should restart the failing CARLA instance.
```

---

## Notes for user (not to paste)

**Why this script exists:** inside the training loop there are
~10 things that could be wrong at once (reward, BC loading, wrapper
stack, vecnorm stats, env vars). The `[NPC-diagnostic]` print I
added to v5 would have eventually told us NPCs are frozen, but
by then we'd have already burned compute launching v5.

This standalone script tests ONLY the NPC plumbing in ~60 seconds,
then cleanly restores async mode. It's surgical.

**What the results tell us:**

- **If all 4 tests PASS:** the reviewer's "NPCs primary" hypothesis
  may actually have been wrong (or the training loop fixes it by
  now). Launch v5.
- **If solo passes but concurrent fails:** reviewer's port-collision
  theory is exactly right. We need to change the TM port allocation
  so two concurrent clients don't share or fight over TM resources.
- **If all 4 fail:** something fundamental is broken (CARLA version,
  map, or a subtle init bug). This is where the diagnostic output
  from the script becomes critical evidence for further debugging.

**What happens next depends on what PC-side reports.** I won't push
anything else until we see the sanity-check results.
