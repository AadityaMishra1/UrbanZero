# PC-Side — NPC Sanity v2 (three new diagnostic tests)

Per Issue #12: NPCs frozen in all modes. The original sanity check
didn't differentiate between three possible causes flagged by you
and the reviewer. This v2 script does.

Three leads to test:
1. **Map issue** — CARLA 0.9.15 default is Town10HD_Opt, not Town01.
   TM pathing might not work on Town10HD_Opt.
2. **No hero vehicle** — `hybrid_physics_mode=True` puts NPCs outside
   the hero radius into dormant state. Sanity check had no ego, so
   ALL NPCs were dormant. (Training has an ego so this wouldn't be
   the training cause — but we need to rule it out here.)
3. **Hybrid physics itself** — may be the cause even when a hero
   exists. Default flipped to OFF in the new script.

## Copy-paste to PC-side

```text
Sanity check v2 pushed. Three new flags to isolate the cause.

  cd ~/UrbanZero && git pull

=== TEST 1: what map is running? ===

  source ~/urbanzero_env/bin/activate
  python3 scripts/sanity_check_npcs.py --port 2000 --ticks 10

Look for the line:
  [sanity] connected. map='<NAME>' world.sync (initial) = ...

Report the map name. If it's Town10HD_Opt or similar, that might be
the root cause — TM path generation may not work on that map.

=== TEST 2: hybrid_physics OFF (new default) ===

The v2 script defaults hybrid_physics to OFF. Just re-run the solo
test:

  python3 scripts/sanity_check_npcs.py --port 2000 --ticks 60 --verbose

If NPCs now MOVE → hybrid_physics was the bug. Training env has
`tm.set_hybrid_physics_mode(True)`; we remove that line.

If NPCs STILL frozen → hybrid_physics wasn't the cause. Move to test 3.

=== TEST 3: with ego vehicle ===

  python3 scripts/sanity_check_npcs.py --port 2000 --ticks 60 --spawn_ego --hybrid_physics --verbose

The ego acts as the "hero" for hybrid_physics radius. If NPCs move
now → we need an ego present for hybrid_physics to work. Training
env spawns ego, so this shouldn't matter in training, but it means
the standalone check needs the ego for a valid test.

=== TEST 4: force-load Town01 ===

  python3 scripts/sanity_check_npcs.py --port 2000 --ticks 60 --load_map Town01 --verbose

If NPCs move on Town01 specifically → the default map (whatever it
is) is the root cause. Training env needs to force-load Town01.

=== REPORT FORMAT ===

  Test 1 (map check):           map='<NAME>'
  Test 2 (hybrid OFF):           PASS/FAIL  avg=X.XX m/s
  Test 3 (with ego):             PASS/FAIL  avg=X.XX m/s
  Test 4 (Town01 forced):        PASS/FAIL  avg=X.XX m/s

Based on WHICH combination passes, the root cause is identified:
- Test 2 passes only → disable hybrid_physics in env
- Test 3 passes (test 2 fails) → keep hybrid_physics but need hero
  (training already has one, so unrelated to training failure)
- Test 4 passes (others fail) → force-load Town01 in env
- All fail → CARLA install issue / version bug, escalate to reviewer
```

## Expected outcomes by hypothesis

| Result | Root cause | Training-env fix |
|---|---|---|
| Test 2 PASS, Test 1 shows Town01 | hybrid_physics always broken | Remove `tm.set_hybrid_physics_mode(True)` |
| Test 3 PASS only | hybrid_physics needs hero | Non-issue in training (has ego) |
| Test 4 PASS only | wrong default map | Add `client.load_world("Town01")` to env init |
| Test 2 AND Test 4 fail, Test 3 passes | multi-factor | Both map + hybrid_physics changes |
| All fail | CARLA install bug | Reinstall CARLA, ask reviewer again |
