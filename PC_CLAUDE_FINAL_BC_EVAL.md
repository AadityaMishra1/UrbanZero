# PC-Side Claude — Final BC Eval

Run deterministic BC eval on `bc_pretrain.zip` to get the result number for the paper.

## Copy-paste everything below into PC-side Claude Code

```text
Run final deterministic BC evaluation. This produces the result number for the paper.

=== STEP 1: kill anything still running ===

  tmux kill-session -t urbanzero 2>/dev/null
  tmux kill-session -t wd 2>/dev/null
  tmux kill-session -t eval 2>/dev/null
  pkill -f "agents/train.py" 2>/dev/null
  pkill -f "scripts/eval_bc.py" 2>/dev/null
  pkill -f "scripts/watchdog.sh" 2>/dev/null
  sleep 3

=== STEP 2: pull latest tip (has hybrid_physics fix from issue #13) ===

  cd ~/UrbanZero
  git fetch origin
  git checkout claude/setup-av-training-VetPV
  git pull
  git log --oneline -3

Verify the env now has hybrid_physics OFF:
  grep -n "hybrid_physics=OFF" env/carla_env.py
  # expect 1 match in _spawn_traffic [TM] print

=== STEP 3: verify CARLA up on port 2000 ===

  timeout 3 bash -c ">/dev/tcp/172.25.176.1/2000" \
    && echo "port 2000: UP" || echo "port 2000: DOWN"

If DOWN, paste relaunch block to user.

=== STEP 4: run the eval — 20 deterministic episodes ===

  mkdir -p ~/urbanzero/eval
  source ~/urbanzero_env/bin/activate
  cd ~/UrbanZero
  python3 scripts/eval_bc.py \
    --model ~/urbanzero/checkpoints/bc_pretrain.zip \
    --episodes 20 \
    --port 2000 \
    --seed 1001 \
    --output ~/urbanzero/eval/bc_final_$(date +%Y%m%d_%H%M).json \
    2>&1 | tee ~/urbanzero/logs/bc_eval_final_$(date +%Y%m%d_%H%M).log

Wall clock: ~20-40 min. Per-episode lines stream to terminal:
  [eval] ep 01/20: reason=COLLISION    RC= 12.3% steps= 280 speed=4.2 m/s reward=-31.5
  ...

At the end, the script prints a summary block:
  ============================================================
  EVALUATION SUMMARY — 20 episodes, port 2000
  ============================================================
    RC mean: X.XX%
    RC median: X.XX%
    RC max: XX.XX%
    %ROUTE_COMPLETE: X.X%
    %COLLISION: XX.X%
    %OFF_ROUTE: XX.X%
    %REALLY_STUCK: X.X%
    avg speed: X.XX m/s
    wall clock: XX.X min

=== STEP 5: report to user ===

Paste the EVALUATION SUMMARY block plus the JSON file path. Also confirm:
  ls -lh ~/urbanzero/eval/bc_final_*.json

Then the user gives those numbers to remote Claude to start the paper.

=== If eval crashes ===

If the script errors out (likely paths: VecNormalize file mismatch, CARLA disconnect, deterministic predict shape error), paste the full Python traceback. Don't try to fix it yourself — report it.
```
