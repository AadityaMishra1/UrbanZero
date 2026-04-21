"""Progress beacon callback.

Writes ~/urbanzero/beacon.json every N wall-clock seconds with current
training health. Used by:
  - scripts/watchdog.sh  to detect a hung trainer (beacon mtime > N min)
  - the user           to glance at health without attaching to tmux

Atomic write (tmp + os.replace) so the watchdog never reads a half-written
file. Mirrors the timing pattern of WallClockCheckpointCallback.
"""

import json
import os
import socket
import time
from collections import deque

from stable_baselines3.common.callbacks import BaseCallback


class BeaconCallback(BaseCallback):
    def __init__(self, beacon_path, experiment="shaped", carla_port=2000,
                 write_seconds=30, rolling_window=50, verbose=0):
        super().__init__(verbose)
        self.beacon_path = beacon_path
        self.experiment = experiment
        self.carla_port = carla_port
        self.write_seconds = write_seconds
        self.start_time = time.time()
        self._last_write = 0.0
        self._ep_returns = deque(maxlen=rolling_window)
        self._ep_lens = deque(maxlen=rolling_window)
        self._ep_rcs = deque(maxlen=rolling_window)
        self._ep_collisions = deque(maxlen=rolling_window)
        self._ep_speeds = deque(maxlen=rolling_window)
        self._ep_count = 0
        self._cur_returns = None
        self._cur_lens = None
        self._cur_speeds = None
        self.status = "starting"
        self.last_checkpoint = None
        self.last_checkpoint_time = None
        os.makedirs(os.path.dirname(beacon_path) or ".", exist_ok=True)

    def _on_training_start(self):
        n = self.training_env.num_envs
        self._cur_returns = [0.0] * n
        self._cur_lens = [0] * n
        self._cur_speeds = [0.0] * n
        self.status = "training"
        self._write()

    def _on_step(self):
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        infos = self.locals.get("infos") or []
        if rewards is not None and dones is not None and self._cur_returns is not None:
            for i in range(len(rewards)):
                self._cur_returns[i] += float(rewards[i])
                self._cur_lens[i] += 1
                if i < len(infos):
                    s = infos[i].get("speed", 0.0)
                    if s:
                        self._cur_speeds[i] = (self._cur_speeds[i] * 0.9
                                               + float(s) * 0.1)
                if dones[i]:
                    self._ep_returns.append(self._cur_returns[i])
                    self._ep_lens.append(self._cur_lens[i])
                    self._ep_speeds.append(self._cur_speeds[i])
                    if i < len(infos):
                        self._ep_rcs.append(float(infos[i].get("route_completion", 0.0)))
                        self._ep_collisions.append(int(infos[i].get("collisions", 0) > 0))
                    self._ep_count += 1
                    self._cur_returns[i] = 0.0
                    self._cur_lens[i] = 0
                    self._cur_speeds[i] = 0.0
        now = time.time()
        if now - self._last_write >= self.write_seconds:
            self._write()
            self._last_write = now
        return True

    def _on_training_end(self):
        self.status = "finished"
        self._write()

    def _scan_latest_checkpoint(self):
        ckpt_dir = os.path.expanduser(f"~/urbanzero/checkpoints/{self.experiment}")
        if not os.path.isdir(ckpt_dir):
            return None, None
        candidates = []
        for fn in os.listdir(ckpt_dir):
            if fn.endswith(".zip"):
                p = os.path.join(ckpt_dir, fn)
                try:
                    candidates.append((os.path.getmtime(p), p))
                except OSError:
                    pass
        if not candidates:
            return None, None
        candidates.sort(reverse=True)
        return candidates[0][1], candidates[0][0]

    def _write(self):
        ckpt, ckpt_t = self._scan_latest_checkpoint()
        if ckpt:
            self.last_checkpoint = ckpt
            self.last_checkpoint_time = ckpt_t
        n = len(self._ep_returns)
        avg = lambda d: (sum(d) / len(d)) if d else 0.0
        elapsed = max(1.0, time.time() - self.start_time)
        beacon = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "ts_unix": int(time.time()),
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "experiment": self.experiment,
            "status": self.status,
            "timesteps": int(self.num_timesteps),
            "fps": round(self.num_timesteps / elapsed, 2),
            "uptime_sec": int(elapsed),
            "rolling_ep_count": n,
            "rolling_ep_return": round(avg(self._ep_returns), 3),
            "rolling_ep_len": round(avg(self._ep_lens), 1),
            "rolling_route_completion": round(avg(self._ep_rcs), 4),
            "rolling_collision_rate": round(avg(self._ep_collisions), 4),
            "rolling_avg_speed_ms": round(avg(self._ep_speeds), 3),
            "total_episodes": self._ep_count,
            "carla_port": self.carla_port,
            "last_checkpoint": self.last_checkpoint,
            "last_checkpoint_unix": int(self.last_checkpoint_time)
                                    if self.last_checkpoint_time else None,
        }
        tmp = self.beacon_path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(beacon, f, indent=2)
            os.replace(tmp, self.beacon_path)
        except OSError as e:
            print(f"[beacon] write failed: {e}")
