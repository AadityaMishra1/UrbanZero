"""
CARLA Leaderboard-style evaluation for UrbanZero.

Metrics tracked:
- Route Completion (RC): fraction of route completed before failure
- Infraction Score (IS): multiplicative penalty for violations
- Driving Score (DS): RC × IS (gold standard metric)
- Collisions per km
- Average speed (m/s)
- Action smoothness (mean |steer_t - steer_{t-1}|)
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class DrivingMetricsCallback(BaseCallback):
    """
    Custom SB3 callback that logs CARLA driving metrics to TensorBoard
    every `eval_freq` timesteps.

    Reads from the info dict returned by CarlaEnv.step():
    - route_completion: float [0, 1]
    - speed: float (m/s)
    - collisions: int
    """

    def __init__(self, eval_freq=10000, verbose=0):
        super().__init__(verbose)
        self.eval_freq = eval_freq

        # Accumulate episode-level metrics
        self.episode_route_completions = []
        self.episode_speeds = []
        self.episode_collisions = []
        self.episode_lengths = []
        self.episode_rewards_sum = []

        # Track per-step data for current episode (per env)
        self._current_speeds = {}
        self._current_steer_deltas = {}
        self._current_rewards = {}
        self._prev_steer = {}

    def _on_step(self) -> bool:
        # Gather info from all envs
        infos = self.locals.get("infos", [])
        for i, info in enumerate(infos):
            if "route_completion" in info:
                speed = info.get("speed", 0.0)
                if i not in self._current_speeds:
                    self._current_speeds[i] = []
                self._current_speeds[i].append(speed)

            # Track episode completion via SB3's dones
            dones = self.locals.get("dones", [])
            if i < len(dones) and dones[i]:
                rc = info.get("route_completion", 0.0)
                collisions = info.get("collisions", 0)
                ep_len = info.get("step", 0)

                self.episode_route_completions.append(rc)
                self.episode_collisions.append(collisions)
                self.episode_lengths.append(ep_len)

                if i in self._current_speeds and self._current_speeds[i]:
                    self.episode_speeds.append(np.mean(self._current_speeds[i]))
                self._current_speeds[i] = []

        # Log metrics at eval_freq intervals
        if self.n_calls % self.eval_freq == 0 and self.episode_route_completions:
            n_episodes = len(self.episode_route_completions)

            avg_rc = np.mean(self.episode_route_completions)
            avg_collisions = np.mean(self.episode_collisions)
            avg_speed = np.mean(self.episode_speeds) if self.episode_speeds else 0.0
            avg_ep_len = np.mean(self.episode_lengths)

            # Driving Score approximation: RC * (1 - collision_rate)
            collision_rate = np.mean([1 if c > 0 else 0 for c in self.episode_collisions])
            approx_ds = avg_rc * (1.0 - collision_rate)

            self.logger.record("driving/route_completion", avg_rc)
            self.logger.record("driving/driving_score", approx_ds)
            self.logger.record("driving/avg_speed_ms", avg_speed)
            self.logger.record("driving/avg_collisions", avg_collisions)
            self.logger.record("driving/collision_rate", collision_rate)
            self.logger.record("driving/avg_episode_length", avg_ep_len)
            self.logger.record("driving/num_episodes", n_episodes)

            if self.verbose > 0:
                reliable = "" if n_episodes >= 20 else f" (LOW sample — {n_episodes} < 20)"
                print(f"\n[Eval @ {self.num_timesteps} steps] "
                      f"RC={avg_rc:.2%} DS={approx_ds:.2%} "
                      f"Speed={avg_speed:.1f}m/s Collisions={avg_collisions:.1f} "
                      f"EpLen={avg_ep_len:.0f} ({n_episodes} episodes){reliable}")

            # Reset accumulators
            self.episode_route_completions = []
            self.episode_collisions = []
            self.episode_speeds = []
            self.episode_lengths = []

        return True
