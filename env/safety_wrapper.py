"""Boundary safety wrapper around CarlaEnv.

Catches non-finite (NaN/Inf) values in obs or reward at the env boundary.
Replaces them with safe values and forces a terminal so:
  - VecNormalize's running mean/var never absorbs an Inf (would poison
    every subsequent reward forever).
  - PPO's policy update never sees a NaN observation.
  - The bug becomes greppable in logs ([NaN-GUARD]) instead of silently
    crashing training mid-rollout, which would lose hours of weights.

This is defense-in-depth: the env-internal clamps in carla_env.py are the
primary defense; this wrapper catches anything they miss.
"""

import numpy as np
from gymnasium import Wrapper


class NaNGuardWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._guard_hits = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        bad = False
        if not np.isfinite(reward):
            print(f"[NaN-GUARD] reward={reward!r} at step={info.get('step')}; "
                  f"forcing terminal, reward->0.0")
            reward = 0.0
            bad = True
        if isinstance(obs, dict):
            for k, v in obs.items():
                if isinstance(v, np.ndarray) and not np.all(np.isfinite(v)):
                    print(f"[NaN-GUARD] obs[{k!r}] non-finite at step={info.get('step')}")
                    obs[k] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                    bad = True
        elif isinstance(obs, np.ndarray) and not np.all(np.isfinite(obs)):
            print(f"[NaN-GUARD] obs non-finite at step={info.get('step')}")
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            bad = True
        if bad:
            terminated = True
            self._guard_hits += 1
            info = dict(info)
            info["nan_terminated"] = True
            info["guard_hits_total"] = self._guard_hits
            # Stamp the termination reason so BeaconCallback's termination_reasons
            # tally doesn't mis-classify as "UNKNOWN". Env-side _compute_reward
            # leaves termination_reason=None on non-terminal steps; if the guard
            # fires on such a step and we don't override, the beacon loses the
            # signal that a NaN-guard actually triggered the terminal.
            info["termination_reason"] = "NAN_GUARD"
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if isinstance(obs, dict):
            for k, v in obs.items():
                if isinstance(v, np.ndarray) and not np.all(np.isfinite(v)):
                    print(f"[NaN-GUARD] reset obs[{k!r}] non-finite; sanitizing")
                    obs[k] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        return obs, info
