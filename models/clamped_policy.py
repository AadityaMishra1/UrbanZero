"""PPO policy with a soft upper bound on log_std only.

Prior version clamped log_std to [-2.0, -0.7], forcing std into [0.14, 0.50].
Per the 7M-run FINAL REPORT, std drifted to the lower bound (0.23) and
entropy_loss went positive (+0.115 = near-deterministic). The clamp was
actively preventing ent_coef from maintaining exploration.

New design (backed by Andrychowicz et al. 2021 "What Matters in On-Policy
Reinforcement Learning", ICLR):
  - no lower bound — let ent_coef (with a floor value, not annealed to
    zero) maintain exploration via its entropy-bonus gradient.
  - soft upper bound log_std <= 0.0 (std <= 1.0). Prevents runaway
    log_std -> +inf (policy becomes uniform noise). This is a safety
    cap, not a target; healthy exploration sits well below it.
"""

from stable_baselines3.common.policies import ActorCriticPolicy


class ClampedStdPolicy(ActorCriticPolicy):
    """PPO policy with a soft upper bound on log_std (std <= 1.0).

    Works with any continuous DiagGaussian action space; the clamp is
    applied in-place before every forward/evaluate/distribution call so
    the optimizer cannot push log_std past the cap between updates.
    """

    LOG_STD_MAX = 0.0  # std <= exp(0.0) = 1.0

    def _clamp(self):
        # Only cap from above. Lower bound deliberately left unbounded
        # so ent_coef's entropy gradient is what resists collapse.
        self.log_std.data.clamp_(max=self.LOG_STD_MAX)

    def forward(self, obs, deterministic=False):
        self._clamp()
        return super().forward(obs, deterministic)

    def evaluate_actions(self, obs, actions):
        self._clamp()
        return super().evaluate_actions(obs, actions)

    def get_distribution(self, obs):
        self._clamp()
        return super().get_distribution(obs)

    def predict_values(self, obs):
        return super().predict_values(obs)
