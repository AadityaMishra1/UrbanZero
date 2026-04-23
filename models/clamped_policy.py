"""PPO policy with a soft upper bound on log_std only.

Prior version clamped log_std to [-2.0, -0.7], forcing std into [0.14, 0.50].
Per the 7M-run FINAL REPORT, std drifted to the lower bound (0.23) and
entropy_loss went positive (+0.115 = near-deterministic). The clamp was
actively preventing ent_coef from maintaining exploration.

New design (backed by Andrychowicz et al. 2021 "What Matters in On-Policy
Reinforcement Learning", ICLR):
  - no lower bound — let ent_coef (with a floor value, not annealed to
    zero) maintain exploration via its entropy-bonus gradient.
  - soft upper bound log_std <= log(0.7) ≈ -0.357 (std <= 0.7). The
    first value here was log(1.0) = 0.0; the 2026-04-22 run with the
    idle_cost fix saturated log_std at that cap for 1500+ episodes
    (std = 0.999 pinned), the policy became pure noise, and rolling
    RC flatlined at 5-6% across 1500 episodes with zero trend. That is
    an exploration pathology, not a gradient problem: at std=1.0 every
    episode is a different random walk and the critic cannot assign
    credit for any action. Andrychowicz 2021 §4.5 identifies std in
    [0.3, 0.7] as the viable band for continuous control from scratch;
    capping at 0.7 keeps exploration pressure high without letting the
    entropy bonus drive the policy past the usable band. Lower floor
    remains unbounded and controlled by ent_coef's gradient only.
"""

from stable_baselines3.common.policies import ActorCriticPolicy
import math


class ClampedStdPolicy(ActorCriticPolicy):
    """PPO policy with a soft upper bound on log_std (std <= 0.7).

    Works with any continuous DiagGaussian action space; the clamp is
    applied in-place before every forward/evaluate/distribution call so
    the optimizer cannot push log_std past the cap between updates.
    """

    # log(0.7) ≈ -0.35667 → std ≤ 0.7 (Andrychowicz 2021 §4.5 viable band)
    LOG_STD_MAX = math.log(0.7)

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
