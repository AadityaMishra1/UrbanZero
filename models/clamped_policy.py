from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import DiagGaussianDistribution
import torch

class ClampedStdPolicy(ActorCriticPolicy):
    """PPO policy with log_std clamped to prevent explosion or collapse."""
    
    def _clamp(self):
        # log_std range: [-1.5, -0.3] → std range: [0.22, 0.74]
        # Old range [-1.0, 0.5] allowed std up to 1.65, which is noise
        # as large as the full action range — causes zig-zagging and
        # inconsistent behavior across episodes.
        self.log_std.data.clamp_(-1.5, -0.3)
    
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
