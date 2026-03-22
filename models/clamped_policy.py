from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import DiagGaussianDistribution
import torch

class ClampedStdPolicy(ActorCriticPolicy):
    """PPO policy with log_std clamped to prevent explosion or collapse."""
    
    def _clamp(self):
        self.log_std.data.clamp_(-1.0, 0.5)
    
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
