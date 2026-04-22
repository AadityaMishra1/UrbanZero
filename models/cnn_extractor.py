"""
Custom CNN feature extractor for autonomous driving.

Replaces SB3's NatureCNN (3-layer, Atari-era) with a deeper architecture
using LayerNorm, inspired by CaRL (NVIDIA AVG, CoRL 2025).

Key improvements over NatureCNN:
- 5 conv layers instead of 3 → larger receptive field for driving scenes
- LayerNorm after each conv → stable training with larger batch sizes
- Separate MLP pathway for state vector → proper multi-modal fusion
- Configurable feature dimension
"""

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class DrivingCNN(BaseFeaturesExtractor):
    """
    Multi-input feature extractor: CNN for semantic segmentation images +
    MLP for vehicle state vector, fused into a shared feature representation.

    Architecture (image pathway):
        Conv2d(C_in, 32, 5, stride=2, pad=2) → LayerNorm → ReLU
        Conv2d(32, 64, 3, stride=2, pad=1)   → LayerNorm → ReLU
        Conv2d(64, 128, 3, stride=2, pad=1)  → LayerNorm → ReLU
        Conv2d(128, 128, 3, stride=2, pad=1) → LayerNorm → ReLU
        Conv2d(128, 256, 3, stride=2, pad=1) → LayerNorm → ReLU
        Flatten → Linear(256*4*4, 256)

    Architecture (state pathway):
        Linear(state_dim, 64) → ReLU → Linear(64, 64) → ReLU

    Fusion:
        Concatenate(image_features, state_features) → Linear(320, features_dim) → ReLU
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        # We must call super with the final features_dim
        super().__init__(observation_space, features_dim)

        image_space = observation_space["image"]
        state_space = observation_space["state"]

        n_channels = image_space.shape[0]  # should be 1 (or 4 with frame stacking)
        h, w = image_space.shape[1], image_space.shape[2]

        # state_dim is robust to VecFrameStack's shape convention: with
        # channels_order={"state": "last"} on a (10,) space, SB3 produces
        # (40,) — but older SB3 versions may produce (10, 4). np.prod handles
        # both and matches whatever the runtime batch actually carries after
        # Flatten in forward().
        import numpy as _np
        state_dim = int(_np.prod(state_space.shape))

        # Image CNN pathway (5 conv layers with LayerNorm)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.LayerNorm([32, h // 2, w // 2]),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([64, h // 4, w // 4]),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([128, h // 8, w // 8]),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([128, h // 16, w // 16]),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([256, h // 32, w // 32]),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute CNN output size
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, h, w)
            cnn_out_size = self.cnn(dummy).shape[1]

        self.cnn_fc = nn.Sequential(
            nn.Linear(cnn_out_size, 256),
            nn.ReLU(),
        )

        # State MLP pathway
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(256 + 64, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        image = observations["image"].float()
        state = observations["state"].float()

        # Flatten state defensively: shape after VecFrameStack may be either
        # (B, state_dim) or (B, base_state_dim, n_stack) depending on the
        # installed SB3 version's "last"-stacking convention. np.prod was used
        # to size state_mlp so flatten-to-(B, state_dim) matches either way.
        if state.dim() > 2:
            state = state.flatten(1)

        # CNN pathway
        cnn_features = self.cnn(image)
        cnn_features = self.cnn_fc(cnn_features)

        # State pathway
        state_features = self.state_mlp(state)

        # Fuse
        combined = torch.cat([cnn_features, state_features], dim=1)
        return self.fusion(combined)
