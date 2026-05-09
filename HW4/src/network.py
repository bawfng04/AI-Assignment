# ==============================================================================
# HW4 — Dueling DQN Network Architecture
# ==============================================================================
"""
Implements the Dueling Network Architecture for Deep Reinforcement Learning.

Reference:
    Wang, Z., et al. (2016). "Dueling Network Architectures for Deep
    Reinforcement Learning." ICML 2016.
    
The key insight is decomposing Q(s,a) into:
    Q(s, a) = V(s) + (A(s, a) - mean(A(s, ·)))
    
This allows the network to learn which states are valuable without
having to learn the effect of each action at each state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network with convolutional feature extraction.
    
    Architecture follows the DeepMind Nature paper's CNN backbone,
    then splits into two streams:
        - Value stream V(s): scalar state-value estimate
        - Advantage stream A(s,a): per-action advantage estimates
    
    The streams are combined using the mean-subtraction formula to
    ensure identifiability:
        Q(s, a) = V(s) + (A(s, a) - mean_a'(A(s, a')))
    
    Args:
        input_shape: Shape of input observations (C, H, W).
                     Typically (4, 84, 84) for stacked Atari frames.
        n_actions:   Number of discrete actions in the environment.
    """

    def __init__(self, input_shape: Tuple[int, ...], n_actions: int) -> None:
        super().__init__()
        
        self.input_shape = input_shape
        self.n_actions = n_actions
        
        # === Convolutional Feature Extractor (shared backbone) ===
        # Follows the architecture from Mnih et al. (2015) Nature paper
        self.features = nn.Sequential(
            # Conv1: 32 filters, 8x8 kernel, stride 4
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            # Conv2: 64 filters, 4x4 kernel, stride 2
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            # Conv3: 64 filters, 3x3 kernel, stride 1
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        
        # Compute the flattened feature size dynamically
        self._feature_size = self._get_conv_output_size(input_shape)
        
        # === Value Stream V(s) ===
        # Outputs a single scalar: the state value
        self.value_stream = nn.Sequential(
            nn.Linear(self._feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        )
        
        # === Advantage Stream A(s, a) ===
        # Outputs one value per action: the advantage of each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(self._feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_actions),
        )
        
        # Initialize weights using Kaiming (He) initialization
        self._initialize_weights()

    def _get_conv_output_size(self, shape: Tuple[int, ...]) -> int:
        """
        Compute the output size of the convolutional layers by
        performing a forward pass with a dummy tensor.
        
        Args:
            shape: Input tensor shape (C, H, W).
            
        Returns:
            Flattened feature vector size (int).
        """
        with torch.no_grad():
            dummy = torch.zeros(1, *shape)
            output = self.features(dummy)
            return int(np.prod(output.shape[1:]))

    def _initialize_weights(self) -> None:
        """
        Apply Kaiming (He) uniform initialization to all layers.
        
        This initialization scheme is designed for ReLU activations
        and helps prevent vanishing/exploding gradients in deep networks.
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Dueling DQN.
        
        Computes Q-values by combining the value and advantage streams:
            Q(s, a) = V(s) + (A(s, a) - mean_a'(A(s, a')))
        
        The mean subtraction ensures identifiability: given Q, we can
        uniquely recover V and A (up to a constant shift).
        
        Args:
            x: Batch of observations, shape (B, C, H, W).
               Values should be in [0, 1] (normalized pixel values).
        
        Returns:
            Q-values for all actions, shape (B, n_actions).
        """
        # Shared convolutional features
        features = self.features(x)
        features = features.reshape(features.size(0), -1)  # Flatten
        
        # Separate streams
        value: torch.Tensor = self.value_stream(features)          # (B, 1)
        advantage: torch.Tensor = self.advantage_stream(features)  # (B, n_actions)
        
        # Combine: Q = V + (A - mean(A))
        # Mean subtraction for identifiability (Wang et al., 2016, Eq. 9)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values

    def get_action(self, state: torch.Tensor) -> int:
        """
        Select the greedy action (argmax Q) for a single state.
        
        Args:
            state: Single observation tensor, shape (1, C, H, W).
        
        Returns:
            Action index with highest Q-value.
        """
        with torch.no_grad():
            q_values = self.forward(state)
            return int(q_values.argmax(dim=1).item())
