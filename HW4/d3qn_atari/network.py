import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingCNN(nn.Module):
    """
    Dueling Double DQN Convolutional Neural Network.
    Expects input shape (Batch, 4, 84, 84).
    """

    def __init__(self, input_dim: int, num_actions: int):
        super(DuelingCNN, self).__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions

        # Feature extraction (CNN)
        self.conv1 = nn.Conv2d(input_dim, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Dueling streams
        # Compute flattened size: 84x84 -> conv1 -> 20x20 -> conv2 -> 9x9 -> conv3 -> 7x7
        # 64 channels * 7 * 7 = 3136
        fc_input_dim = 3136

        # Value stream V(s)
        self.value_fc = nn.Linear(fc_input_dim, 512)
        self.value = nn.Linear(512, 1)

        # Advantage stream A(s, a)
        self.advantage_fc = nn.Linear(fc_input_dim, 512)
        self.advantage = nn.Linear(512, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, C, H, W). Values expected in [0, 1].
        Returns:
            Q(s, a) values of shape (B, num_actions).
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        val = F.relu(self.value_fc(x))
        val = self.value(val)

        adv = F.relu(self.advantage_fc(x))
        adv = self.advantage(adv)

        # Combine streams: Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        adv_mean = adv.mean(dim=1, keepdim=True)
        q_values = val + (adv - adv_mean)

        return q_values
