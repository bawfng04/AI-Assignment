# ==============================================================================
# HW4 — Dueling Double DQN (D3QN) Agent
# ==============================================================================
"""
D3QN Agent combining:
    1. Double DQN — decouples action selection from evaluation to fix
       overestimation bias (van Hasselt et al., 2016).
    2. Dueling Architecture — decomposes Q into V(s) + A(s,a) for better
       state-value estimation (Wang et al., 2016).
    3. Prioritized Experience Replay — samples important transitions
       more frequently (Schaul et al., 2016).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any

from src.network import DuelingDQN
from src.replay_buffer import PrioritizedReplayBuffer


class D3QNAgent:
    """
    Dueling Double DQN Agent with Prioritized Experience Replay.
    
    Key mechanisms:
        - epsilon-greedy exploration with linear decay
        - Double DQN target computation (online selects, target evaluates)
        - Huber loss (SmoothL1) weighted by IS weights from PER
        - Hard or soft (Polyak) target network updates
    
    Args:
        state_shape:  Observation shape (C, H, W).
        n_actions:    Number of discrete actions.
        config:       Dictionary of hyperparameters from YAML config.
        device:       Torch device ('cpu' or 'cuda').
    """

    def __init__(self, state_shape: tuple, n_actions: int,
                 config: Dict[str, Any], device: torch.device) -> None:
        self.n_actions = n_actions
        self.device = device
        self.gamma = config["agent"]["gamma"]
        self.tau = config["agent"]["tau"]
        self.target_update_freq = config["agent"]["target_update_freq"]
        self.grad_clip = config["agent"]["grad_clip"]
        self.batch_size = config["agent"]["batch_size"]

        # Exploration schedule
        self.epsilon = config["epsilon"]["start"]
        self.epsilon_end = config["epsilon"]["end"]
        self.epsilon_decay_steps = config["epsilon"]["decay_steps"]
        self._eps_step = 0

        # Networks
        self.online_net = DuelingDQN(state_shape, n_actions).to(device)
        self.target_net = DuelingDQN(state_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # Target net is never trained directly

        # Optimizer & loss
        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=config["agent"]["learning_rate"],
        )
        self.loss_fn = nn.SmoothL1Loss(reduction="none")  # Huber loss

        # Replay buffer (PER)
        buf_cfg = config["buffer"]
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=buf_cfg["capacity"],
            alpha=buf_cfg["alpha"],
            beta_start=buf_cfg["beta_start"],
            beta_end=buf_cfg["beta_end"],
            beta_anneal_steps=buf_cfg["beta_anneal_steps"],
            epsilon=buf_cfg["epsilon"],
        )

        # Counters
        self.learn_step_counter = 0

    @property
    def current_epsilon(self) -> float:
        """Current epsilon value (linearly decayed)."""
        return self.epsilon

    def select_action(self, state: np.ndarray) -> int:
        """
        Select action using epsilon-greedy policy.
        
        With probability epsilon: random action (exploration).
        Otherwise: greedy action from online network (exploitation).
        """
        # Decay epsilon
        self._eps_step += 1
        self.epsilon = max(
            self.epsilon_end,
            1.0 - (1.0 - self.epsilon_end) * self._eps_step / self.epsilon_decay_steps,
        )

        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        state_t = torch.tensor(state, dtype=torch.float32,
                               device=self.device).unsqueeze(0)
        return self.online_net.get_action(state_t)

    def store_transition(self, state: np.ndarray, action: int,
                         reward: float, next_state: np.ndarray,
                         done: bool) -> None:
        """Store a transition in the replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)

    def learn(self) -> Dict[str, float]:
        """
        Perform one gradient step using Double DQN with PER.
        
        Returns dict with: loss, mean_q, grad_norm.
        
        Double DQN target:
            y = r + gamma * Q_target(s', argmax_a' Q_online(s', a'))
        """
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0, "mean_q": 0.0, "grad_norm": 0.0}

        # Sample from PER
        (states, actions, rewards, next_states, dones,
         tree_indices, is_weights) = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states_t = torch.tensor(states, device=self.device)
        actions_t = torch.tensor(actions, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t = torch.tensor(dones, device=self.device).unsqueeze(1)
        weights_t = torch.tensor(is_weights, device=self.device).unsqueeze(1)

        # --- Double DQN Target Computation ---
        # Step 1: Online network selects best action for next state
        with torch.no_grad():
            next_q_online = self.online_net(next_states_t)
            best_actions = next_q_online.argmax(dim=1, keepdim=True)

            # Step 2: Target network evaluates that action
            next_q_target = self.target_net(next_states_t)
            next_q_value = next_q_target.gather(1, best_actions)

            # TD target: y = r + gamma * Q_target(s', a*) * (1 - done)
            td_target = rewards_t + self.gamma * next_q_value * (1.0 - dones_t)

        # Current Q-values for taken actions
        current_q = self.online_net(states_t).gather(1, actions_t)

        # Weighted Huber loss (IS weights from PER)
        element_wise_loss = self.loss_fn(current_q, td_target)
        loss = (element_wise_loss * weights_t).mean()

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        grad_norm = nn.utils.clip_grad_norm_(
            self.online_net.parameters(), self.grad_clip
        )
        self.optimizer.step()

        # Update PER priorities with new TD-errors
        td_errors = (current_q - td_target).detach().cpu().numpy().flatten()
        self.replay_buffer.update_priorities(tree_indices, np.abs(td_errors))

        # Update target network
        self.learn_step_counter += 1
        if self.tau < 1.0:
            self._soft_update()
        elif self.learn_step_counter % self.target_update_freq == 0:
            self._hard_update()

        return {
            "loss": loss.item(),
            "mean_q": current_q.mean().item(),
            "grad_norm": float(grad_norm),
        }

    def _hard_update(self) -> None:
        """Copy online network weights to target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def _soft_update(self) -> None:
        """Polyak averaging: theta_target = tau*theta_online + (1-tau)*theta_target."""
        for target_param, online_param in zip(
            self.target_net.parameters(), self.online_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )

    def save_checkpoint(self, filepath: str) -> None:
        """Save agent state to a checkpoint file."""
        checkpoint = {
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "learn_step_counter": self.learn_step_counter,
            "eps_step": self._eps_step,
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> None:
        """Load agent state from a checkpoint file."""
        checkpoint = torch.load(filepath, map_location=self.device,
                                weights_only=True)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.learn_step_counter = checkpoint["learn_step_counter"]
        self._eps_step = checkpoint["eps_step"]
