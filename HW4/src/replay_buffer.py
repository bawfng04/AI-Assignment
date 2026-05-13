# ==============================================================================
# HW4 — Prioritized Experience Replay (PER) with SumTree
# ==============================================================================
"""
Implements Prioritized Experience Replay using a Sum-Tree data structure.

Reference:
    Schaul, T., et al. (2016). "Prioritized Experience Replay." ICLR 2016.
"""

# PER + sumtree
# 

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class Transition:
    """A single experience transition (s, a, r, s', done)."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class SumTree:
    """Binary Sum-Tree for O(log N) proportional-priority sampling."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data: list[Optional[Transition]] = [None] * capacity
        self.write_idx = 0
        self.size = 0

    def _propagate(self, idx: int, delta: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def _retrieve(self, idx: int, value: float) -> int:
        left = 2 * idx + 1
        right = 2 * idx + 2
        if left >= len(self.tree):
            return idx
        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.tree[left])

    @property
    def total_priority(self) -> float:
        return float(self.tree[0])

    @property
    def max_priority(self) -> float:
        if self.size == 0:
            return 1.0
        leaf_start = self.capacity - 1
        return float(np.max(self.tree[leaf_start:leaf_start + self.size]))

    def add(self, priority: float, transition: Transition) -> None:
        tree_idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = transition
        self.update(tree_idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_idx: int, priority: float) -> None:
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, delta)

    def get(self, value: float) -> Tuple[int, float, Transition]:
        tree_idx = self._retrieve(0, value)
        data_idx = tree_idx - self.capacity + 1
        return tree_idx, self.tree[tree_idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    PER buffer with importance-sampling correction.
    
    Args:
        capacity: Max transitions. alpha: priority exponent.
        beta_start/end: IS exponent annealing range.
        beta_anneal_steps: Steps to anneal beta. epsilon: prevents zero priority.
    """

    def __init__(self, capacity: int = 100000, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_end: float = 1.0,
                 beta_anneal_steps: int = 100000, epsilon: float = 1e-6) -> None:
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_anneal_steps = beta_anneal_steps
        self.epsilon = epsilon
        self._step = 0

    @property
    def beta(self) -> float:
        fraction = min(1.0, self._step / max(1, self.beta_anneal_steps))
        return self.beta_start + fraction * (self.beta_end - self.beta_start)

    def __len__(self) -> int:
        return self.tree.size

    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool) -> None:
        # Luu state dang uint8 de tiet kiem RAM (giam 4x so voi float32)
        # state hien tai la float32 [0,1] tu preprocess_observation
        state_u8 = (state * 255).clip(0, 255).astype(np.uint8)
        next_state_u8 = (next_state * 255).clip(0, 255).astype(np.uint8)
        transition = Transition(state=state_u8, action=action, reward=reward,
                                next_state=next_state_u8, done=done)
        max_p = self.tree.max_priority
        priority = max_p ** self.alpha if max_p > 0 else 1.0
        self.tree.add(priority, transition)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray,
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a prioritized batch with IS weights (stratified)."""
        self._step += 1
        states, actions, rewards, next_states, dones = [], [], [], [], []
        indices, priorities = [], []
        segment = self.tree.total_priority / batch_size

        for i in range(batch_size):
            value = np.random.uniform(segment * i, segment * (i + 1))
            tree_idx, priority, transition = self.tree.get(value)
            if transition is None:
                value = np.random.uniform(0, self.tree.total_priority)
                tree_idx, priority, transition = self.tree.get(value)
            indices.append(tree_idx)
            priorities.append(priority)
            states.append(transition.state)
            actions.append(transition.action)
            rewards.append(transition.reward)
            next_states.append(transition.next_state)
            dones.append(transition.done)

        priorities_arr = np.array(priorities, dtype=np.float64)
        sampling_probs = np.maximum(priorities_arr / self.tree.total_priority, 1e-10)
        is_weights = (len(self) * sampling_probs) ** (-self.beta)
        is_weights /= is_weights.max()

        return (np.array(states, dtype=np.uint8).astype(np.float32) / 255.0,
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.uint8).astype(np.float32) / 255.0,
                np.array(dones, dtype=np.float32),
                np.array(indices, dtype=np.int64),
                np.array(is_weights, dtype=np.float32))

    def update_priorities(self, tree_indices: np.ndarray,
                          td_errors: np.ndarray) -> None:
        """Update priorities: p_i = (|delta_i| + eps)^alpha."""
        for idx, td_error in zip(tree_indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(int(idx), float(priority))
