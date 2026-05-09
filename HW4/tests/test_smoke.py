# ==============================================================================
# Smoke Test — Validates D3QN components without Atari environment
# ==============================================================================
"""
Tests the core algorithm components (network, agent, replay buffer)
using synthetic data. Does NOT require ale-py/Atari ROMs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import yaml
from src.network import DuelingDQN
from src.replay_buffer import PrioritizedReplayBuffer
from src.agent import D3QNAgent


def test_dueling_network() -> None:
    """Test Dueling DQN forward pass and output shapes."""
    print("[1/4] Testing DuelingDQN network...")
    
    input_shape = (4, 84, 84)  # 4 stacked grayscale frames
    n_actions = 6
    batch_size = 8
    
    net = DuelingDQN(input_shape, n_actions)
    
    # Random batch of observations
    x = torch.randn(batch_size, *input_shape)
    q_values = net(x)
    
    assert q_values.shape == (batch_size, n_actions), \
        f"Expected shape ({batch_size}, {n_actions}), got {q_values.shape}"
    
    # Test greedy action selection
    single_state = torch.randn(1, *input_shape)
    action = net.get_action(single_state)
    assert 0 <= action < n_actions, f"Action {action} out of range [0, {n_actions})"
    
    # Count parameters
    total_params = sum(p.numel() for p in net.parameters())
    print(f"   ✓ Output shape: {q_values.shape}")
    print(f"   ✓ Greedy action: {action}")
    print(f"   ✓ Total parameters: {total_params:,}")
    print()


def test_per_buffer() -> None:
    """Test PER buffer add/sample/update cycle."""
    print("[2/4] Testing PrioritizedReplayBuffer...")
    
    buffer = PrioritizedReplayBuffer(
        capacity=1000, alpha=0.6, beta_start=0.4,
        beta_end=1.0, beta_anneal_steps=500, epsilon=1e-6,
    )
    
    # Fill with dummy transitions
    state_shape = (4, 84, 84)
    for i in range(100):
        state = np.random.randn(*state_shape).astype(np.float32)
        next_state = np.random.randn(*state_shape).astype(np.float32)
        buffer.add(state, action=i % 6, reward=1.0,
                   next_state=next_state, done=False)
    
    assert len(buffer) == 100, f"Expected size 100, got {len(buffer)}"
    
    # Sample a batch
    batch_size = 32
    states, actions, rewards, next_states, dones, indices, weights = \
        buffer.sample(batch_size)
    
    assert states.shape == (batch_size, *state_shape)
    assert actions.shape == (batch_size,)
    assert weights.shape == (batch_size,)
    assert np.all(weights > 0) and np.all(weights <= 1.0)
    
    # Update priorities
    td_errors = np.random.uniform(0.01, 1.0, size=batch_size)
    buffer.update_priorities(indices, td_errors)
    
    print(f"   ✓ Buffer size: {len(buffer)}")
    print(f"   ✓ Sample shapes: states={states.shape}, weights={weights.shape}")
    print(f"   ✓ IS weights range: [{weights.min():.4f}, {weights.max():.4f}]")
    print(f"   ✓ Beta: {buffer.beta:.4f}")
    print()


def test_agent() -> None:
    """Test D3QN agent action selection and learning step."""
    print("[3/4] Testing D3QNAgent...")
    
    # Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "configs", "default.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Override for fast testing
    config["buffer"]["capacity"] = 500
    config["agent"]["batch_size"] = 8
    config["epsilon"]["decay_steps"] = 100
    
    state_shape = (4, 84, 84)
    n_actions = 6
    device = torch.device("cpu")
    
    agent = D3QNAgent(state_shape, n_actions, config, device)
    
    # Test action selection
    state = np.random.randn(*state_shape).astype(np.float32)
    action = agent.select_action(state)
    assert 0 <= action < n_actions
    
    # Fill buffer with dummy transitions
    for _ in range(50):
        s = np.random.randn(*state_shape).astype(np.float32)
        ns = np.random.randn(*state_shape).astype(np.float32)
        agent.store_transition(s, np.random.randint(n_actions), 1.0, ns, False)
    
    # Test learning step
    metrics = agent.learn()
    
    print(f"   ✓ Action selected: {action}")
    print(f"   ✓ Epsilon: {agent.current_epsilon:.4f}")
    print(f"   ✓ Loss: {metrics['loss']:.6f}")
    print(f"   ✓ Mean Q: {metrics['mean_q']:.6f}")
    print(f"   ✓ Grad norm: {metrics['grad_norm']:.6f}")
    print()


def test_checkpoint() -> None:
    """Test save/load checkpoint roundtrip."""
    print("[4/4] Testing checkpoint save/load...")
    
    config_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "configs", "default.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    config["buffer"]["capacity"] = 100
    config["agent"]["batch_size"] = 4
    
    state_shape = (4, 84, 84)
    n_actions = 6
    device = torch.device("cpu")
    
    agent = D3QNAgent(state_shape, n_actions, config, device)
    
    # Save
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "_test_checkpoint.pt")
    agent.save_checkpoint(ckpt_path)
    
    # Load into new agent
    agent2 = D3QNAgent(state_shape, n_actions, config, device)
    agent2.load_checkpoint(ckpt_path)
    
    # Verify weights match
    for p1, p2 in zip(agent.online_net.parameters(),
                      agent2.online_net.parameters()):
        assert torch.equal(p1, p2), "Checkpoint mismatch!"
    
    # Cleanup
    os.remove(ckpt_path)
    
    print(f"   ✓ Checkpoint saved and loaded successfully")
    print(f"   ✓ Network weights match after roundtrip")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("  D3QN Smoke Test — Component Validation")
    print("=" * 60)
    print()
    
    test_dueling_network()
    test_per_buffer()
    test_agent()
    test_checkpoint()
    
    print("=" * 60)
    print("  ✅ ALL TESTS PASSED")
    print("=" * 60)
