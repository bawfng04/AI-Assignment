# ==============================================================================
# HW4 — Dueling Double DQN (D3QN) for Atari Games
# ==============================================================================
"""
Production-grade Reinforcement Learning framework implementing
Dueling Double Deep Q-Network (D3QN) with Prioritized Experience Replay.

Modules:
    - network: Dueling DQN CNN architecture
    - agent: D3QN agent with Double DQN target computation
    - replay_buffer: Prioritized Experience Replay with SumTree
    - utils: DeepMind-standard Atari environment wrappers
    - logger: TensorBoard experiment tracking
"""
