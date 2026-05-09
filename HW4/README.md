# 🎮 Dueling Double DQN (D3QN) for Atari Games

A production-grade PyTorch implementation of **Dueling Double Deep Q-Network (D3QN)** with **Prioritized Experience Replay (PER)** for Atari game environments.

## 🏗️ Architecture

This implementation combines three key advances in Deep RL:

| Component | Paper | Key Benefit |
|---|---|---|
| **Double DQN** | van Hasselt et al., 2016 | Eliminates overestimation bias |
| **Dueling Architecture** | Wang et al., 2016 | Better state-value estimation |
| **Prioritized Replay** | Schaul et al., 2016 | Efficient experience utilization |

## 📁 Project Structure

```
HW4/
├── configs/default.yaml      # Hyperparameter configuration
├── src/
│   ├── network.py             # Dueling DQN CNN architecture
│   ├── agent.py               # D3QN agent (selection, learning, updates)
│   ├── replay_buffer.py       # PER with SumTree data structure
│   ├── utils.py               # DeepMind Atari wrappers
│   └── logger.py              # TensorBoard logging
├── train.py                   # Training loop
├── eval.py                    # Evaluation & recording
└── report/report.md           # Theory report skeleton
```

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Training
```bash
# Default: PongNoFrameskip-v4
python train.py

# Custom environment
python train.py --env BreakoutNoFrameskip-v4 --total-timesteps 2000000

# Use specific config
python train.py --config configs/default.yaml --seed 123
```

### Evaluation
```bash
# Evaluate trained agent
python eval.py --checkpoint checkpoints/d3qn_PongNoFrameskip-v4_final.pt

# Record gameplay as GIF
python eval.py --checkpoint checkpoints/d3qn_final.pt --record gameplay.gif

# Record as MP4
python eval.py --checkpoint checkpoints/d3qn_final.pt --record gameplay.mp4
```

### TensorBoard
```bash
tensorboard --logdir runs/
```

## ⚙️ Key Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `gamma` | 0.99 | Discount factor |
| `learning_rate` | 1e-4 | Adam optimizer LR |
| `batch_size` | 32 | Mini-batch size |
| `buffer_capacity` | 100,000 | Replay buffer size |
| `target_update_freq` | 10,000 | Target network update interval |
| `epsilon_decay_steps` | 100,000 | Linear ε decay duration |
| `PER alpha` | 0.6 | Priority exponent |
| `PER beta` | 0.4 → 1.0 | IS weight annealing |

## 📊 Tracked Metrics (TensorBoard)

- `episode/reward` — Per-episode total reward
- `episode/epsilon` — Exploration rate decay
- `train/loss` — Huber loss per training step
- `train/mean_q_value` — Average Q-value estimates
- `train/grad_norm` — Gradient norm (monitors stability)
- `buffer/per_beta` — PER importance-sampling exponent

## 📚 References

1. Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." *Nature*.
2. van Hasselt, H., et al. (2016). "Deep Reinforcement Learning with Double Q-learning." *AAAI*.
3. Wang, Z., et al. (2016). "Dueling Network Architectures for Deep Reinforcement Learning." *ICML*.
4. Schaul, T., et al. (2016). "Prioritized Experience Replay." *ICLR*.
