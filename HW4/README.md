# 🎮 Dueling Double DQN (D3QN) for Atari Games
**Trí tuệ Nhân tạo - Bài Tập Lớn 4**

A production-grade PyTorch implementation of **Dueling Double Deep Q-Network (D3QN)** with **Prioritized Experience Replay (PER)** for Atari game environments (specifically optimized for `ALE/Pong-v5`). 

This project also includes a complete Streamlit Web Dashboard for monitoring, mathematical documentation, and baseline comparisons (Vanilla DQN & Double DQN).

## 🏗️ Architecture
This implementation combines three key advances in Deep RL:

| Component | Paper | Key Benefit |
|---|---|---|
| **Double DQN** | van Hasselt et al., 2016 | Eliminates overestimation bias |
| **Dueling Architecture** | Wang et al., 2016 | Better state-value estimation |
| **Prioritized Replay** | Schaul et al., 2016 | Efficient experience utilization |

## 📁 Project Structure
```text
HW4/
├── configs/default.yaml        # Hyperparameter configuration
├── src/
│   ├── network.py              # Dueling DQN CNN architecture
│   ├── agent.py                # D3QN agent (Double target, Hard/Soft updates)
│   ├── replay_buffer.py        # PER with SumTree data structure O(log N)
│   ├── utils.py                # DeepMind Atari wrappers (Frame Stacking, Grayscale)
│   └── logger.py               # TensorBoard & Console logging
├── compare/                    # Baseline Architectures (for ablation studies)
│   ├── vanilla_dqn/train_dqn.py   # Vanilla DQN (No Dueling, No PER)
│   └── double_dqn/train_ddqn.py   # Double DQN (No Dueling, No PER)
├── report_template/            # LaTeX Source Code for Academic Report
├── app.py                      # 🌟 Streamlit Web Dashboard UI
├── plot_logs.py                # Parser for heuristic.log to generate Metric PNGs
├── plot_baselines.py           # Generates comparative plots (D3QN vs Baselines)
├── train.py                    # Main Training loop
└── eval.py                     # Evaluation & video recording script
```

## 🚀 Quick Start

### 1. Installation
Ensure you are using Python 3.10 or 3.11 for maximum compatibility with `ale-py`.
```bash
pip install -r requirements.txt
pip install "gymnasium[atari,accept-rom-license]" ale-py
pip install streamlit matplotlib
```

### 2. Training the Main Model (D3QN)
```bash
# Default: ALE/Pong-v5
python train.py

# Custom environment & steps
python train.py --env ALE/Breakout-v5 --total-timesteps 2000000
```

### 3. Training Baselines (For Comparison)
To prove the superiority of D3QN, run the baselines:
```bash
python compare/vanilla_dqn/train_dqn.py --config configs/default.yaml
python compare/double_dqn/train_ddqn.py --config configs/default.yaml
```

### 4. Evaluation & Recording Gameplay
*Note: Due to Gymnasium namespace updates, always explicitly provide the `--env` flag.*
```bash
# Record gameplay as MP4
python eval.py --checkpoint checkpoints/d3qn_ALE/Pong-v5_20000.pt --env ALE/Pong-v5 --record gameplay.mp4

# Record as GIF
python eval.py --checkpoint checkpoints/d3qn_ALE/Pong-v5_20000.pt --env ALE/Pong-v5 --record gameplay.gif
```

## 📊 Streamlit Web Dashboard
Instead of just relying on the terminal, you can visualize the entire training process and watch the AI play using the integrated web app.

First, extract the logs and generate plots:
```bash
python plot_logs.py
python plot_baselines.py
```
Then, launch the UI:
```bash
streamlit run app.py --server.headless true
```

Play vs AI:
```bash
pip install pygame
python play_vs_ai.py --ckpt checkpoints/d3qn_ALE_Pong-v5_final.pt
# Hoặc đơn giản chỉ cần gõ lệnh sau (do đã được cấu hình đường dẫn mặc định):
# python play_vs_ai.py
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

## 📚 References
1. Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." *Nature*.
2. van Hasselt, H., et al. (2016). "Deep Reinforcement Learning with Double Q-learning." *AAAI*.
3. Wang, Z., et al. (2016). "Dueling Network Architectures for Deep Reinforcement Learning." *ICML*.
4. Schaul, T., et al. (2016). "Prioritized Experience Replay." *ICLR*.
