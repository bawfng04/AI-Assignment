# file này huấn luyện mô hình dqn cổ điển (vanilla dqn) nguyên thủy nhất để làm mốc so sánh (baseline).
# mô hình này hoàn toàn thô sơ:
# 1. dùng bộ nhớ thường (fifo deque).
# 2. dùng mạng cnn thẳng ra q-value (không dueling).
# 3. công thức cập nhật gốc: lấy trực tiếp max q từ mạng target (rất dễ bị ảo tưởng sức mạnh - overestimation bias).

import sys
import os
import argparse
import yaml
import time
import collections
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils import make_atari_env, set_global_seeds, preprocess_observation
from src.logger import RLLogger

# ==========================================================
# 1. Standard Replay Buffer (No PER)
# ==========================================================
class StandardReplayBuffer:
    # hàng đợi lưu trữ cơ bản, không có cơ chế chấm điểm ưu tiên
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
        
    def __len__(self):
        return len(self.buffer)

# ==========================================================
# 2. Vanilla DQN Network (No Dueling)
# ==========================================================
class VanillaDQNNetwork(nn.Module):
    # mạng dqn tiêu chuẩn: cnn -> flatten -> linear -> q-values
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        # Compute dynamic size
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            out = self.features(dummy)
            feature_size = int(np.prod(out.shape[1:]))
            
        self.fc = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_actions)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)
        
    def get_action(self, state):
        with torch.no_grad():
            q_values = self.forward(state)
            return int(q_values.argmax(dim=1).item())

# ==========================================================
# 3. Vanilla DQN Agent
# ==========================================================
class VanillaDQNAgent:
    # tác nhân dqn nguyên thủy cập nhật theo thuật toán q-learning sâu gốc
    def __init__(self, state_shape, n_actions, config, device):
        self.n_actions = n_actions
        self.device = device
        self.gamma = config["agent"]["gamma"]
        self.batch_size = config["agent"]["batch_size"]
        self.target_update_freq = config["agent"]["target_update_freq"]
        
        # Epsilon
        self.epsilon_start = config["epsilon"]["start"]
        self.epsilon_end = config["epsilon"]["end"]
        self.epsilon_decay_steps = config["epsilon"]["decay_steps"]
        self._eps_step = 0
        
        # Networks
        self.online_net = VanillaDQNNetwork(state_shape, n_actions).to(device)
        self.target_net = VanillaDQNNetwork(state_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=float(config["agent"]["lr"]))
        self.replay_buffer = StandardReplayBuffer(config["buffer"]["capacity"])
        self.learn_step_counter = 0

    @property
    def current_epsilon(self) -> float:
        if self._eps_step >= self.epsilon_decay_steps:
            return self.epsilon_end
        return self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (self._eps_step / self.epsilon_decay_steps)

    def select_action(self, state: np.ndarray) -> int:
        self._eps_step += 1
        if random.random() < self.current_epsilon:
            return random.randint(0, self.n_actions - 1)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.online_net.get_action(state_t)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0, "mean_q": 0.0, "grad_norm": 0.0}

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Q(s, a)
        current_q = self.online_net(states_t).gather(1, actions_t).squeeze(1)
        mean_q = float(current_q.mean().item())

        # công thức nguyên thủy: lấy thẳng giá trị lớn nhất từ mạng target
        # (chính kẽ hở này gây ra hiện tượng thiên lệch tích lũy, định giá sai lệch)
        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(1)[0]
            td_target = rewards_t + self.gamma * max_next_q * (1.0 - dones_t)

        loss = F.huber_loss(current_q, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        grad_norm = float(sum(p.grad.data.norm(2).item() ** 2 for p in self.online_net.parameters() if p.grad is not None) ** 0.5)
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return {"loss": float(loss.item()), "mean_q": mean_q, "grad_norm": grad_norm}

# ==========================================================
# 4. Training Loop
# ==========================================================
def train(config, device):
    seed = config["seed"]
    set_global_seeds(seed)

    ckpt_dir = Path("checkpoints/vanilla_dqn")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    env_name = config["env"]["name"]
    env = make_atari_env(
        env_name, seed=seed,
        frame_stack=config["env"]["frame_stack"],
        clip_rewards=config["env"]["clip_rewards"],
        episodic_life=config["env"]["episodic_life"],
    )
    
    obs_shape = env.observation_space.shape
    state_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
    n_actions = env.action_space.n

    print("=" * 60)
    print(f"  Vanilla DQN Training (Baseline) — {env_name}")
    print("=" * 60)

    agent = VanillaDQNAgent(state_shape, n_actions, config, device)
    logger = RLLogger(log_dir="logs/vanilla_dqn")

    total_timesteps = config["training"]["total_timesteps"]
    learning_starts = config["training"]["learning_starts"]
    train_freq = config["training"]["train_freq"]

    obs, _ = env.reset()
    state = preprocess_observation(obs)

    episode_reward = 0.0
    episode_length = 0
    episode_count = 0
    recent_rewards = []

    pbar = tqdm(range(1, total_timesteps + 1), desc="Training", unit="step")

    for step in pbar:
        action = agent.select_action(state)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = preprocess_observation(next_obs)

        agent.store_transition(state, action, float(reward), next_state, done)
        state = next_state
        episode_reward += float(reward)
        episode_length += 1

        metrics = {"loss": 0.0}
        if step >= learning_starts and step % train_freq == 0:
            metrics = agent.learn()

        if done:
            episode_count += 1
            recent_rewards.append(episode_reward)
            avg_reward = np.mean(recent_rewards[-100:])
            
            pbar.set_postfix({
                "ep": episode_count,
                "reward": f"{episode_reward:.1f}",
                "avg100": f"{avg_reward:.1f}",
                "eps": f"{agent.current_epsilon:.3f}",
                "loss": f"{metrics['loss']:.4f}",
            })

            obs, _ = env.reset()
            state = preprocess_observation(obs)
            episode_reward = 0.0
            episode_length = 0

    agent.target_net.load_state_dict(agent.online_net.state_dict())
    torch.save(agent.online_net.state_dict(), ckpt_dir / f"vanilla_dqn_final.pt")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../../configs/default.yaml")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(config, device)
