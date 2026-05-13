import json
import re

def create_cell(cell_type, source_str):
    lines = source_str.split("\n")
    source = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        source.append(lines[-1])
    cell = {"cell_type": cell_type, "metadata": {}, "source": source}
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell

def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "colab": {"provenance": [], "gpuType": "T4"},
        "accelerator": "GPU"
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

cells = notebook['cells']

# ===================== CELL 1: Header =====================
cells.append(create_cell("markdown",
"# \U0001F3AE Dueling Double DQN (D3QN) \u2014 Atari RL Agent\n"
"**B\u00e0i T\u1eadp L\u1edbn 4 \u2014 Tr\u00ed tu\u1ec7 Nh\u00e2n t\u1ea1o**\n\n"
"Notebook t\u1ed5ng h\u1ee3p to\u00e0n b\u1ed9 m\u00e3 ngu\u1ed3n d\u1ef1 \u00e1n D3QN + Prioritized Experience Replay c\u00f9ng c\u00e1c Baseline so s\u00e1nh.\n\n"
"**Ki\u1ebfn tr\u00fac \u0111\u01b0\u1ee3c ki\u1ec3m th\u1eed:**\n"
"- **D3QN (Ours)**: Double DQN + Dueling Network + Prioritized Experience Replay (SumTree).\n"
"- **Double DQN Baseline**: Double Target, Standard Replay Buffer (uint8 optimized).\n"
"- **Vanilla DQN Baseline**: Standard Target, Standard Replay Buffer (uint8 optimized)."))

# ===================== CELL 2: Setup =====================
cells.append(create_cell("markdown", "## 1. C\u00e0i \u0111\u1eb7t M\u00f4i tr\u01b0\u1eddng & Th\u01b0 vi\u1ec7n"))
cells.append(create_cell("code",
"# Cai dat moi truong Atari\n"
"!pip install -q gymnasium[atari] ale-py autorom\n"
"!pip install -q opencv-python-headless imageio[ffmpeg]\n\n"
"import subprocess\n"
"subprocess.run(['AutoROM', '--accept-license'], capture_output=True)\n\n"
"# Tao san cac thu muc luu tru can thiet\n"
"import os\n"
"for d in ['logs/vanilla_dqn', 'logs/double_dqn', 'checkpoints/vanilla_dqn', 'checkpoints/double_dqn']:\n"
"    os.makedirs(d, exist_ok=True)\n"
"print('Setup thu muc thanh cong!')"))

# ===================== CELL 3: Imports =====================
cells.append(create_cell("markdown", "## 2. Import Th\u01b0 vi\u1ec7n v\u00e0 Kh\u1edfi t\u1ea1o"))
cells.append(create_cell("code",
"import os\n"
"import random\n"
"import collections\n"
"import numpy as np\n"
"import torch\n"
"import torch.nn as nn\n"
"import torch.nn.functional as F\n"
"import torch.optim as optim\n"
"import gymnasium as gym\n"
"import ale_py\n"
"import cv2\n"
"import imageio\n"
"import matplotlib.pyplot as plt\n"
"from collections import deque\n"
"from dataclasses import dataclass\n"
"from pathlib import Path\n"
"from typing import Optional, Tuple, Any, Dict, List, SupportsFloat\n"
"from tqdm.notebook import tqdm\n"
"from IPython.display import Video, display\n\n"
"# Dang ky namespace ALE voi Gymnasium v1.x\n"
"gym.register_envs(ale_py)\n\n"
"device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
"print(f'Device: {device}')"))

# ===================== CELL 4: Utils =====================
cells.append(create_cell("markdown", "## 3. C\u00e1c H\u00e0m Ti\u1ec1n x\u1eed l\u00fd (Atari Wrappers)"))
utils_code = read_file("src/utils.py")
utils_code = re.sub(r'^.*?(?=def set_global_seeds)', '', utils_code, flags=re.DOTALL)
cells.append(create_cell("code", utils_code.strip()))

# ===================== CELL 5: Replay Buffers =====================
cells.append(create_cell("markdown", "## 4. C\u1ea5u tr\u00fac B\u1ed9 nh\u1edb \u0110\u1ec7m (Replay Buffers)\n"
"Bao g\u1ed5m **Prioritized Replay Buffer** (cho D3QN) v\u00e0 **Standard Replay Buffer** (cho Baselines).\n"
"T\u1ea5t c\u1ea3 \u0111\u1ec1u \u0111\u01b0\u1ee3c t\u1ed1i \u01b0u h\u00f3a l\u01b0u tr\u1eef d\u01b0\u1edbi d\u1ea1ng `uint8` \u0111\u1ec3 tr\u00e1nh OOM tr\u00ean Colab."))

buffer_code = read_file("src/replay_buffer.py")
buffer_code = re.sub(r'^.*?(?=@dataclass)', '', buffer_code, flags=re.DOTALL)

# Tích hợp thêm StandardReplayBuffer tối ưu uint8 ngay trong cell này
std_buffer_code = """
class StandardReplayBuffer:
    \"\"\"Standard Experience Replay Buffer optimized with uint8 storage.\"\"\"
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        # Nén state dạng uint8 (tiết kiệm 4x RAM)
        s_u8 = (state * 255).clip(0, 255).astype(np.uint8)
        ns_u8 = (next_state * 255).clip(0, 255).astype(np.uint8)
        self.buffer.append((s_u8, action, reward, ns_u8, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Giải nén sang float32 khi đưa lên GPU
        s_arr = np.array(states, dtype=np.uint8).astype(np.float32) / 255.0
        ns_arr = np.array(next_states, dtype=np.uint8).astype(np.float32) / 255.0

        return (
            torch.tensor(s_arr, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(ns_arr, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)
"""
cells.append(create_cell("code", buffer_code.strip() + "\n\n" + std_buffer_code.strip()))

# ===================== CELL 6: Networks =====================
cells.append(create_cell("markdown", "## 5. Ki\u1ebfn tr\u00fac M\u1ea1ng N\u01a1-ron\n"
"Bao g\u1ed5m **Dueling DQN** (chia nh\u00e1nh Value v\u00e0 Advantage) v\u00e0 **Standard Network** (cho Baselines)."))

network_code = read_file("src/network.py")
network_code = re.sub(r'^.*?(?=class DuelingDQN)', '', network_code, flags=re.DOTALL)

# Tích hợp thêm Standard DQN Network
std_network_code = """
class StandardDQN(nn.Module):
    \"\"\"Standard Deep Q-Network without Dueling architecture.\"\"\"
    def __init__(self, input_shape: Tuple[int, int, int], n_actions: int):
        super().__init__()
        c, h, w = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).view(x.size(0), -1))
"""
cells.append(create_cell("code", network_code.strip() + "\n\n" + std_network_code.strip()))

# ===================== CELL 7: Agents =====================
cells.append(create_cell("markdown", "## 6. C\u00e1c T\u00e1c t\u1eed (Agents)"))
agent_code = read_file("src/agent.py")
agent_code = re.sub(r'^.*?(?=class D3QNAgent)', '', agent_code, flags=re.DOTALL)

# Tích hợp BaselineAgent hỗ trợ Vanilla và Double DQN
std_agent_code = """
class BaselineAgent:
    \"\"\"Agent hỗ trợ huấn luyện Vanilla DQN hoặc Double DQN.\"\"\"
    def __init__(self, state_shape: Tuple[int, int, int], n_actions: int, config: dict, device: torch.device, is_double: bool = False):
        self.n_actions = n_actions
        self.device = device
        self.gamma = config['agent']['gamma']
        self.batch_size = config['agent']['batch_size']
        self.target_update_freq = config['agent']['target_update_freq']
        self.is_double = is_double

        self.online_net = StandardDQN(state_shape, n_actions).to(device)
        self.target_net = StandardDQN(state_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=config['agent']['lr'])
        self.memory = StandardReplayBuffer(config['buffer']['capacity'])
        self.step_count = 0

    def get_action(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.online_net(state_t).argmax(dim=1).item())

    def train_step(self) -> float:
        if len(self.memory) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states, actions = states.to(self.device), actions.to(self.device)
        rewards, next_states, dones = rewards.to(self.device), next_states.to(self.device), dones.to(self.device)

        q_values = self.online_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            if self.is_double:
                # Double Target: Chọn hành động bằng Online, tính giá trị bằng Target
                next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions).squeeze(-1)
            else:
                # Standard Target
                next_q = self.target_net(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        loss = F.mse_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())
"""
cells.append(create_cell("code", agent_code.strip() + "\n\n" + std_agent_code.strip()))

# ===================== CELL 8: Logger =====================
cells.append(create_cell("markdown", "## 7. Tr\u00ecnh Ghi nh\u1eadt k\u00fd (Logger)"))
cells.append(create_cell("code",
"class RLLogger:\n"
"    def __init__(self, log_dir='logs'):\n"
"        os.makedirs(log_dir, exist_ok=True)\n"
"        self.log_path = os.path.join(log_dir, 'training.log')\n"
"        self.f = open(self.log_path, 'w')\n"
"    def log_episode(self, ep, reward, length, epsilon):\n"
"        line = f'ep={ep}, reward={reward:.1f}, length={length}, eps={epsilon:.4f}'\n"
"        self.f.write(line + '\\n')\n"
"        self.f.flush()\n"
"    def close(self):\n"
"        self.f.close()"))

# ===================== CELL 9: Config =====================
cells.append(create_cell("markdown", "## 8. C\u1ea5u h\u00ecnh Si\u00eau tham s\u1ed1 (RAM Optimized)"))
config_code = (
"config = {\n"
"    'env': {\n"
"        'name': 'ALE/Pong-v5',\n"
"        'frame_stack': 4,\n"
"        'clip_rewards': True,\n"
"        'episodic_life': True\n"
"    },\n"
"    'agent': {\n"
"        'gamma': 0.99,\n"
"        'lr': 1e-4,\n"
"        'learning_rate': 1e-4,\n"
"        'batch_size': 32,\n"
"        'target_update_freq': 10000,\n"
"        'tau': 1.0,\n"
"        'grad_clip': 10.0\n"
"    },\n"
"    'buffer': {\n"
"        'capacity': 30000,  # uint8 tối ưu RAM\n"
"        'alpha': 0.6,\n"
"        'beta_start': 0.4,\n"
"        'beta_end': 1.0,\n"
"        'beta_frames': 100000,\n"
"        'beta_anneal_steps': 100000,\n"
"        'epsilon': 1e-6\n"
"    },\n"
"    'epsilon': {\n"
"        'start': 1.0,\n"
"        'end': 0.01,\n"
"        'decay_steps': 100000\n"
"    },\n"
"    'training': {\n"
"        'total_timesteps': 300000,\n"
"        'learning_starts': 10000,\n"
"        'train_freq': 4,\n"
"        'save_interval': 50000,\n"
"        'log_interval': 10\n"
"    },\n"
"    'paths': {\n"
"        'checkpoint_dir': 'checkpoints',\n"
"        'log_dir': 'logs'\n"
"    },\n"
"    'seed': 42\n"
"}\n"
"print('Config san sang!')"
)
cells.append(create_cell("code", config_code))

# ===================== CELL 10: Unified Trainer =====================
cells.append(create_cell("markdown", "## 9. H\u00e0m Hu\u1ea5n luy\u1ec7n H\u1ee3p nh\u1ea5t (Unified Trainer)\n"
"H\u1ed7 tr\u1ee3 \u0111\u1ecbnh tuy\u1ebfn m\u00f4 h\u00ecnh linh ho\u1ea1t, \u0111\u1ea3m b\u1ea3o ph\u00e2n t\u00e1ch th\u01b0 mi\u1ec1n d\u1eef li\u1ec7u."))

trainer_code = """
def train_model(model_type: str, custom_log_dir: str, custom_ckpt_dir: str):
    \"\"\"
    Unified trainer for D3QN, Vanilla DQN and Double DQN.

    D3QNAgent API  : select_action(state) | store_transition(...) | learn() | replay_buffer
    BaselineAgent  : get_action(state, eps) | memory.add(...) | train_step() | memory
    \"\"\"
    print(f"\\n{'='*15} BAT DAU TRAIN: {model_type.upper()} {'='*15}")
    set_global_seeds(config['seed'])
    env = make_atari_env(
        config['env']['name'],
        seed=config['seed'],
        clip_rewards=config['env']['clip_rewards'],
        episodic_life=config['env']['episodic_life'],
    )

    obs_shape   = env.observation_space.shape           # (H, W, C)
    state_shape = (obs_shape[2], obs_shape[0], obs_shape[1])  # -> (C, H, W)
    n_actions   = env.action_space.n

    logger   = RLLogger(custom_log_dir)
    ckpt_dir = Path(custom_ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    is_d3qn = (model_type == 'd3qn')
    if is_d3qn:
        agent = D3QNAgent(state_shape, n_actions, config, device)
    else:
        agent = BaselineAgent(state_shape, n_actions, config, device,
                              is_double=(model_type == 'ddqn'))

    eps_start     = config['epsilon']['start']
    eps_end       = config['epsilon']['end']
    eps_decay     = config['epsilon']['decay_steps']
    total_steps   = config['training']['total_timesteps']
    learn_start   = config['training']['learning_starts']
    train_freq    = config['training']['train_freq']
    save_interval = config['training']['save_interval']
    log_interval  = config['training']['log_interval']
    safe_env      = config['env']['name'].replace('/', '_')

    state, _ = env.reset()
    state     = preprocess_observation(state)
    ep_reward, ep_len = 0.0, 0
    ep_count  = 0
    recent_rewards = deque(maxlen=100)

    pbar = tqdm(total=total_steps, desc=f"[{model_type.upper()}]")

    for step in range(1, total_steps + 1):
        # Epsilon schedule cho Baseline (D3QN tự quản lý nội bộ)
        epsilon = eps_end + (eps_start - eps_end) * max(0.0, 1.0 - step / eps_decay)

        # --- Chọn hành động ---
        if is_d3qn:
            action = agent.select_action(state)         # D3QNAgent API
        else:
            action = agent.get_action(state, epsilon)   # BaselineAgent API

        # --- Bước môi trường ---
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_state = preprocess_observation(next_obs)
        done = terminated or truncated

        # --- Lưu transition ---
        if is_d3qn:
            agent.store_transition(state, action, reward, next_state, done)  # -> replay_buffer
        else:
            agent.memory.add(state, action, reward, next_state, done)        # -> StandardReplayBuffer

        state      = next_state
        ep_reward += reward
        ep_len    += 1

        # --- Học ---
        if step >= learn_start and step % train_freq == 0:
            if is_d3qn:
                agent.learn()       # xử lý cả target-network update nội bộ
            else:
                agent.train_step()

        # --- Kết thúc episode ---
        if done:
            ep_count += 1
            cur_eps = agent.current_epsilon if is_d3qn else epsilon
            recent_rewards.append(ep_reward)
            logger.log_episode(ep_count, ep_reward, ep_len, cur_eps)

            if ep_count % log_interval == 0:
                avg_r    = np.mean(recent_rewards)
                buf_size = len(agent.replay_buffer) if is_d3qn else len(agent.memory)
                pbar.write(f"[Ep {ep_count}] Avg R(100): {avg_r:.2f} | Eps: {cur_eps:.4f} | Buf: {buf_size} | Steps: {step}")

            state, _ = env.reset()
            state     = preprocess_observation(state)
            ep_reward, ep_len = 0.0, 0

        # --- Checkpoint định kỳ ---
        if step % save_interval == 0:
            pt_path = ckpt_dir / f"{model_type}_{safe_env}_{step}.pt"
            if is_d3qn:
                agent.save_checkpoint(str(pt_path))
            else:
                torch.save(agent.online_net.state_dict(), str(pt_path))
            pbar.write(f"  [Checkpoint] Saved: {pt_path}")

        pbar.update(1)

    pbar.close()
    logger.close()
    env.close()

    # --- Checkpoint cuối ---
    final_path = ckpt_dir / f"{model_type}_{safe_env}_final.pt"
    if is_d3qn:
        agent.save_checkpoint(str(final_path))
    else:
        torch.save(agent.online_net.state_dict(), str(final_path))
    print(f"  [*] Hoan tat! Final checkpoint: {final_path}")
    return agent
"""
cells.append(create_cell("code", trainer_code.strip()))

# ===================== CELL 11: Execute Pipelines =====================

cells.append(create_cell("markdown", "## 10. K\u00edch ho\u1ea1t Hu\u1ea5n luy\u1ec7n Tu\u1ea7n t\u1ef1 (Overnight Execution)\n"
"Cell n\u00e0y \u0111\u01b0\u1ee3c thi\u1ebft k\u1ebf \u0111\u1ec3 **treo m\u00e1y ch\u1ea1y qua \u0111\u00eam**. H\u1ec7 th\u1ed1ng s\u1ebd l\u1ea7n l\u01b0\u1ee3t train c\u1ea3 3 m\u00f4 h\u00ecnh v\u00e0 t\u1ef1 \u0111\u1ed9ng x\u00f3a b\u1ed9 nh\u1edb \u0111\u1ec7m gi\u1eefa c\u00e1c l\u1ea7n ch\u1ea1y \u0111\u1ec3 tr\u00e1nh tr\u00e0n RAM."))

execute_code = """
import gc
import time

# Danh sách cấu hình các mô hình cần chạy
pipelines = [
    {'type': 'd3qn', 'log': 'logs', 'ckpt': 'checkpoints'},
    {'type': 'vanilla', 'log': 'logs/vanilla_dqn', 'ckpt': 'checkpoints/vanilla_dqn'},
    {'type': 'ddqn', 'log': 'logs/double_dqn', 'ckpt': 'checkpoints/double_dqn'}
]

trained_agents = {}
start_time_total = time.time()

for pipe in pipelines:
    m_type = pipe['type']
    print(f"\\n>>> CHUẨN BỊ HUẤN LUYỆN: {m_type.upper()}")
    
    try:
        # Kích hoạt tiến trình huấn luyện
        agent = train_model(m_type, pipe['log'], pipe['ckpt'])
        trained_agents[m_type] = agent
        print(f"[OK] Huấn luyện thành công {m_type.upper()}")
        
    except Exception as e:
        print(f"[LỖI NGIÊM TRỌNG] Quá trình train {m_type.upper()} thất bại: {e}")
        print("Hệ thống sẽ bỏ qua và chạy tiếp mô hình tiếp theo để tránh lãng phí thời gian treo máy.")
    
    # --- CƠ CHẾ BẢO VỆ BỘ NHỚ (CRITICAL FOR OVERNIGHT TRAIN) ---
    print(">>> Đang dọn dẹp bộ nhớ đệm (Garbage Collection)...")
    # Xóa tham chiếu tạm thời để giải phóng Replay Buffer cũ
    if 'agent' in locals():
        del agent
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(2) # Nghỉ ngơi I/O

hours, rem = divmod(time.time() - start_time_total, 3600)
minutes, seconds = divmod(rem, 60)
print(f"\\n{'='*40}\\n[HOÀN TẤT] Tổng thời gian treo máy: {int(hours)}h {int(minutes)}m {int(seconds)}s\\n{'='*40}")

# Giữ lại D3QN agent chính trong scope để chạy Evaluation bên dưới
if 'd3qn' in trained_agents:
    d3qn_agent = trained_agents['d3qn']
"""
cells.append(create_cell("code", execute_code.strip()))

# ===================== CELL 12: Evaluation =====================
cells.append(create_cell("markdown", "## 11. \u0110\u00e1nh gi\u00e1 & Xu\u1ea5t Gameplay Video"))
eval_func = read_file("eval.py")
match_eval = re.search(r'(def evaluate\(.*?)(def main\b|\nif __name__)', eval_func, re.DOTALL)
eval_body = match_eval.group(1).strip() if match_eval else "# ERROR"
cells.append(create_cell("code", eval_body))

cells.append(create_cell("code",
"# Đánh giá D3QN Agent nếu có sẵn trong bộ nhớ\n"
"if 'd3qn_agent' in locals():\n"
"    rewards = evaluate(d3qn_agent, config['env']['name'], n_episodes=5, seed=42, record_path='gameplay.mp4')\n"
"    display(Video('gameplay.mp4', embed=True))\n"
"else:\n"
"    print('D3QN Agent chưa được load hoặc train trong phiên này.')"))

# ===================== CELL 13: Plot =====================
cells.append(create_cell("markdown", "## 12. T\u1ed5ng h\u1ee3p & So s\u00e1nh c\u00e1c \u0110\u01b0\u1eddng cong Hu\u1ea5n luy\u1ec7n"))
plot_code = r"""import re as regex
import os
import matplotlib.pyplot as plt
import numpy as np

def load_log_data(log_path):
    episodes, rewards = [], []
    if not os.path.exists(log_path):
        return episodes, rewards
    with open(log_path, 'r') as f:
        for line in f:
            m = regex.search(r'ep=(\d+),\s*reward=([\-\d\.]+)', line)
            if m:
                episodes.append(int(m.group(1)))
                rewards.append(float(m.group(2)))
    return episodes, rewards

logs = {
    'Vanilla DQN': ('logs/vanilla_dqn/training.log', '#2ca02c'),
    'Double DQN': ('logs/double_dqn/training.log', '#ff7f0e'),
    'D3QN (Ours)': ('logs/training.log', '#1f77b4')
}

plt.figure(figsize=(12, 6))
window = 50

for label, (path, color) in logs.items():
    eps, rews = load_log_data(path)
    if not eps:
        print(f"[*] Chưa có data cho {label}")
        continue
    
    print(f"[*] {label}: load thành công {len(eps)} episodes.")
    plt.plot(eps, rews, alpha=0.15, color=color)
    if len(rews) >= window:
        smoothed = np.convolve(rews, np.ones(window)/window, mode='valid')
        plt.plot(eps[window-1:], smoothed, color=color, linewidth=2.5, label=f"{label} (MA-{window})")
    else:
        plt.plot(eps, rews, color=color, linewidth=2, label=label)

plt.xlabel('Episodes', fontsize=12)
plt.ylabel('Average Reward', fontsize=12)
plt.title('Performance Comparison: D3QN vs Baselines on ALE/Pong-v5', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='lower right')
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('comparison_curve.png', dpi=300)
plt.show()
"""
cells.append(create_cell("code", plot_code.strip()))

# ===================== Write Notebook =====================
with open("HW4_D3QN.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("Done! Created HW4_D3QN.ipynb (Unified Architecture)")
print("Running validation...")

import subprocess, sys
result = subprocess.run([sys.executable, "validate_notebook.py"], capture_output=True, text=True, encoding="utf-8", errors="replace")
print(result.stdout)
if result.returncode != 0:
    print("[VALIDATION FAILED]")
else:
    print("[VALIDATION PASSED] Safe to upload to Colab!")

