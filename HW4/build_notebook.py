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

# ===================== CELL 1: Title =====================
cells.append(create_cell("markdown",
"# \U0001F3AE Dueling Double DQN (D3QN) \u2014 Atari RL Agent\n"
"**B\u00e0i T\u1eadp L\u1edbn 4 \u2014 Tr\u00ed tu\u1ec7 Nh\u00e2n t\u1ea1o**\n\n"
"Notebook t\u1ed5ng h\u1ee3p to\u00e0n b\u1ed9 m\u00e3 ngu\u1ed3n d\u1ef1 \u00e1n D3QN + Prioritized Experience Replay.\n\n"
"**Ki\u1ebfn tr\u00fac k\u1ebft h\u1ee3p 3 c\u1ea3i ti\u1ebfn ch\u00ednh:**\n"
"- **Double DQN** \u2014 Ch\u1ed1ng Overestimation Bias\n"
"- **Dueling Network** \u2014 T\u00e1ch bi\u1ec7t V(s) v\u00e0 A(s,a)\n"
"- **Prioritized Experience Replay** \u2014 L\u1ea5y m\u1eabu th\u00f4ng minh b\u1eb1ng SumTree"))

# ===================== CELL 2: pip install =====================
cells.append(create_cell("markdown", "## 1. C\u00e0i \u0111\u1eb7t th\u01b0 vi\u1ec7n"))
cells.append(create_cell("code",
"# Cai dat moi truong Atari\n"
"!pip install -q gymnasium[atari] ale-py autorom\n"
"!pip install -q opencv-python-headless imageio[ffmpeg]\n\n"
"import subprocess\n"
"subprocess.run(['AutoROM', '--accept-license'], capture_output=True)"))

# ===================== CELL 3: Imports =====================
cells.append(create_cell("markdown", "## 2. Import th\u01b0 vi\u1ec7n"))
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
"from typing import Optional, Tuple, Any, SupportsFloat, Dict, List\n"
"from tqdm.notebook import tqdm\n"
"from IPython.display import Video, display\n\n"
"# Dang ky namespace ALE voi Gymnasium (bat buoc voi gymnasium >= 1.0)\n"
"gym.register_envs(ale_py)\n\n"
"device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
"print(f'Device: {device}')\n"
"print(f'ALE envs registered: {len([e for e in gym.envs.registry if \"ALE\" in e])} envs')"))

# ===================== CELL 4: Utils (Atari Wrappers) =====================
cells.append(create_cell("markdown",
"## 3. Ti\u1ec1n x\u1eed l\u00fd M\u00f4i tr\u01b0\u1eddng Atari (Wrappers)\n"
"C\u00e1c wrapper theo chu\u1ea9n DeepMind: NoopReset \u2192 MaxAndSkip \u2192 EpisodicLife \u2192 Fire \u2192 WarpFrame \u2192 ClipReward \u2192 FrameStack"))

utils_code = read_file("src/utils.py")
# Cut only the file header docstring + import lines, keep from set_global_seeds onward
utils_code = re.sub(r'^.*?(?=def set_global_seeds)', '', utils_code, flags=re.DOTALL)
cells.append(create_cell("code", utils_code.strip()))

# ===================== CELL 5: Replay Buffer (SumTree + PER) =====================
cells.append(create_cell("markdown",
"## 4. Prioritized Experience Replay (SumTree & Buffer)\n"
"C\u1ea5u tr\u00fac SumTree cho ph\u00e9p l\u1ea5y m\u1eabu theo x\u00e1c su\u1ea5t \u01b0u ti\u00ean v\u1edbi th\u1eddi gian O(log N)."))

buffer_code = read_file("src/replay_buffer.py")
# Keep from @dataclass onward (includes Transition + SumTree + PER)
buffer_code = re.sub(r'^.*?(?=@dataclass)', '', buffer_code, flags=re.DOTALL)
cells.append(create_cell("code", buffer_code.strip()))

# ===================== CELL 6: Network (Dueling DQN) =====================
cells.append(create_cell("markdown",
"## 5. Ki\u1ebfn tr\u00fac M\u1ea1ng N\u01a1-ron (Dueling DQN)\n"
"T\u00e1ch hai lu\u1ed3ng Value V(s) v\u00e0 Advantage A(s,a), k\u1ebft h\u1ee3p b\u1eb1ng c\u00f4ng th\u1ee9c:\n"
"$$Q(s,a) = V(s) + A(s,a) - \\\\text{mean}(A)$$"))

network_code = read_file("src/network.py")
network_code = re.sub(r'^.*?(?=class DuelingDQN)', '', network_code, flags=re.DOTALL)
cells.append(create_cell("code", network_code.strip()))

# ===================== CELL 7: Agent (D3QN) =====================
cells.append(create_cell("markdown",
"## 6. T\u00e1c t\u1eed D3QN (Agent Logic)\n"
"K\u1ebft h\u1ee3p Double DQN + Dueling + PER trong m\u1ed9t Agent duy nh\u1ea5t."))

agent_code = read_file("src/agent.py")
agent_code = re.sub(r'^.*?(?=class D3QNAgent)', '', agent_code, flags=re.DOTALL)
cells.append(create_cell("code", agent_code.strip()))

# ===================== CELL 8: Logger =====================
cells.append(create_cell("markdown", "## 7. Logger (phi\u00ean b\u1ea3n \u0111\u01a1n gi\u1ea3n cho Colab)"))
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
"    def log_training_step(self, step, loss, mean_q, grad_norm):\n"
"        pass\n"
"    def log_buffer_stats(self, step, size, beta):\n"
"        pass\n"
"    def close(self):\n"
"        self.f.close()"))

# ===================== CELL 9: Config =====================
cells.append(create_cell("markdown",
"## 8. C\u1ea5u h\u00ecnh Hu\u1ea5n luy\u1ec7n\n"
"Config n\u00e0y \u0111\u00e3 \u0111\u01b0\u1ee3c t\u1ed1i u\u01b0 cho Colab FREE tier (12GB RAM).\n"
"Buffer uint8 + 30K capacity = ~1.6GB RAM. \u0110\u1ec3 train to\u00e0n b\u1ed9 300K steps m\u1ea5t kho\u1ea3ng 1-2 gi\u1edd."))

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
"    # Buffer 30K uint8: chi ~1.6GB RAM (phu hop Colab FREE)\n"
"    'buffer': {\n"
"        'capacity': 30000,\n"
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
"        'total_timesteps': 300000,  # ~1-2 gio tren Colab T4\n"
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
"}\n\n"
"print('Config da san sang!')\n"
"print(f\"   Env: {config['env']['name']}\")\n"
"print(f\"   Buffer: {config['buffer']['capacity']:,} transitions (uint8, ~1.6GB RAM)\")\n"
"print(f\"   Total timesteps: {config['training']['total_timesteps']:,}\")"
)
cells.append(create_cell("code", config_code))

# ===================== CELL 10: Train function =====================
cells.append(create_cell("markdown",
"## 9. V\u00f2ng l\u1eb7p Hu\u1ea5n luy\u1ec7n (Training Loop)\n"
"\u0110\u00e2y l\u00e0 h\u00e0m `train()` ch\u00ednh. Ch\u1ea1y cell b\u00ean d\u01b0\u1edbi \u0111\u1ec3 b\u1eaft \u0111\u1ea7u hu\u1ea5n luy\u1ec7n."))

train_func = read_file("train.py")
# Extract only the train() function (lines 60-198)
match = re.search(r'(def train\(config.*?(?=\ndef main|\nif __name__))', train_func, re.DOTALL)
if match:
    train_body = match.group(1).strip()
else:
    train_body = "# ERROR: Could not extract train function"
cells.append(create_cell("code", train_body))

# ===================== CELL 11: Run training =====================
cells.append(create_cell("markdown",
"## 10. Hu\u1ea5n luy\u1ec7n ho\u1eb7c Load Checkpoint\n"
"Ch\u1ecdn **Option A** (train m\u1edbi) ho\u1eb7c **Option B** (load file `.pt` \u0111\u00e3 c\u00f3) r\u1ed3i ch\u1ea1y cell t\u01b0\u01a1ng \u1ee9ng."))

cells.append(create_cell("code",
"# ========== OPTION A: Train m\u1edbi t\u1eeb \u0111\u1ea7u ==========\n"
"# B\u1ecf comment 1 d\u00f2ng d\u01b0\u1edbi n\u1ebfu mu\u1ed1n train m\u1edbi\n"
"# agent = train(config, device)\n\n"
"# ========== OPTION B: Load checkpoint \u0111\u00e3 train s\u1eb5n ==========\n"
"# Thay ten file checkpoint cho dung voi file cua ban\n"
"CHECKPOINT = 'checkpoints/d3qn_ALE_Pong-v5_final.pt'\n\n"
"import os, glob\n"
"# Tu dong tim checkpoint moi nhat neu file khong ton tai\n"
"if not os.path.exists(CHECKPOINT):\n"
"    pts = sorted(glob.glob('checkpoints/*.pt'))\n"
"    CHECKPOINT = pts[-1] if pts else None\n"
"    print(f'Auto-found checkpoint: {CHECKPOINT}')\n\n"
"if CHECKPOINT and os.path.exists(CHECKPOINT):\n"
"    env_tmp = make_atari_env(config['env']['name'], seed=42, clip_rewards=False, episodic_life=False)\n"
"    obs_shape = env_tmp.observation_space.shape\n"
"    state_shape = (obs_shape[2], obs_shape[0], obs_shape[1])\n"
"    n_actions = env_tmp.action_space.n\n"
"    env_tmp.close()\n\n"
"    agent = D3QNAgent(state_shape, n_actions, config, device)\n"
"    agent.load_checkpoint(CHECKPOINT)\n"
"    agent.online_net.eval()\n"
"    print(f'Agent loaded from: {CHECKPOINT}')\n"
"else:\n"
"    print('Khong tim thay checkpoint! Hay chay Option A de train truoc.')"))

# ===================== CELL 12: Evaluation =====================
cells.append(create_cell("markdown",
"## 11. \u0110\u00e1nh gi\u00e1 & Ghi h\u00ecnh Gameplay\n"
"Agent m\u00e0u **xanh l\u00e1 (b\u00ean ph\u1ea3i)** l\u00e0 D3QN c\u1ee7a b\u1ea1n. \n"
"Agent m\u00e0u **cam (b\u00ean tr\u00e1i)** l\u00e0 bot AI c\u1ee7a Atari.\n"
"S\u1ed1 tr\u00ean m\u00e0n h\u00ecnh l\u00e0 t\u1ec9 s\u1ed1: `\u0110\u1ed1i th\u1ee7 | Agent`.\n"
"Reward -19 = thua 19 \u0111i\u1ec3m t\u1ed5ng c\u1ed9ng. C\u1ea7n ~1-2M steps \u0111\u1ec3 agent b\u1eaft \u0111\u1ea7u th\u1eafng."))

eval_func = read_file("eval.py")
# Extract evaluate() and save_recording()
match_eval = re.search(r'(def evaluate\(.*?)(def main\b|\nif __name__)', eval_func, re.DOTALL)
if match_eval:
    eval_body = match_eval.group(1).strip()
else:
    eval_body = "# ERROR: Could not extract evaluate function"
cells.append(create_cell("code", eval_body))

cells.append(create_cell("code",
"# Chay evaluation va ghi hinh (5 episodes)\n"
"rewards = evaluate(agent, config['env']['name'], n_episodes=5, seed=42, record_path='gameplay.mp4')\n\n"
"# Hien thi video ngay trong Colab\n"
"display(Video('gameplay.mp4', embed=True))"))

# Cell download checkpoint + video ve may
cells.append(create_cell("markdown", "### \u2b07\ufe0f T\u1ea3i file v\u1ec1 m\u00e1y"))
cells.append(create_cell("code",
"from google.colab import files\n\n"
"# Tai video gameplay\n"
"if os.path.exists('gameplay.mp4'):\n"
"    files.download('gameplay.mp4')\n\n"
"# Tai checkpoint cuoi cung\n"
"ckpts = sorted(glob.glob('checkpoints/*.pt'))\n"
"if ckpts:\n"
"    files.download(ckpts[-1])\n"
"    print(f'Downloaded: {ckpts[-1]}')\n\n"
"# Tai log training\n"
"if os.path.exists('logs/training.log'):\n"
"    files.download('logs/training.log')"))

# ===================== CELL 13: Baseline section header =====================
cells.append(create_cell("markdown",
"---\n"
"## Baseline Comparison: Vanilla DQN vs Double DQN\n"
"Ch\u1ea1y 2 cell b\u00ean d\u01b0\u1edbi \u0111\u1ec3 hu\u1ea5n luy\u1ec7n c\u00e1c baseline v\u1edbi **c\u00f9ng b\u1ed9 siêu tham s\u1ed1** nh\u01b0 D3QN.\n"
"K\u1ebft qu\u1ea3 s\u1ebd \u0111\u01b0\u1ee3c so s\u00e1nh trong cell cu\u1ed1i."))

# ===================== CELL 14: Vanilla DQN code =====================
cells.append(create_cell("markdown",
"### Vanilla DQN Baseline\n"
"Không có Dueling, không có PER — ch\u1ec9 dùng Replay Buffer ng\u1eabu nhiên."))

dqn_code = read_file("compare/vanilla_dqn/train_dqn.py")
# Xóa phần sys.path và from src import (sẽ dùng hàm đã define ở trên)
dqn_code = re.sub(r'import sys\n', '', dqn_code)
dqn_code = re.sub(r'import yaml\n', '', dqn_code)
dqn_code = re.sub(r'import time\n', '', dqn_code)
dqn_code = re.sub(r'import argparse\n', '', dqn_code)
dqn_code = re.sub(r'sys\.path\.append.*\n', '', dqn_code)
dqn_code = re.sub(r'from src\.\S+ import .*\n', '', dqn_code)
dqn_code = re.sub(r'from src import .*\n', '', dqn_code)
# Xóa hàm main() và if __name__
dqn_code = re.sub(r'\nif __name__.*', '', dqn_code, flags=re.DOTALL)
# Rename train() -> train_vanilla() de tranh conflict voi D3QN
dqn_code = re.sub(r'^def train\(', 'def train_vanilla(', dqn_code, flags=re.MULTILINE)
cells.append(create_cell("code", dqn_code.strip()))

cells.append(create_cell("code",
"# Chay Vanilla DQN baseline\n"
"print('=== Training Vanilla DQN Baseline ===')\n"
"vanilla_config = dict(config)  # Copy config tu D3QN\n"
"vanilla_agent = train_vanilla(vanilla_config, device)"))

# ===================== CELL 15: Double DQN code =====================
cells.append(create_cell("markdown",
"### Double DQN Baseline\n"
"Thêm cơ chế Double target (chọn action bằng online, evaluate bằng target), v\u1eabn không có Dueling hay PER."))

ddqn_code = read_file("compare/double_dqn/train_ddqn.py")
ddqn_code = re.sub(r'import sys\n', '', ddqn_code)
ddqn_code = re.sub(r'import yaml\n', '', ddqn_code)
ddqn_code = re.sub(r'import time\n', '', ddqn_code)
ddqn_code = re.sub(r'import argparse\n', '', ddqn_code)
ddqn_code = re.sub(r'sys\.path\.append.*\n', '', ddqn_code)
ddqn_code = re.sub(r'from src\.\S+ import .*\n', '', ddqn_code)
ddqn_code = re.sub(r'from src import .*\n', '', ddqn_code)
ddqn_code = re.sub(r'\nif __name__.*', '', ddqn_code, flags=re.DOTALL)
# Rename train() -> train_double() de tranh conflict
ddqn_code = re.sub(r'^def train\(', 'def train_double(', ddqn_code, flags=re.MULTILINE)
cells.append(create_cell("code", ddqn_code.strip()))

cells.append(create_cell("code",
"# Chay Double DQN baseline\n"
"print('=== Training Double DQN Baseline ===')\n"
"double_config = dict(config)\n"
"double_agent = train_double(double_config, device)"))

# ===================== CELL 16: Plot =====================
cells.append(create_cell("markdown", "## 12. Tr\u1ef1c quan h\u00f3a k\u1ebft qu\u1ea3"))
cells.append(create_cell("code",
"import re as regex\n"
"import os\n\n"
"log_path = 'logs/training.log'\n"
"if not os.path.exists(log_path):\n"
"    print('Log chua co, hay train truoc!')\n"
"else:\n"
"    episodes_list, rewards_list = [], []\n"
"    with open(log_path, 'r') as f:\n"
"        for line in f:\n"
"            m = regex.search(r'ep=(\\\\d+),\\\\s*reward=([\\\\-\\\\d\\\\.]+)', line)\n"
"            if m:\n"
"                episodes_list.append(int(m.group(1)))\n"
"                rewards_list.append(float(m.group(2)))\n"
"    plt.figure(figsize=(12, 5))\n"
"    plt.plot(episodes_list, rewards_list, alpha=0.3, color='blue', label='Raw Reward')\n"
"    window = 50\n"
"    if len(rewards_list) > window:\n"
"        smoothed = np.convolve(rewards_list, np.ones(window)/window, mode='valid')\n"
"        plt.plot(range(window, len(rewards_list)+1), smoothed, color='red', linewidth=2, label=f'MA-{window}')\n"
"    plt.xlabel('Episode')\n"
"    plt.ylabel('Reward')\n"
"    plt.title('D3QN Training Progress')\n"
"    plt.legend()\n"
"    plt.grid(True, alpha=0.3)\n"
"    plt.tight_layout()\n"
"    plt.savefig('training_curve.png', dpi=150)\n"
"    plt.show()\n"
"    print(f'Total episodes: {len(episodes_list)}')"))



# ===================== Write + Validate =====================
with open("HW4_D3QN.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("Done! Created HW4_D3QN.ipynb")
print("Running validation...")

import subprocess, sys
result = subprocess.run([sys.executable, "validate_notebook.py"], capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("[VALIDATION FAILED]")
else:
    print("[VALIDATION PASSED] Safe to upload to Colab!")
