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
# file này hiện thực d3qn (dueling double dqn) hoàn chỉnh.
# nó đóng vai trò là "não bộ" kết hợp cả 3 kỹ thuật:
# 1. mạng kiến trúc dueling (đã làm ở network.py).
# 2. cơ chế double dqn: tách việc chọn hành động và tính giá trị ra 2 mạng khác nhau để tránh bị ảo tưởng sức mạnh (overestimation bias).
# 3. bộ nhớ per (đã làm ở replay_buffer.py) để lấy mẫu khôn ngoan hơn.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any

from src.network import DuelingDQN
from src.replay_buffer import PrioritizedReplayBuffer


class D3QNAgent:
    # lớp quản lý toàn bộ chu trình sống của agent: chọn hành động, lưu bộ nhớ, và cập nhật trọng số.

    def __init__(self, state_shape: tuple, n_actions: int,
                 config: Dict[str, Any], device: torch.device) -> None:
        self.n_actions = n_actions
        self.device = device
        self.gamma = config["agent"]["gamma"]  # hệ số suy giảm (càng gần 1 càng tính toán xa về tương lai)
        self.tau = config["agent"]["tau"]  # hệ số cập nhật mềm (nếu dùng soft update)
        self.target_update_freq = config["agent"]["target_update_freq"]  # chu kỳ chép đè trọng số (nếu dùng hard update)
        self.grad_clip = config["agent"]["grad_clip"]
        self.batch_size = config["agent"]["batch_size"]

        # lịch trình giảm dần sự tò mò (epsilon): ban đầu tò mò khám phá nhiều, về sau khai thác đồ thị tối ưu
        self.epsilon = config["epsilon"]["start"]
        self.epsilon_end = config["epsilon"]["end"]
        self.epsilon_decay_steps = config["epsilon"]["decay_steps"]
        self._eps_step = 0

        # khởi tạo 2 mạng: online (để train và đi bài) và target (chỉ dùng làm mốc tính điểm, không train trực tiếp)
        self.online_net = DuelingDQN(state_shape, n_actions).to(device)
        self.target_net = DuelingDQN(state_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # bộ tối ưu adam và hàm mất mát huber (giúp gradient không bị văng mạnh khi gặp mẫu đột biến)
        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=config["agent"]["learning_rate"],
        )
        self.loss_fn = nn.SmoothL1Loss(reduction="none")

        # khởi tạo bộ nhớ per
        buf_cfg = config["buffer"]
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=buf_cfg["capacity"],
            alpha=buf_cfg["alpha"],
            beta_start=buf_cfg["beta_start"],
            beta_end=buf_cfg["beta_end"],
            beta_anneal_steps=buf_cfg["beta_anneal_steps"],
            epsilon=buf_cfg["epsilon"],
        )

        self.learn_step_counter = 0

    @property
    def current_epsilon(self) -> float:
        return self.epsilon

    def select_action(self, state: np.ndarray) -> int:
        # chọn hành động theo chính sách epsilon-greedy.
        # giảm dần epsilon theo thời gian
        self._eps_step += 1
        self.epsilon = max(
            self.epsilon_end,
            1.0 - (1.0 - self.epsilon_end) * self._eps_step / self.epsilon_decay_steps,
        )

        # tung đồng xu: nếu rơi vào tỉ lệ epsilon thì bấm bừa 1 nút (tò mò khám phá)
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        # ngược lại: đưa ảnh vào mạng online để chọn hành động có điểm q cao nhất
        state_t = torch.tensor(state, dtype=torch.float32,
                               device=self.device).unsqueeze(0)
        return self.online_net.get_action(state_t)

    def store_transition(self, state: np.ndarray, action: int,
                         reward: float, next_state: np.ndarray,
                         done: bool) -> None:
        # đẩy kinh nghiệm vào bộ nhớ per
        self.replay_buffer.add(state, action, reward, next_state, done)

    def learn(self) -> Dict[str, float]:
        # hàm cốt lõi thực thi thuật toán double dqn kết hợp per
        
        # nếu chưa gom đủ 1 batch thì chưa học
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0, "mean_q": 0.0, "grad_norm": 0.0}

        # bốc 1 lô dữ liệu từ bộ nhớ per kèm theo các trọng số is_weights
        (states, actions, rewards, next_states, dones,
         tree_indices, is_weights) = self.replay_buffer.sample(self.batch_size)

        # đẩy dữ liệu sang card đồ họa (gpu)
        states_t = torch.tensor(states, device=self.device)
        actions_t = torch.tensor(actions, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t = torch.tensor(dones, device=self.device).unsqueeze(1)
        weights_t = torch.tensor(is_weights, device=self.device).unsqueeze(1)

        # --- logic xử lý của double dqn ---
        # bước 1: dùng mạng online để chọn hành động tối ưu cho trạng thái kế tiếp.
        # (ai tự hỏi: "nếu ở bước tiếp theo, mình sẽ làm gì tốt nhất?")
        with torch.no_grad():
            next_q_online = self.online_net(next_states_t)
            best_actions = next_q_online.argmax(dim=1, keepdim=True)

            # bước 2: dùng mạng target để chấm điểm cho hành động vừa chọn ở bước 1.
            # (nhờ một giám khảo độc lập chấm điểm để tránh trường hợp tự biên tự diễn, dẫn đến ảo tưởng sức mạnh).
            next_q_target = self.target_net(next_states_t)
            next_q_value = next_q_target.gather(1, best_actions)

            # tính toán mục tiêu td (td target): y = r + gamma * q_target * (chưa game over)
            td_target = rewards_t + self.gamma * next_q_value * (1.0 - dones_t)

        # lấy điểm q hiện tại do mạng online dự đoán
        current_q = self.online_net(states_t).gather(1, actions_t)

        # tính mất mát huber nhân với trọng số hiệu chỉnh is_weights từ per
        element_wise_loss = self.loss_fn(current_q, td_target)
        loss = (element_wise_loss * weights_t).mean()

        # lan truyền ngược để tối ưu trọng số
        self.optimizer.zero_grad()
        loss.backward()

        # cắt xén gradient (clipping) để tránh việc cập nhật quá đà phá hỏng mạng
        grad_norm = nn.utils.clip_grad_norm_(
            self.online_net.parameters(), self.grad_clip
        )
        self.optimizer.step()

        # tính toán lại sai số td (td-error) thực tế và gửi ngược về cập nhật cây sumtree
        td_errors = (current_q - td_target).detach().cpu().numpy().flatten()
        self.replay_buffer.update_priorities(tree_indices, np.abs(td_errors))

        # đồng bộ hóa trọng số sang mạng target
        self.learn_step_counter += 1
        if self.tau < 1.0:
            self._soft_update()  # cập nhật từ từ từng chút một (polyak averaging)
        elif self.learn_step_counter % self.target_update_freq == 0:
            self._hard_update()  # chép đè toàn bộ sau mỗi n chu kỳ

        return {
            "loss": loss.item(),
            "mean_q": current_q.mean().item(),
            "grad_norm": float(grad_norm),
        }

    def _hard_update(self) -> None:
        # chép toàn bộ não bộ từ online sang target
        self.target_net.load_state_dict(self.online_net.state_dict())

    def _soft_update(self) -> None:
        # cập nhật trượt mượt mà: chỉ lấy một tỉ lệ tau rất nhỏ từ mạng online đắp sang target
        for target_param, online_param in zip(
            self.target_net.parameters(), self.online_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )

    def save_checkpoint(self, filepath: str) -> None:
        # lưu lại tiến độ học tập (trọng số, bộ tối ưu, số bước)
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
        # tải lại checkpoint đã lưu
        checkpoint = torch.load(filepath, map_location=self.device,
                                weights_only=True)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.learn_step_counter = checkpoint["learn_step_counter"]
        self._eps_step = checkpoint["eps_step"]
