# ==============================================================================
# HW4 — Prioritized Experience Replay (PER) with SumTree
# ==============================================================================
"""
Implements Prioritized Experience Replay using a Sum-Tree data structure.

Reference:
    Schaul, T., et al. (2016). "Prioritized Experience Replay." ICLR 2016.
"""

# file này hiện thực bộ nhớ trải nghiệm ưu tiên (per) dùng cấu trúc cây SumTree.
# mục đích: giúp ai lấy mẫu các ván game thông minh hơn (ván nào ai đoán sai nhiều thì độ ưu tiên cao, lấy ra học nhiều hơn),
# đồng thời nén mảng ảnh về dạng uint8 để tránh bị tràn ram trên colab.

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class Transition:
    # cấu trúc lưu trữ 1 bước game: trạng thái, hành động, phần thưởng, trạng thái kế tiếp, game over chưa
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class SumTree:
    # hiện thực cây nhị phân sum-tree. mảng cây có kích thước 2*capacity - 1.
    # nút gốc chứa tổng tất cả độ ưu tiên. các nút lá chứa độ ưu tiên của từng sample.
    # việc này giúp tìm kiếm/lấy mẫu theo tỉ lệ ưu tiên cực nhanh chỉ tốn o(log n).

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        # mảng lưu cây nhị phân (gốc ở index 0, lá bắt đầu từ capacity - 1)
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        # mảng thực tế chứa các đối tượng transition tương ứng với các lá
        self.data: list[Optional[Transition]] = [None] * capacity
        self.write_idx = 0
        self.size = 0

    def _propagate(self, idx: int, delta: float) -> None:
        # hàm đệ quy đi ngược từ lá lên gốc để cộng dồn lượng chênh lệch (delta) vào các nút cha
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def _retrieve(self, idx: int, value: float) -> int:
        # hàm đệ quy đi từ gốc xuống lá để tìm index của sample tương ứng với giá trị ngẫu nhiên value
        left = 2 * idx + 1
        right = 2 * idx + 2
        if left >= len(self.tree):
            return idx
        # nếu giá trị nhỏ hơn nút con trái, đi sang trái
        if value <= self.tree[left]:
            return self._retrieve(left, value)
        # ngược lại, trừ đi phần bên trái và đi sang phải
        else:
            return self._retrieve(right, value - self.tree[left])

    @property
    def total_priority(self) -> float:
        # tổng độ ưu tiên của toàn bộ buffer chính là giá trị ở nút gốc
        return float(self.tree[0])

    @property
    def max_priority(self) -> float:
        # tìm độ ưu tiên lớn nhất trong các lá hiện có để gán cho sample mới vào
        if self.size == 0:
            return 1.0
        leaf_start = self.capacity - 1
        return float(np.max(self.tree[leaf_start:leaf_start + self.size]))

    def add(self, priority: float, transition: Transition) -> None:
        # thêm sample mới vào mảng data và gán độ ưu tiên vào nút lá tương ứng trên cây
        tree_idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = transition
        self.update(tree_idx, priority)
        # trượt index quay vòng (xoay vòng ghi đè nếu đầy buffer)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_idx: int, priority: float) -> None:
        # cập nhật lại độ ưu tiên ở nút lá, sau đó lan truyền thay đổi lên gốc
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, delta)

    def get(self, value: float) -> Tuple[int, float, Transition]:
        # truy vấn sample dựa trên một con số ngẫu nhiên
        tree_idx = self._retrieve(0, value)
        data_idx = tree_idx - self.capacity + 1
        return tree_idx, self.tree[tree_idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    # bộ nhớ per bọc bên ngoài sumtree, quản lý việc lấy mẫu và tính toán trọng số is_weights
    # (importance sampling) để hiệu chỉnh sai số thiên lệch khi train.

    def __init__(self, capacity: int = 100000, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_end: float = 1.0,
                 beta_anneal_steps: int = 100000, epsilon: float = 1e-6) -> None:
        self.tree = SumTree(capacity)
        self.alpha = alpha  # mức độ phụ thuộc vào độ ưu tiên (0 là ngẫu nhiên, 1 là ưu tiên tuyệt đối)
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_anneal_steps = beta_anneal_steps
        self.epsilon = epsilon  # một số cực nhỏ cộng thêm vào để tránh sample có ưu tiên = 0
        self._step = 0

    @property
    def beta(self) -> float:
        # tăng dần hệ số beta từ start lên end theo thời gian train để khử hoàn toàn bias ở cuối
        fraction = min(1.0, self._step / max(1, self.beta_anneal_steps))
        return self.beta_start + fraction * (self.beta_end - self.beta_start)

    def __len__(self) -> int:
        return self.tree.size

    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool) -> None:
        # tối ưu hóa ram: ép mảng số thực float32 (0-1) về dạng số nguyên uint8 (0-255).
        # giải pháp này giúp giảm dung lượng ram xuống 4 lần, giải quyết dứt điểm lỗi tràn bộ nhớ (oom).
        state_u8 = (state * 255).clip(0, 255).astype(np.uint8)
        next_state_u8 = (next_state * 255).clip(0, 255).astype(np.uint8)
        transition = Transition(state=state_u8, action=action, reward=reward,
                                next_state=next_state_u8, done=done)
        # gán độ ưu tiên cao nhất cho ván mới để đảm bảo nó được bốc ra học ít nhất 1 lần
        max_p = self.tree.max_priority
        priority = max_p ** self.alpha if max_p > 0 else 1.0
        self.tree.add(priority, transition)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray,
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # lấy một lô ngẫu nhiên (batch) dựa trên phân phối xác suất ưu tiên
        self._step += 1
        states, actions, rewards, next_states, dones = [], [], [], [], []
        indices, priorities = [], []
        # chia dải tổng ưu tiên thành các đoạn nhỏ bằng nhau để bốc mẫu đều (stratified sampling)
        segment = self.tree.total_priority / batch_size

        for i in range(batch_size):
            value = np.random.uniform(segment * i, segment * (i + 1))
            tree_idx, priority, transition = self.tree.get(value)
            # bẫy logic an toàn nếu gặp ô trống
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

        # tính toán trọng số hiệu chỉnh is_weights
        priorities_arr = np.array(priorities, dtype=np.float64)
        sampling_probs = np.maximum(priorities_arr / self.tree.total_priority, 1e-10)
        is_weights = (len(self) * sampling_probs) ** (-self.beta)
        # chuẩn hóa trọng số theo max để giữ cho gradient ổn định, không bị bùng nổ
        is_weights /= is_weights.max()

        # khi xuất batch ra để đưa vào mạng nơ-ron, chia lại cho 255 để trả về tensor số thực [0-1]
        return (np.array(states, dtype=np.uint8).astype(np.float32) / 255.0,
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.uint8).astype(np.float32) / 255.0,
                np.array(dones, dtype=np.float32),
                np.array(indices, dtype=np.int64),
                np.array(is_weights, dtype=np.float32))

    def update_priorities(self, tree_indices: np.ndarray,
                          td_errors: np.ndarray) -> None:
        # hàm này được gọi sau khi train xong 1 lô. ta lấy sai số dự đoán (td-error)
        # để cập nhật lại độ ưu tiên ngược về cây sumtree.
        for idx, td_error in zip(tree_indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(int(idx), float(priority))
