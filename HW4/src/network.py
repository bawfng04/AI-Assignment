# ==============================================================================
# HW4 — Dueling DQN Network Architecture
# ==============================================================================
"""
Implements the Dueling Network Architecture for Deep Reinforcement Learning.

Reference:
    Wang, Z., et al. (2016). "Dueling Network Architectures for Deep
    Reinforcement Learning." ICML 2016.
    
# file này hiện thực kiến trúc dueling dqn (dueling deep q-network).
# ý tưởng cốt lõi: thay vì tính thẳng q-value cho từng hành động, ta tách ra làm 2 luồng:
# 1. giá trị trạng thái v(s): cho biết hoàn cảnh hiện tại tốt hay xấu (ví dụ bóng đang bay về phía mình).
# 2. lợi thế hành động a(s,a): cho biết hành động này tốt hơn các hành động khác bao nhiêu.
# việc tách này giúp mạng học nhanh hơn vì nhiều lúc chỉ cần biết trạng thái an toàn là đủ, không cần thử hết các nút bấm.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class DuelingDQN(nn.Module):
    # mạng dueling dqn dùng cnn để đọc ảnh pixel từ atari pong.

    def __init__(self, input_shape: Tuple[int, ...], n_actions: int) -> None:
        super().__init__()
        
        self.input_shape = input_shape
        self.n_actions = n_actions
        
        # dùng 3 lớp tích chập (cnn) giống hệt paper gốc của deepmind để trích xuất đặc trưng từ 4 frame ảnh gộp
        self.features = nn.Sequential(
            # lớp 1: 32 bộ lọc, cửa sổ 8x8, bước trượt 4
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            # lớp 2: 64 bộ lọc, cửa sổ 4x4, bước trượt 2
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            # lớp 3: 64 bộ lọc, cửa sổ 3x3, bước trượt 1
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        
        # tự động tính toán kích thước mảng 1d sau khi duỗi thẳng ảnh ra từ cnn
        self._feature_size = self._get_conv_output_size(input_shape)
        
        # luồng 1: tính giá trị v(s) (trả về 1 con số duy nhất)
        self.value_stream = nn.Sequential(
            nn.Linear(self._feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        )
        
        # luồng 2: tính lợi thế a(s,a) (trả về mảng các con số tương ứng với số lượng hành động)
        self.advantage_stream = nn.Sequential(
            nn.Linear(self._feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_actions),
        )
        
        # khởi tạo trọng số kaiming (he) để tránh bị triệt tiêu gradient khi dùng hàm kích hoạt relu
        self._initialize_weights()

    def _get_conv_output_size(self, shape: Tuple[int, ...]) -> int:
        # hàm phụ trợ: đẩy thử 1 tensor toàn số 0 qua cnn để đo kích thước mảng đầu ra
        with torch.no_grad():
            dummy = torch.zeros(1, *shape)
            output = self.features(dummy)
            return int(np.prod(output.shape[1:]))

    def _initialize_weights(self) -> None:
        # khởi tạo trọng số ngẫu nhiên chuẩn kaiming uniform
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # luồng chạy chính của mạng:
        # đầu vào x là tensor mảng ảnh (đã chia 255 để nằm trong dải 0-1)
        
        # đi qua cnn để lấy đặc trưng
        features = self.features(x)
        # duỗi thẳng (flatten) về mảng 1d
        features = features.reshape(features.size(0), -1)
        
        # tách đi qua 2 luồng song song
        value: torch.Tensor = self.value_stream(features)          # (batch, 1)
        advantage: torch.Tensor = self.advantage_stream(features)  # (batch, n_actions)
        
        # gộp lại theo công thức dueling dqn: q = v + (a - trung bình các a)
        # việc trừ đi trung bình (mean subtraction) nhằm ép mạng không được ăn gian (đảm bảo tính duy nhất,
        # không thể dịch chuyển v và a tùy tiện mà vẫn ra cùng 1 giá trị q).
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values

    def get_action(self, state: torch.Tensor) -> int:
        # chọn hành động tốt nhất (greedy) bằng cách lấy vị trí có q-value cao nhất
        with torch.no_grad():
            q_values = self.forward(state)
            return int(q_values.argmax(dim=1).item())
