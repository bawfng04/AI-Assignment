# Báo Cáo: Dueling Double DQN (D3QN) cho Trò Chơi Atari

> **Học phần:** Trí Tuệ Nhân Tạo — Bài Tập Lớn 4  
> **Chủ đề:** Reinforcement Learning với Deep Q-Networks nâng cao  
> **Thuật toán:** Dueling Double DQN + Prioritized Experience Replay

---

## 1. Giới Thiệu & Động Lực (Motivation)

### 1.1 Tại sao Reinforcement Learning cho Atari?

Trò chơi Atari đã trở thành benchmark tiêu chuẩn cho Reinforcement Learning (RL) kể từ khi DeepMind chứng minh rằng một agent duy nhất có thể đạt trình độ siêu nhân trong nhiều game chỉ bằng cách học trực tiếp từ pixel đầu vào (Mnih et al., 2015). Đây là bài toán lý tưởng vì:

- **High-dimensional input:** Quan sát là hình ảnh 210×160×3 pixel
- **Sparse/delayed rewards:** Phần thưởng không xuất hiện ngay lập tức
- **Diverse strategies:** Mỗi game yêu cầu chiến lược hoàn toàn khác nhau

### 1.2 Từ Q-Learning đến Deep Q-Network (DQN)

**Tabular Q-Learning** lưu trữ $Q(s, a)$ trong bảng tra cứu, không khả thi cho không gian trạng thái liên tục hoặc rất lớn (Atari có $\sim 10^{30000}$ trạng thái pixel khác nhau). **DQN** (Mnih et al., 2015) thay thế bảng Q bằng mạng neural sâu:

$$Q(s, a; \theta) \approx Q^*(s, a)$$

Hai cải tiến quan trọng của DQN so với Q-Learning truyền thống:
1. **Experience Replay:** Phá vỡ tương quan giữa các mẫu liên tiếp
2. **Target Network:** Ổn định mục tiêu huấn luyện bằng mạng $\theta^-$ cập nhật chậm

### 1.3 Vấn đề Overestimation Bias — Tại sao cần Double DQN?

**Vanilla DQN** sử dụng cùng một mạng để **chọn** và **đánh giá** hành động tốt nhất:

$$y^{DQN} = r + \gamma \max_{a'} Q(s', a'; \theta^-)$$

Toán tử $\max$ gây ra **overestimation bias** (Thrun & Schwartz, 1993): do nhiễu trong ước lượng Q, $\max$ luôn có xu hướng chọn giá trị bị ước lượng cao hơn thực tế. Sai lệch này tích lũy theo thời gian và dẫn đến chính sách tối ưu phụ (suboptimal policy).

**Double DQN** (van Hasselt et al., 2016) khắc phục bằng cách tách biệt hai bước:
- **Online network $\theta$** chọn hành động: $a^* = \arg\max_{a'} Q(s', a'; \theta)$
- **Target network $\theta^-$** đánh giá hành động đó: $Q(s', a^*; \theta^-)$

### 1.4 Dueling Architecture — Ước lượng giá trị trạng thái tốt hơn

Trong nhiều trạng thái, giá trị hành động cụ thể không quan trọng — điều quan trọng là trạng thái đó tốt hay xấu. **Dueling DQN** (Wang et al., 2016) tách Q-value thành hai thành phần:
- **Value function $V(s)$:** Trạng thái này có giá trị bao nhiêu?
- **Advantage function $A(s, a)$:** Hành động $a$ tốt hơn trung bình bao nhiêu?

Điều này cho phép mạng học giá trị trạng thái mà không cần phải đánh giá mọi hành động — đặc biệt hữu ích khi nhiều hành động không ảnh hưởng đến kết quả.

---

## 2. Cơ Sở Toán Học (Mathematical Background)

### 2.1 Markov Decision Process (MDP)

Bài toán RL được mô hình hóa như một MDP $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ với:
- $\mathcal{S}$: Không gian trạng thái
- $\mathcal{A}$: Không gian hành động  
- $P(s'|s, a)$: Xác suất chuyển trạng thái
- $R(s, a)$: Hàm phần thưởng
- $\gamma \in [0, 1)$: Hệ số chiết khấu

### 2.2 Phương trình Bellman (Bellman Equation)

Giá trị tối ưu thỏa mãn phương trình Bellman tối ưu:

$$Q^*(s, a) = \mathbb{E}_{s'} \left[ r + \gamma \max_{a'} Q^*(s', a') \mid s, a \right]$$

Trong DQN, phương trình này được xấp xỉ bằng mạng neural $Q(s, a; \theta)$ và tối ưu hóa qua gradient descent.

### 2.3 Hàm Mất Mát của Double DQN (Loss Function)

Mục tiêu cập nhật (TD target) của **Double DQN**:

$$y^{DDQN} = r + \gamma Q\left(s', \underbrace{\arg\max_{a'} Q(s', a'; \theta)}_{\text{online network chọn}};\ \theta^-\right)$$

Hàm mất mát (Huber Loss cho ổn định):

$$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( y^{DDQN} - Q(s, a; \theta) \right)^2 \right]$$

Với Prioritized Experience Replay, loss được nhân với trọng số importance-sampling $w_i$:

$$L(\theta) = \frac{1}{N} \sum_{i=1}^{N} w_i \cdot \left( y_i^{DDQN} - Q(s_i, a_i; \theta) \right)^2$$

### 2.4 Kiến Trúc Dueling (Dueling Architecture)

Giá trị Q được phân tách thành **Value** và **Advantage**:

$$Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a'; \theta, \alpha) \right)$$

Trong đó:
- $V(s; \theta, \beta)$: **State value** — giá trị nội tại của trạng thái $s$
- $A(s, a; \theta, \alpha)$: **Advantage** — lợi thế tương đối của hành động $a$
- Phép trừ trung bình $\frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a')$ đảm bảo tính **identifiability**: cho một giá trị $Q(s,a)$, ta có thể xác định duy nhất $V(s)$ và $A(s,a)$

### 2.5 Prioritized Experience Replay (PER)

Xác suất lấy mẫu transition $i$ tỷ lệ với TD-error:

$$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$$

trong đó $p_i = |\delta_i| + \epsilon$ là priority, $\alpha$ điều khiển mức độ ưu tiên (0 = uniform, 1 = full priority).

Trọng số importance-sampling để sửa bias:

$$w_i = \left( \frac{1}{N \cdot P(i)} \right)^\beta$$

$\beta$ được annealing từ $\beta_0$ đến $1.0$ để đảm bảo correction hoàn toàn vào cuối training.

---

## 3. Kiến Trúc Mạng Neural (Network Architecture)

### 3.1 CNN Feature Extractor (Shared Backbone)

Theo chuẩn DeepMind (Mnih et al., 2015):

| Layer | Type | Filters | Kernel | Stride | Output |
|-------|------|---------|--------|--------|--------|
| Input | — | — | — | — | 4 × 84 × 84 |
| Conv1 | Conv2D + ReLU | 32 | 8×8 | 4 | 32 × 20 × 20 |
| Conv2 | Conv2D + ReLU | 64 | 4×4 | 2 | 64 × 9 × 9 |
| Conv3 | Conv2D + ReLU | 64 | 3×3 | 1 | 64 × 7 × 7 |
| Flatten | — | — | — | — | 3136 |

### 3.2 Dueling Streams

Sau khi flatten, features được chia thành hai luồng song song:

```
                  ┌─────────────────┐
                  │  CNN Backbone    │
                  │  (shared)        │
                  └────────┬────────┘
                           │
                    ┌──────┴──────┐
                    │             │
             ┌──────┴──────┐  ┌──┴──────────┐
             │ Value Stream│  │ Advantage   │
             │ FC(3136→512)│  │ Stream      │
             │ ReLU        │  │ FC(3136→512)│
             │ FC(512→1)   │  │ ReLU        │
             │             │  │ FC(512→|A|) │
             └──────┬──────┘  └──────┬──────┘
                    │                │
                    └────────┬───────┘
                             │
                    Q = V + (A - mean(A))
```

---

## 4. Triển Khai (Implementation Details)

### 4.1 Environment Wrappers

*[Mô tả chi tiết các wrapper: NoopReset, MaxAndSkip, EpisodicLife, FireReset, WarpFrame, ClipReward, FrameStack — đã implement trong `src/utils.py`]*

### 4.2 Thuật Toán Training

```
Algorithm: D3QN với Prioritized Experience Replay
────────────────────────────────────────────────
Input: Environment E, hyperparameters
Initialize: Online network Q(θ), Target network Q(θ⁻) ← Q(θ)
Initialize: PER buffer D with SumTree

For step t = 1, 2, ..., T:
    1. Observe state sₜ
    2. Select action aₜ = ε-greedy(Q(sₜ; θ))
    3. Execute aₜ, observe rₜ, sₜ₊₁, done
    4. Store (sₜ, aₜ, rₜ, sₜ₊₁, done) in D with max priority
    
    If t > learning_starts and t mod train_freq == 0:
        5. Sample batch {(sⱼ, aⱼ, rⱼ, s'ⱼ, dⱼ, wⱼ)} ~ PER(D)
        6. Compute targets:
           a* = argmax_a' Q(s'ⱼ, a'; θ)     [online selects]
           yⱼ = rⱼ + γ Q(s'ⱼ, a*; θ⁻)       [target evaluates]
        7. Compute weighted loss: L = Σ wⱼ · Huber(yⱼ, Q(sⱼ, aⱼ; θ))
        8. Update θ via Adam with gradient clipping
        9. Update PER priorities: pⱼ = |δⱼ| + ε
    
    If t mod target_update_freq == 0:
        10. θ⁻ ← θ  (hard update)
    
    Anneal ε linearly: ε₀ → ε_min over decay_steps
    Anneal β linearly: β₀ → 1.0 over anneal_steps
```

---

## 5. Kết Quả Thực Nghiệm (Experimental Results)

### 5.1 Training Curves

*[Chèn TensorBoard screenshots:]*
- Episode Reward vs. Training Steps
- Loss vs. Training Steps  
- Epsilon Decay Schedule
- Mean Q-value over Training

### 5.2 So Sánh Hiệu Suất

| Thuật Toán | Avg. Reward (Pong) | Avg. Reward (Breakout) |
|---|---|---|
| Random Agent | -20.0 | ~1.0 |
| Vanilla DQN | — | — |
| **D3QN + PER (Ours)** | **—** | **—** |

*[Điền kết quả sau khi training]*

### 5.3 Gameplay Recording

*[Chèn GIF/MP4 gameplay từ `eval.py --record`]*

---

## 6. Phân Tích & Thảo Luận (Analysis)

### 6.1 Ảnh hưởng của Dueling Architecture

*[Phân tích sự khác biệt giữa V(s) và A(s,a) — attention maps cho thấy Value stream tập trung vào tổng thể, Advantage stream tập trung vào vùng liên quan đến hành động]*

### 6.2 Ảnh hưởng của PER

*[So sánh PER vs Uniform replay: PER hội tụ nhanh hơn nhờ tập trung vào transitions có TD-error cao]*

### 6.3 Overestimation Bias — Double DQN

*[Biểu đồ Q-value estimates vs actual returns cho thấy Double DQN giảm overestimation]*

---

## 7. Kết Luận (Conclusion)

*[Tóm tắt đóng góp: Implement D3QN + PER, chứng minh ưu điểm qua Atari benchmarks, phân tích từng thành phần]*

---

## Tài Liệu Tham Khảo (References)

1. Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). "Human-level control through deep reinforcement learning." *Nature*, 518(7540), 529–533.

2. van Hasselt, H., Guez, A., & Silver, D. (2016). "Deep Reinforcement Learning with Double Q-learning." *Proceedings of AAAI*.

3. Wang, Z., Schaul, T., Hessel, M., et al. (2016). "Dueling Network Architectures for Deep Reinforcement Learning." *Proceedings of ICML*.

4. Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). "Prioritized Experience Replay." *Proceedings of ICLR*.

5. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

6. Thrun, S., & Schwartz, A. (1993). "Issues in using function approximation for reinforcement learning." *Proceedings of the Fourth Connectionist Models Summer School*.
