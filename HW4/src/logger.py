# ==============================================================================
# HW4 — TensorBoard Experiment Logger
# ==============================================================================
# file này là lớp tiện ích hỗ trợ ghi nhận số liệu (log) trong quá trình huấn luyện bằng tensorboard.
# mục đích: theo dõi sát sao tiến độ học tập (loss, phần thưởng qua từng tập, epsilon) để sau này
# có dữ liệu xuất ra bảng phân tích trên giao diện streamlit.

from torch.utils.tensorboard import SummaryWriter
from typing import Optional


class RLLogger:
    # quản lý việc ghi log các chỉ số vào thư mục được chỉ định

    def __init__(self, log_dir: str = "runs") -> None:
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_episode(self, episode: int, reward: float, length: int,
                    epsilon: float) -> None:
        # ghi nhận thông số tổng kết khi kết thúc một ván game (tập)
        self.writer.add_scalar("episode/reward", reward, episode)
        self.writer.add_scalar("episode/length", length, episode)
        self.writer.add_scalar("episode/epsilon", epsilon, episode)

    def log_training_step(self, step: int, loss: float,
                          mean_q: float, grad_norm: float) -> None:
        # ghi nhận các thông số kỹ thuật nội bộ sau mỗi bước cập nhật gradient
        self.writer.add_scalar("train/loss", loss, step)
        self.writer.add_scalar("train/mean_q_value", mean_q, step)
        self.writer.add_scalar("train/grad_norm", grad_norm, step)

    def log_evaluation(self, step: int, mean_reward: float,
                       std_reward: float) -> None:
        # ghi nhận kết quả điểm số trung bình khi chạy chế độ chấm điểm (eval mode)
        self.writer.add_scalar("eval/mean_reward", mean_reward, step)
        self.writer.add_scalar("eval/std_reward", std_reward, step)

    def log_buffer_stats(self, step: int, buffer_size: int,
                         beta: float) -> None:
        # ghi log giám sát trạng thái đầy của bộ nhớ và hệ số beta của per
        self.writer.add_scalar("buffer/size", buffer_size, step)
        self.writer.add_scalar("buffer/per_beta", beta, step)

    def close(self) -> None:
        # đẩy hết dữ liệu đệm ra file đĩa và đóng bộ ghi
        self.writer.flush()
        self.writer.close()
