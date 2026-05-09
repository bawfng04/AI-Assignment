# ==============================================================================
# HW4 — TensorBoard Experiment Logger
# ==============================================================================
"""Lightweight wrapper around TensorBoard's SummaryWriter for RL metrics."""

from torch.utils.tensorboard import SummaryWriter
from typing import Optional


class RLLogger:
    """
    TensorBoard logger for tracking RL training metrics.
    
    Logs: episode rewards, loss, epsilon, learning rate,
    Q-value statistics, and evaluation results.
    
    Args:
        log_dir: Directory for TensorBoard event files.
    """

    def __init__(self, log_dir: str = "runs") -> None:
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_episode(self, episode: int, reward: float, length: int,
                    epsilon: float) -> None:
        """Log end-of-episode metrics."""
        self.writer.add_scalar("episode/reward", reward, episode)
        self.writer.add_scalar("episode/length", length, episode)
        self.writer.add_scalar("episode/epsilon", epsilon, episode)

    def log_training_step(self, step: int, loss: float,
                          mean_q: float, grad_norm: float) -> None:
        """Log per-training-step metrics."""
        self.writer.add_scalar("train/loss", loss, step)
        self.writer.add_scalar("train/mean_q_value", mean_q, step)
        self.writer.add_scalar("train/grad_norm", grad_norm, step)

    def log_evaluation(self, step: int, mean_reward: float,
                       std_reward: float) -> None:
        """Log evaluation results."""
        self.writer.add_scalar("eval/mean_reward", mean_reward, step)
        self.writer.add_scalar("eval/std_reward", std_reward, step)

    def log_buffer_stats(self, step: int, buffer_size: int,
                         beta: float) -> None:
        """Log replay buffer statistics."""
        self.writer.add_scalar("buffer/size", buffer_size, step)
        self.writer.add_scalar("buffer/per_beta", beta, step)

    def close(self) -> None:
        """Flush and close the writer."""
        self.writer.flush()
        self.writer.close()
