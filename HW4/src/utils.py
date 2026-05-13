# ==============================================================================
# HW4 — DeepMind Atari Environment Wrappers & Utilities
# ==============================================================================
# file này chứa các lớp bọc (wrappers) để tiền xử lý môi trường atari pong theo đúng chuẩn của deepmind.
# mục đích: biến game gốc (ảnh màu, chạy nhanh, điểm số ngẫu nhiên) thành dạng dữ liệu chuẩn hóa
# giúp mạng nơ-ron dễ đọc và học nhanh hơn.
# thứ tự áp dụng: noop -> skip frame -> episodic life -> fire reset -> warp (xám/resize) -> clip reward -> stack frame.

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
import numpy as np
import cv2
import torch
import random
from typing import Optional, Tuple, Any, SupportsFloat
from collections import deque


def set_global_seeds(seed: int) -> None:
    # cố định seed cho tất cả các thư viện để đảm bảo code chạy lại ra kết quả giống nhau (reproducibility)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================================
# Atari Wrappers
# ==============================================================================

class NoopResetEnv(gym.Wrapper):
    # lớp 1: khi mới vào game, đứng im (noop) một số bước ngẫu nhiên.
    # việc này giúp ván game bắt đầu ở các trạng thái đa dạng, tránh việc ai học vẹt một kịch bản cố định.

    def __init__(self, env: gym.Env, noop_max: int = 30) -> None:
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    # lớp 2: nhảy cóc khung hình (frame skipping). thay vì xử lý từng frame (game chạy 60fps rất thừa thãi),
    # ta gom 4 frame làm 1 bước, lấy max pixel của 2 frame cuối để tránh bị nhấp nháy hình ảnh.

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = np.zeros(
            (2,) + env.observation_space.shape, dtype=np.uint8
        )

    def step(self, action: int) -> Tuple[np.ndarray, SupportsFloat, bool, bool, dict]:
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += float(reward)
            if terminated or truncated:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info


class EpisodicLifeEnv(gym.Wrapper):
    # lớp 3: cứ mất 1 mạng (life) là coi như thua luôn ván đó (chỉ áp dụng lúc train).
    # giúp ai hiểu ngay hành động sai lầm dẫn đến mất mạng để né tránh kịp thời.

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action: int) -> Tuple[np.ndarray, SupportsFloat, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, dict]:
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class FireResetEnv(gym.Wrapper):
    # lớp 4: tự động bấm nút bắn (fire) lúc mới vào để game bắt đầu thả bóng.

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, dict]:
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class WarpFrame(gym.ObservationWrapper):
    # lớp 5: chuyển ảnh màu sang đen trắng (grayscale) và bóp kích thước về 84x84 pixel.
    # việc này giúp giảm triệt để khối lượng tính toán cho cnn.

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84) -> None:
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8
        )

    def observation(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height),
                           interpolation=cv2.INTER_AREA)
        return frame[:, :, np.newaxis]


class ClipRewardEnv(gym.RewardWrapper):
    # lớp 6: kìm hãm phần thưởng về dải -1 (thua), 0 (hòa), 1 (thắng).
    # giúp mạng ổn định, không bị sốc khi chuyển sang các game có thang điểm số lớn.

    def reward(self, reward: SupportsFloat) -> float:
        return float(np.sign(float(reward)))


class FrameStackEnv(gym.Wrapper):
    # lớp 7: chập 4 khung hình liên tiếp nhau thành 1 mảng đa kênh (channel).
    # nếu chỉ nhìn 1 ảnh tĩnh, ai không thể biết quả bóng đang bay sang trái hay phải.
    # chập 4 ảnh giúp mạng cnn nhận thức được hướng đi và vận tốc của vệt bóng.

    def __init__(self, env: gym.Env, n_frames: int = 4) -> None:
        super().__init__(env)
        self.n_frames = n_frames
        self.frames: deque = deque([], maxlen=n_frames)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(shp[0], shp[1], shp[2] * n_frames),
            dtype=np.uint8,
        )

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action: int) -> Tuple[np.ndarray, SupportsFloat, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        assert len(self.frames) == self.n_frames
        return np.concatenate(list(self.frames), axis=2)


# ==============================================================================
# Environment Factory
# ==============================================================================

def make_atari_env(
    env_name: str,
    seed: int = 42,
    frame_stack: int = 4,
    clip_rewards: bool = True,
    episodic_life: bool = True,
    render_mode: Optional[str] = None,
) -> gym.Env:
    # hàm xưởng (factory function): gộp tuần tự 7 lớp wrapper ở trên để bọc lại môi trường gốc
    env = gym.make(env_name, render_mode=render_mode)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if episodic_life:
        env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    env = FrameStackEnv(env, n_frames=frame_stack)
    env.reset(seed=seed)
    return env


def preprocess_observation(obs: np.ndarray) -> np.ndarray:
    # hàm phụ trợ: đổi vị trí các chiều từ dạng (chiều cao, chiều rộng, số kênh) của ảnh
    # sang dạng chuẩn (số kênh, chiều cao, chiều rộng) của pytorch, đồng thời chia 255.
    obs = np.array(obs, dtype=np.float32) / 255.0
    return np.transpose(obs, (2, 0, 1))  # HWC -> CHW
