# ==============================================================================
# HW4 — DeepMind Atari Environment Wrappers & Utilities
# ==============================================================================
"""
Standard Atari preprocessing wrappers following DeepMind's protocol.

Reference:
    Mnih, V., et al. (2015). "Human-level control through deep RL." Nature.

Wrappers applied (in order):
    1. NoopResetEnv     — Random no-ops at episode start
    2. MaxAndSkipEnv    — Frame skipping with max-pooling
    3. EpisodicLifeEnv  — Treat life loss as episode boundary (training only)
    4. FireResetEnv     — Press FIRE to start after reset
    5. WarpFrame        — Grayscale + resize to 84x84
    6. ClipRewardEnv    — Clip rewards to {-1, 0, +1}
    7. FrameStackEnv    — Stack last N frames as channels
"""

import gymnasium as gym
import numpy as np
import cv2
import torch
import random
from typing import Optional, Tuple, Any, SupportsFloat
from collections import deque

# xử lí ảnh: stack frame, resize, grayscale, clip reward, ...

# ==============================================================================
# Reproducibility
# ==============================================================================

def set_global_seeds(seed: int) -> None:
    """Set seeds for torch, numpy, random, and CUDA for reproducibility."""
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
    """Sample random no-ops on reset for stochastic initial states."""

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
    """Return max-pooled obs over last 2 of every `skip` frames."""

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
    """Make end-of-life == end-of-episode (training only)."""

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
    """Press FIRE at reset for environments that require it."""

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
    """Convert to grayscale and resize to 84x84."""

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
    """Clip rewards to {-1, 0, +1} using np.sign."""

    def reward(self, reward: SupportsFloat) -> float:
        return float(np.sign(float(reward)))


class FrameStackEnv(gym.Wrapper):
    """Stack the last `n_frames` as observation channels."""

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
    """
    Create a fully-wrapped Atari environment following DeepMind's protocol.
    
    Args:
        env_name:      Gymnasium Atari environment ID.
        seed:          Random seed for the environment.
        frame_stack:   Number of frames to stack.
        clip_rewards:  Whether to clip rewards to {-1, 0, +1}.
        episodic_life: Whether to treat life loss as episode end.
        render_mode:   Gymnasium render mode (None, 'human', 'rgb_array').
    
    Returns:
        Fully wrapped Gymnasium environment.
    """
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
    """Convert HWC uint8 obs to CHW float32 in [0, 1]."""
    obs = np.array(obs, dtype=np.float32) / 255.0
    return np.transpose(obs, (2, 0, 1))  # HWC -> CHW
