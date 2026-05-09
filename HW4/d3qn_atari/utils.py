import gym
from gym.spaces import Box
import numpy as np
import cv2


class RestrictActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # For simplicity, map specific actions if needed.
        # By default we can leave this or map to necessary actions.


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs, {}


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84):
        super(WarpFrame, self).__init__(env)
        self.width = width
        self.height = height
        self.observation_space = Box(
            low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8
        )

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        return frame[:, :, None]


def make_env(env_name):
    # Depending on gym/gymnasium version
    try:
        env = gym.make(env_name, render_mode="rgb_array")
    except:
        env = gym.make(env_name)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if "NoFrameskip" in env_name:
        try:
            env = gym.wrappers.AtariPreprocessing(
                env, frame_skip=4, terminal_on_life_loss=True
            )
            env = gym.wrappers.FrameStack(env, num_stack=4)
        except AttributeError:
            env = gym.wrappers.AtariPreprocessing(env, frame_skip=4)
            env = gym.wrappers.FrameStack(env, num_stack=4)
    return env
