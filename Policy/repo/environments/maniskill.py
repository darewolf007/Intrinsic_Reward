import glob
import gymnasium as gym
import numpy as np
import os
import random
import torch
from gymnasium.spaces import Box
from gymnasium import spaces
from sapien.core import Pose
from transforms3d.euler import euler2quat
from collections import OrderedDict

QPOS_LOW = np.array(
    [0.0, np.pi * 2 / 8, 0, -np.pi * 5 / 8, 0, np.pi * 7 / 8, np.pi / 4, 0.04, 0.04]
)
QPOS_HIGH = np.array(
    [0.0, np.pi * 1 / 8, 0, -np.pi * 5 / 8, 0, np.pi * 6 / 8, np.pi / 4, 0.04, 0.04]
)
BASE_POSE = Pose([-0.615, 0, 0.05])
CUBE_HALF_SIZE = 0.02
xyz = np.hstack([0.0, 0.0, CUBE_HALF_SIZE])
quat = np.array([1.0, 0.0, 0.0, 0.0])
OBJ_INIT_POSE = Pose(xyz, quat)

# maniskill2的工具函数
def flatten_dict_space_keys(space: spaces.Dict, prefix="") -> spaces.Dict:
    """Flatten a dict of spaces by expanding its keys recursively."""
    out = OrderedDict()
    for k, v in space.spaces.items():
        if isinstance(v, spaces.Dict):
            out.update(flatten_dict_space_keys(v, prefix + k + "/").spaces)
        else:
            out[prefix + k] = v
    return spaces.Dict(out)

# maniskill2的工具函数
def flatten_state_dict(state_dict: dict) -> np.ndarray:
    """Flatten a dictionary containing states recursively.

    Args:
        state_dict: a dictionary containing scalars or 1-dim vectors.

    Raises:
        AssertionError: If a value of @state_dict is an ndarray with ndim > 2.

    Returns:
        np.ndarray: flattened states.

    Notes:
        The input is recommended to be ordered (e.g. OrderedDict).
        However, since python 3.7, dictionary order is guaranteed to be insertion order.
    """
    states = []
    for key, value in state_dict.items():
        if isinstance(value, dict):
            state = flatten_state_dict(value)
            if state.size == 0:
                state = None
        elif isinstance(value, (tuple, list)):
            state = None if len(value) == 0 else value
        elif isinstance(value, (bool, np.bool_, int, np.int32, np.int64)):
            # x = np.array(1) > 0 is np.bool_ instead of ndarray
            state = int(value)
        elif isinstance(value, (float, np.float32, np.float64)):
            state = np.float32(value)
        elif isinstance(value, np.ndarray):
            if value.ndim > 2:
                raise AssertionError(
                    "The dimension of {} should not be more than 2.".format(key)
                )
            state = value if value.size > 0 else None
        else:
            raise TypeError("Unsupported type: {}".format(type(value)))
        if state is not None:
            states.append(state)
    if len(states) == 0:
        return np.empty(0)
    else:
        return np.hstack(states)


class ManiSkillWrapper(gym.Wrapper):
    def __init__(self, env, pixel_obs):
        super().__init__(env)
        self.env = env
        self._pixel_obs = pixel_obs
        if pixel_obs:
            self._observation_space = Box(
                low=0, high=255, shape=(3, 64, 64), dtype=np.uint8
            )
        else:
            # States include robot proprioception (agent) and task information (extra)
            obs_space = self.env.observation_space
            state_spaces = []
            state_spaces.extend(
                flatten_dict_space_keys(obs_space["agent"]).spaces.values()
            )
            state_spaces.extend(
                flatten_dict_space_keys(obs_space["extra"]).spaces.values()
            )
            # Concatenate all the state spaces
            state_size = sum([space.shape[0] for space in state_spaces])
            self._observation_space = Box(-np.inf, np.inf, shape=(state_size,))

    def observation(self, observation):
        if self._pixel_obs:
            random_obs = self.env.background_randomization(observation, target_size=64)
            if isinstance(random_obs["left_camera"], torch.Tensor):
                random_obs["left_camera"] = random_obs["left_camera"].cpu().numpy()
            obs = random_obs["left_camera"].squeeze() #(1,128,128,3)
            obs = obs.transpose(2, 0, 1).copy()
            return obs
        else:
            # Concatenate all the states
            state = np.hstack(
                [
                    flatten_state_dict(observation["agent"]),
                    flatten_state_dict(observation["extra"]),
                ]
            )
            return state

    def reset(self):
        observation, info = self.env.reset(seed=7, options=None)  
        return self.observation(observation)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        # reward从tensor变成scalar
        reward = reward.item() if isinstance(reward, torch.Tensor) else reward
        return self.observation(obs), reward, done, truncated, info