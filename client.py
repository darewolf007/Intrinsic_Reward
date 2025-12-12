import sys
from pathlib import Path
import cv2
import time
import threading
import requests
import io
import signal
import pandas as pd
import numpy as np
import contextlib
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Union
from websocket import websocket_client_policy
from PIL import Image
from termcolor import colored

@dataclass
class FR3Config:
    # --- robot setting ---
    robot_id: str = "fr3"
    robot_ip: str = "172.16.0.2"
    load_gripper: bool = True
    relative_dynamics_factor: float = 0.05
    buffer_size: int = 10
    home: bool = True                 

    # --- camera setting ---
    scene_camera_id: Optional[str] = "938422072347" # left camera
    right_camera_id: Optional[str] = "233522073398" # front camera
    wrist_camera_id: Optional[str] = "112322074840" # wrist camera
    fps: int = 15
    width: int = 1280
    height: int = 720
    camera_buffer: int = 5

    # --- control mode ---
    action_mode: str = "POSITION_ABSOLUTE"  # ["POSITION_DELTA", "JOINT_DELTA", "POSITION_ABSOLUTE", "JOINT_ABSOLUTE"， "POSITION_BASE_DELTA"]
    img_update_rate: int = 15            
    asynchronous: bool = False           # move asynchronous
    action_chunk: int = 5                # move step num


@dataclass
class TaskConfig:
    task_prompt: str = "pick up the cube"
    algorithm: str = "repo"
    is_online: bool = True
    server_host: str = "10.184.17.177"
    server_port: int = 8059
    image_size: int = 64
    def __repr__(self):
        return f"<TaskConfig {self.algorithm} | online={self.is_online}>"


def quaternion_raw_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return np.stack((ow, ox, oy, oz), axis=-1)

def standardize_quaternion(q: np.ndarray) -> np.ndarray:
    return np.where(q[..., 0:1] < 0, -q, q)

def quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)

class RobotClient:
    _action_registry: dict[str, callable] = {}

    def __init__(self, cfg):
        self.cfg = cfg
        self.image_size = cfg.image_size
        self.policy_client = websocket_client_policy.WebsocketClientPolicy(cfg.server_host, cfg.server_port)

    def get_action(self, observation):
        # obs = self.pre_input(observation)
        infer_actions = self.policy_client.infer(observation)["actions"]
        # actions = self.post_output(infer_actions, observation["end_effector"])
        # print(actions)
        return infer_actions

    def center_crop_and_resize(self, img, target_size):
        H, W = img.shape[:2]
        short = min(H, W)
        y1 = (H - short) // 2
        x1 = (W - short) // 2
        cropped = img[y1:y1 + short, x1:x1 + short]
        if target_size is not None:
            resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)
            return resized
        return cropped
    
    def pre_input(self, observation):
        obs = self.center_crop_and_resize(observation['scene_image'], self.image_size)
        cv2.imshow("input image", obs[:,:,::-1])
        cv2.waitKey(1)
        obs = np.transpose(obs, (2, 0, 1))
        return obs
    
    def post_output(self, action, state):
        action = self.process_action(action, state)
        return action
    
    def normalize_gripper_action(self, action: np.ndarray, binarize: bool = True) -> np.ndarray:
        action = np.array(action)
        normalized_action = action.copy()
        if binarize:
            normalized_action[-1] = 1.0 if normalized_action[-1] >= 0.5 else 0.0
        return normalized_action
    
    def _clip_and_scale_action(self, action, low, high):
        action = np.clip(action, -1, 1)
        return 0.5 * (high + low) + 0.5 * (high - low) * action

    def clip_and_scale_action(self, action):
        pos_action = self._clip_and_scale_action(
            action[:3], 
            np.array([-0.1, -0.1, -0.1]), 
            np.array([0.1, 0.1, 0.1])
        )
        rot_action = action[3:].copy()
        rot_norm = np.linalg.norm(rot_action)
        if rot_norm > 1:
            rot_action = rot_action * (1.0 / rot_norm)
        
        rot_action = rot_action * -0.1
        return pos_action, rot_action

    def process_action(self, action, tcp_state): 
        action = np.array(action)
        pose_part = action[:6]
        pos_action, rot_action = self.clip_and_scale_action(pose_part)
        rotation = R.from_euler("xyz", rot_action, degrees=False).as_quat()[[3, 0, 1, 2]]
        target_q = quaternion_multiply(rotation, tcp_state[3:])
        target_p = pos_action + tcp_state[:3]
        gripper_part = np.array([0])
        return np.concatenate([target_p, target_q, gripper_part])

    
if __name__ == "__main__":
    import cv2
    import imageio
    import numpy as np
    from pathlib import Path
    from Policy.repo.run.load_repo import AgentLoader
    import torch
    class VideoRecorder:
        def __init__(self, root_dir, render_size=256, fps=20):
            if root_dir is not None:
                self.save_dir = Path(root_dir)
            else:
                self.save_dir = None

            self.render_size = render_size
            self.fps = fps
            self.frames = []

        def init(self, obs, enabled=True):
            self.frames = []
            self.enabled = True
            self.record(obs)

        def record(self, obs):
            self.frames.append(np.transpose(obs, (1, 2, 0)))

        def save(self, file_name):
            if self.enabled and len(self.frames) > 0:
                path = self.save_dir / file_name
                video_data = np.array(self.frames)
                if video_data.dtype != np.uint8:
                    if video_data.max() > 1.0:
                        video_data = video_data.astype(np.uint8)
                    else:
                        video_data = (video_data * 255).astype(np.uint8)
                imageio.mimsave(str(path), video_data, fps=self.fps, codec="libx264")
                print(f"Video saved to {path}")
    agent = AgentLoader()
    video = VideoRecorder(root_dir="/data2/user/sunhaowen/hw_mine/Intrinsic_Reward")
    agent.load(algo="repo", task="maniskill-PickCube", seed=3407, ckpt_dir = "/data2/user/sunhaowen/hw_mine/Intrinsic_Reward/Policy/repo/logdir/repo/maniskill-PickCube/test2_1280/3407")
    obs = agent.eval_env.reset()
    video.init(obs)
    done = False
    truncated = False
    episode_reward = 0
    episode_success = 0
    action_list = []
    client = RobotClient(TaskConfig())
    with torch.no_grad():
        while not (done or truncated):
            action = client.get_action(obs)
            action_list.append(action)
            next_obs, reward, terminated, truncated, info = agent.eval_env.step(action)
            done = terminated or truncated
            video.record(next_obs)
            obs = next_obs
            episode_reward += reward
            episode_success += info.get("success", 0)
    np.save("action.npy", np.array(action_list))
    video.save("test.mp4")