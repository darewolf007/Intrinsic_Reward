from typing import Any, Dict, Union
import cv2
import numpy as np
import sapien
import torch
import os
import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Fetch, Panda, XArm6Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from .utils import center_crop_and_resize
from .table_scene import TableSceneBuilder
from sapien.render import RenderBodyComponent

PICK_CUBE_DOC_STRING = """**Task Description:**
A simple task where the objective is to grasp a red cube with the {robot_id} robot and move it to a target goal position. This is also the *baseline* task to test whether a robot with manipulation
capabilities can be simulated and trained properly. Hence there is extra code for some robots to set them up properly in this environment as well as the table scene builder.

**Randomizations:**
- the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
- the cube's z-axis rotation is randomized to a random angle
- the target goal position (marked by a green sphere) of the cube has its xy position randomized in the region [0.1, 0.1] x [-0.1, -0.1] and z randomized in [0, 0.3]

**Success Conditions:**
- the cube position is within `goal_thresh` (default 0.025m) euclidean distance of the goal position
- the robot is static (q velocity < 0.2)
"""
RANDOM_KEYS = [
    "camera_random",
    "light_random",
    "texture_random",
    "background_random",
]

@register_env("PickCube-v2", max_episode_steps=50)
class PickCubeEnv(BaseEnv):
    """
    **Task Description:**
    A simple task where the objective is to grasp a red cube and move it to a target goal position.

    **Randomizations:**
    - the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
    - the cube's z-axis rotation is randomized to a random angle
    - the target goal position (marked by a green sphere) of the cube has its xy position randomized in the region [0.1, 0.1] x [-0.1, -0.1] and z randomized in [0, 0.3]

    **Success Conditions:**
    - the cube position is within `goal_thresh` (default 0.025m) euclidean distance of the goal position
    - the robot is static (q velocity < 0.2)
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = [
        "panda",
        "fetch",
        "xarm6_robotiq",
    ]
    agent: Union[Panda, Fetch, XArm6Robotiq]
    cube_half_size = 0.02
    goal_thresh = 0.025

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.random_cfg = {key: kwargs.pop(key, False) for key in RANDOM_KEYS}
        print("Randomization config:", self.random_cfg)
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.scene_background_image = None
        if self.random_cfg["background_random"]:
            background_name = "background"
        else:
            background_name = "background1"
        self.scene_background_path = sorted(os.listdir(os.path.join(os.path.dirname(__file__), background_name)), key=lambda x: x.lower())
        front_files = [f for f in self.scene_background_path if f.lower().startswith("main")]
        eye_files = [f for f in self.scene_background_path if f.lower().startswith("eye")]
        left_files = [f for f in self.scene_background_path if f.lower().startswith("left")]
        self.scene_background_list = [cv2.cvtColor(cv2.imread(os.path.join(os.path.dirname(__file__), background_name, file_name)), cv2.COLOR_BGR2RGB) for file_name in front_files]
        # self.tensor_scene_background_list = [torch.from_numpy(image).float() for image in self.scene_background_list]
        self.eye_background_list = [cv2.cvtColor(cv2.imread(os.path.join(os.path.dirname(__file__), background_name, file_name)), cv2.COLOR_BGR2RGB) for file_name in eye_files]
        # self.tensor_eye_background_list = [torch.from_numpy(image).float() for image in self.eye_background_list]
        self.left_background_list = [cv2.cvtColor(cv2.imread(os.path.join(os.path.dirname(__file__), background_name, file_name)), cv2.COLOR_BGR2RGB) for file_name in left_files]
        # self.tensor_left_background_list = [torch.from_numpy(image).float() for image in self.left_background_list]
        self.left_base_pose = sapien_utils.look_at(eye=[-0.38155, 1.2102138255883726, 0.4851141636379064], target=[-0.35978, 0.13906, 0.30676])
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        calib_p = torch.tensor(
            [0.553,
            -0.027871877550427117,
            0.6863282894133899],
            dtype=torch.float32,
            device=self.device,
        )
        calib_q = torch.tensor(
            [0.03128285, -0.37759845, 0.00461725, 0.92542925],
            dtype=torch.float32,
            device=self.device,
        )
        calib_p = calib_p.unsqueeze(0).repeat(self.num_envs, 1)  # (num_envs, 3)
        calib_q = calib_q.unsqueeze(0).repeat(self.num_envs, 1)  # (num_envs, 4)
        pose = Pose.create_from_pq(
            p=calib_p,
            q=calib_q,
        )
        intrinsic = np.array([[905.0100708007812, 0.0, 653.8479614257812], [0.0, 905.0863037109375, 373.97210693359375], [0.0, 0.0, 1.0]])
        intrinsic_128 = np.array([[90.501, 0.0,   65.38],[0.0,    160.9, 66.23],[0.0,    0.0,   1.0]])
        base_pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        if self.random_cfg["camera_random"]:
            return [CameraConfig("left_camera", pose=sapien.Pose(), width=128, height=128, intrinsic=intrinsic_128, near=0.01, far=100, mount=self.cam_mount),
            CameraConfig("base_camera", pose=pose, width=128, height=128, intrinsic=intrinsic_128, near=0.01, far=100)]
        else:
            # return [CameraConfig("left_camera", self.left_base_pose, width=1280, height=720, intrinsic=intrinsic, near=0.01, far=100),
            # CameraConfig("base_camera", pose=pose, width=1280, height=720, intrinsic=intrinsic_128, near=0.01, far=100)]
            return [CameraConfig("left_camera", self.left_base_pose, width=128, height=128, intrinsic=intrinsic, near=0.01, far=100),
            CameraConfig("base_camera", pose=pose, width=128, height=128, intrinsic=intrinsic_128, near=0.01, far=100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        # self._hidden_objects.append(self.goal_site)
        seg_ids = torch.tensor([0], dtype=torch.int16, device=self.device)
        self.seg_ids = torch.concatenate([seg_ids, self.unwrapped.scene.actors["ground"].per_scene_id]).to(self.device)
        if self.random_cfg["camera_random"]:
            cam_mount = self.scene.create_actor_builder()
            cam_mount.initial_pose=Pose.create(self.left_base_pose)
            self.cam_mount = cam_mount.build_kinematic("camera_mount")
            # self.cam_mount = self.scene.create_actor_builder().build_kinematic("camera_mount")
        if self.random_cfg["texture_random"]:
            # render_body_component1: RenderBodyComponent = self.table_scene.table._objs[0].find_component_by_type(RenderBodyComponent)
            render_body_component2: RenderBodyComponent = self.cube._objs[0].find_component_by_type(RenderBodyComponent)
            render_body_component_list = [render_body_component2]
            for render_body_component in render_body_component_list:
                for render_shape in render_body_component.render_shapes:
                    for part in render_shape.parts:
                        color = np.random.uniform(low=0., high=3., size=(3, )).tolist() + [1]
                        part.material.set_base_color(color)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[:, 2] = self.cube_half_size
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))
            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            goal_xyz[:, 2] = torch.rand((b)) * 0.3 + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))
            if self.random_cfg["camera_random"]:
                pose = self.left_base_pose
                pose = Pose.create(pose)
                pose = pose * Pose.create_from_pq(
                    p=torch.rand((self.num_envs, 3)) * 0.05,
                    q=randomization.random_quaternions(
                        n=self.num_envs, device=self.device, bounds=(-np.pi / 20, np.pi / 20)
                    ),
                )
                self.cam_mount.set_pose(pose)

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp.pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
            )
        return obs

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
            <= self.goal_thresh
        )
        is_grasped = self.agent.is_grasping(self.cube)
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_obj_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.cube.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped

        qvel_without_gripper = self.agent.robot.get_qvel()
        if self.robot_uids == "xarm6_robotiq":
            qvel_without_gripper = qvel_without_gripper[..., :-6]
        elif self.robot_uids == "panda":
            qvel_without_gripper = qvel_without_gripper[..., :-2]
        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(qvel_without_gripper, axis=1)
        )
        reward += static_reward * info["is_obj_placed"]

        reward[info["success"]] = 5
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5
    
    def background_randomization(self, obs: Any, target_size=64):
        random_obs = {}
        for cam_name in obs["sensor_data"].keys():
            if "hand" in cam_name:
                green_screen_image = self.eye_background_image
            elif "left" in cam_name:
                green_screen_image = self.left_background_image
            elif "base" in cam_name:
                green_screen_image = self.scene_background_image
            else:
                green_screen_image = self.scene_background_image
            camera_data = obs["sensor_data"][cam_name]
            seg = camera_data["segmentation"].to(self.device)
            mask = torch.zeros_like(seg, device=self.device)
            mask[torch.isin(seg, self.seg_ids)] = 1
            camera_data["rgb"] = camera_data["rgb"].to(self.device) * (1 - mask) + green_screen_image * mask
            # crop_image = center_crop_and_resize(camera_data["rgb"].cpu().numpy()[0], target_size)
            imgs = camera_data["rgb"].cpu().numpy()   # shape: (N, H, W, C) 或 (N, C, H, W)
            cropped_list = [center_crop_and_resize(img, target_size) for img in imgs]
            crop_image = np.stack(cropped_list, axis=0)
            random_obs[cam_name] = crop_image
        return random_obs
    
    def reset(self, *args, **kwargs):
        scene_background_index = np.random.randint(len(self.scene_background_list))
        eye_background_index = np.random.randint(len(self.eye_background_list))
        left_background_index = np.random.randint(len(self.left_background_list))
        # self.scene_background_image = torch.from_numpy(self.scene_background_list[scene_background_index]).float().to(self.device)
        # self.eye_background_image = torch.from_numpy(self.eye_background_list[eye_background_index]).float().to(self.device)
        # self.left_background_image = torch.from_numpy(self.left_background_list[left_background_index]).float().to(self.device)
        self.scene_background_image = torch.from_numpy(center_crop_and_resize(self.scene_background_list[scene_background_index], 128)).float().to(self.device)
        self.eye_background_image = torch.from_numpy(center_crop_and_resize(self.eye_background_list[eye_background_index], 128)).float().to(self.device)
        self.left_background_image = torch.from_numpy(center_crop_and_resize(self.left_background_list[left_background_index], 128)).float().to(self.device)
        return super().reset(*args, **kwargs)
    
    def _load_lighting(self, options: dict):
        if self.random_cfg["light_random"]:
            for scene in self.scene.sub_scenes:
                amb_intensity = np.random.uniform(0.0, 0.4)
                scene.ambient_light = [amb_intensity, amb_intensity, amb_intensity]
                azimuth = np.random.uniform(0, 2 * np.pi)
                elevation = np.random.uniform(np.deg2rad(15), np.deg2rad(85))
                lx = np.cos(elevation) * np.cos(azimuth)
                ly = np.cos(elevation) * np.sin(azimuth)
                lz = -np.sin(elevation)
                main_light_intensity = np.random.uniform(0.5, 4.0)
                scene.add_directional_light(
                    [lx, ly, lz], 
                    [main_light_intensity, main_light_intensity, main_light_intensity], 
                    shadow=True, 
                    shadow_scale=5, 
                    shadow_map_size=4096
                )
                if np.random.random() > 0.5:
                    fill_intensity = np.random.uniform(0.1, 1.0)
                    scene.add_directional_light(
                        [-lx, -ly, -1],
                        [fill_intensity, fill_intensity, fill_intensity]
                    )

        else:
            super()._load_lighting(options)


@register_env("PickCube-v3", max_episode_steps=50)
class RepoPickCubeEnv(BaseEnv):
    """
    **Task Description:**
    A simple task where the objective is to grasp a red cube and move it to a target goal position.

    **Randomizations:**
    - the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
    - the cube's z-axis rotation is randomized to a random angle
    - the target goal position (marked by a green sphere) of the cube has its xy position randomized in the region [0.1, 0.1] x [-0.1, -0.1] and z randomized in [0, 0.3]

    **Success Conditions:**
    - the cube position is within `goal_thresh` (default 0.025m) euclidean distance of the goal position
    - the robot is static (q velocity < 0.2)
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = [
        "panda",
        "fetch",
        "xarm6_robotiq",
    ]
    agent: Union[Panda, Fetch, XArm6Robotiq]
    cube_half_size = 0.02
    goal_thresh = 0.025

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.random_cfg = {key: kwargs.pop(key, False) for key in RANDOM_KEYS}
        print("Randomization config:", self.random_cfg)
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.scene_background_image = None
        if self.random_cfg["background_random"]:
            background_name = "background"
        else:
            background_name = "background1"
        self.scene_background_path = sorted(os.listdir(os.path.join(os.path.dirname(__file__), background_name)), key=lambda x: x.lower())
        front_files = [f for f in self.scene_background_path if f.lower().startswith("main")]
        eye_files = [f for f in self.scene_background_path if f.lower().startswith("eye")]
        left_files = [f for f in self.scene_background_path if f.lower().startswith("left")]
        self.scene_background_list = [cv2.cvtColor(cv2.imread(os.path.join(os.path.dirname(__file__), background_name, file_name)), cv2.COLOR_BGR2RGB) for file_name in front_files]
        # self.tensor_scene_background_list = [torch.from_numpy(image).float() for image in self.scene_background_list]
        self.eye_background_list = [cv2.cvtColor(cv2.imread(os.path.join(os.path.dirname(__file__), background_name, file_name)), cv2.COLOR_BGR2RGB) for file_name in eye_files]
        # self.tensor_eye_background_list = [torch.from_numpy(image).float() for image in self.eye_background_list]
        self.left_background_list = [cv2.cvtColor(cv2.imread(os.path.join(os.path.dirname(__file__), background_name, file_name)), cv2.COLOR_BGR2RGB) for file_name in left_files]
        # self.tensor_left_background_list = [torch.from_numpy(image).float() for image in self.left_background_list]
        self.left_base_pose = sapien_utils.look_at(eye=[-0.38155, 1.2102138255883726, 0.4851141636379064], target=[-0.35978, 0.13906, 0.30676])
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        calib_p = torch.tensor(
            [0.553,
            -0.027871877550427117,
            0.6863282894133899],
            dtype=torch.float32,
            device=self.device,
        )
        calib_q = torch.tensor(
            [0.03128285, -0.37759845, 0.00461725, 0.92542925],
            dtype=torch.float32,
            device=self.device,
        )
        calib_p = calib_p.unsqueeze(0).repeat(self.num_envs, 1)  # (num_envs, 3)
        calib_q = calib_q.unsqueeze(0).repeat(self.num_envs, 1)  # (num_envs, 4)
        pose = Pose.create_from_pq(
            p=calib_p,
            q=calib_q,
        )
        intrinsic = np.array([[905.0100708007812, 0.0, 653.8479614257812], [0.0, 905.0863037109375, 373.97210693359375], [0.0, 0.0, 1.0]])
        intrinsic_128 = np.array([[90.501, 0.0,   65.38],[0.0,    160.9, 66.23],[0.0,    0.0,   1.0]])
        base_pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        if self.random_cfg["camera_random"]:
            return [CameraConfig("left_camera", pose=sapien.Pose(), width=128, height=128, intrinsic=intrinsic_128, near=0.01, far=100, mount=self.cam_mount),
            CameraConfig("base_camera", pose=pose, width=128, height=128, intrinsic=intrinsic_128, near=0.01, far=100)]
        else:
            return [CameraConfig("left_camera", self.left_base_pose, width=1280, height=720, intrinsic=intrinsic, near=0.01, far=100),
            CameraConfig("base_camera", pose=pose, width=1280, height=720, intrinsic=intrinsic_128, near=0.01, far=100)]
            # return [CameraConfig("left_camera", self.left_base_pose, width=128, height=128, intrinsic=intrinsic, near=0.01, far=100),
            # CameraConfig("base_camera", pose=pose, width=128, height=128, intrinsic=intrinsic_128, near=0.01, far=100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        # self._hidden_objects.append(self.goal_site)
        seg_ids = torch.tensor([0], dtype=torch.int16, device=self.device)
        self.seg_ids = torch.concatenate([seg_ids, self.unwrapped.scene.actors["ground"].per_scene_id]).to(self.device)
        if self.random_cfg["camera_random"]:
            cam_mount = self.scene.create_actor_builder()
            cam_mount.initial_pose=Pose.create(self.left_base_pose)
            self.cam_mount = cam_mount.build_kinematic("camera_mount")
            # self.cam_mount = self.scene.create_actor_builder().build_kinematic("camera_mount")
        if self.random_cfg["texture_random"]:
            # render_body_component1: RenderBodyComponent = self.table_scene.table._objs[0].find_component_by_type(RenderBodyComponent)
            render_body_component2: RenderBodyComponent = self.cube._objs[0].find_component_by_type(RenderBodyComponent)
            render_body_component_list = [render_body_component2]
            for render_body_component in render_body_component_list:
                for render_shape in render_body_component.render_shapes:
                    for part in render_shape.parts:
                        color = np.random.uniform(low=0., high=3., size=(3, )).tolist() + [1]
                        part.material.set_base_color(color)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[:, 2] = self.cube_half_size
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))
            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            goal_xyz[:, 2] = torch.rand((b)) * 0.3 + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))
            if self.random_cfg["camera_random"]:
                pose = self.left_base_pose
                pose = Pose.create(pose)
                pose = pose * Pose.create_from_pq(
                    p=torch.rand((self.num_envs, 3)) * 0.05,
                    q=randomization.random_quaternions(
                        n=self.num_envs, device=self.device, bounds=(-np.pi / 20, np.pi / 20)
                    ),
                )
                self.cam_mount.set_pose(pose)

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp.pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
            )
        return obs

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
            <= self.goal_thresh
        )
        is_grasped = self.agent.is_grasping(self.cube)
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_obj_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.cube.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped

        qvel_without_gripper = self.agent.robot.get_qvel()
        if self.robot_uids == "xarm6_robotiq":
            qvel_without_gripper = qvel_without_gripper[..., :-6]
        elif self.robot_uids == "panda":
            qvel_without_gripper = qvel_without_gripper[..., :-2]
        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(qvel_without_gripper, axis=1)
        )
        reward += static_reward * info["is_obj_placed"]

        reward[info["success"]] = 5
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5
    
    def background_randomization(self, obs: Any, target_size=64):
        random_obs = {}
        for cam_name in obs["sensor_data"].keys():
            if "hand" in cam_name:
                green_screen_image = self.eye_background_image
            elif "left" in cam_name:
                green_screen_image = self.left_background_image
            elif "base" in cam_name:
                green_screen_image = self.scene_background_image
            else:
                green_screen_image = self.scene_background_image
            camera_data = obs["sensor_data"][cam_name]
            seg = camera_data["segmentation"].to(self.device)
            mask = torch.zeros_like(seg, device=self.device)
            mask[torch.isin(seg, self.seg_ids)] = 1
            camera_data["rgb"] = camera_data["rgb"].to(self.device) * (1 - mask) + green_screen_image * mask
            # crop_image = center_crop_and_resize(camera_data["rgb"].cpu().numpy()[0], target_size)
            imgs = camera_data["rgb"].cpu().numpy()   # shape: (N, H, W, C) 或 (N, C, H, W)
            cropped_list = [center_crop_and_resize(img, target_size) for img in imgs]
            crop_image = np.stack(cropped_list, axis=0)
            random_obs[cam_name] = crop_image
        return random_obs
    
    def reset(self, *args, **kwargs):
        scene_background_index = np.random.randint(len(self.scene_background_list))
        eye_background_index = np.random.randint(len(self.eye_background_list))
        left_background_index = np.random.randint(len(self.left_background_list))
        self.scene_background_image = torch.from_numpy(self.scene_background_list[scene_background_index]).float().to(self.device)
        self.eye_background_image = torch.from_numpy(self.eye_background_list[eye_background_index]).float().to(self.device)
        self.left_background_image = torch.from_numpy(self.left_background_list[left_background_index]).float().to(self.device)
        # self.scene_background_image = torch.from_numpy(center_crop_and_resize(self.scene_background_list[scene_background_index], 128)).float().to(self.device)
        # self.eye_background_image = torch.from_numpy(center_crop_and_resize(self.eye_background_list[eye_background_index], 128)).float().to(self.device)
        # self.left_background_image = torch.from_numpy(center_crop_and_resize(self.left_background_list[left_background_index], 128)).float().to(self.device)
        return super().reset(*args, **kwargs)
    
    def _load_lighting(self, options: dict):
        if self.random_cfg["light_random"]:
            for scene in self.scene.sub_scenes:
                amb_intensity = np.random.uniform(0.0, 0.4)
                scene.ambient_light = [amb_intensity, amb_intensity, amb_intensity]
                azimuth = np.random.uniform(0, 2 * np.pi)
                elevation = np.random.uniform(np.deg2rad(15), np.deg2rad(85))
                lx = np.cos(elevation) * np.cos(azimuth)
                ly = np.cos(elevation) * np.sin(azimuth)
                lz = -np.sin(elevation)
                main_light_intensity = np.random.uniform(0.5, 4.0)
                scene.add_directional_light(
                    [lx, ly, lz], 
                    [main_light_intensity, main_light_intensity, main_light_intensity], 
                    shadow=True, 
                    shadow_scale=5, 
                    shadow_map_size=4096
                )
                if np.random.random() > 0.5:
                    fill_intensity = np.random.uniform(0.1, 1.0)
                    scene.add_directional_light(
                        [-lx, -ly, -1],
                        [fill_intensity, fill_intensity, fill_intensity]
                    )

        else:
            super()._load_lighting(options)