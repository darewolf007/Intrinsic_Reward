import sys
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["LAZY_LEGACY_OP"] = "0"
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["TORCH_LOGS"] = "+recompiles"
sys.path.insert(0, "/data2/user/sunhaowen/hw_mine/Intrinsic_Reward/Policy/repo")
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from copy import deepcopy
import mani_skill.envs
from setup import AttrDict, parse_arguments, set_seed, set_device, setup_logger
from algorithms.repo import Dreamer, MultitaskDreamer, RePo, MultitaskRePo, TIA
from environments import make_env, make_multitask_env
import os
import numpy as np
from common.utils import (
    get_device,
    to_torch,
    to_np,
    FreezeParameters,
    lambda_return,
    preprocess,
    postprocess,
)
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#set offline
os.environ["WANDB_MODE"] = "offline"
def get_config():
    config = AttrDict()
    config.algo = "repo"
    config.env_id = "maniskill-PickCube"
    config.expr_name = "tiaocan"
    config.seed = 3407
    config.use_gpu = True
    config.gpu_id = 0
    
    # Dreamer
    config.pixel_obs = True
    config.num_steps = 500000
    config.replay_size = 800000
    config.prefill = 5000
    config.train_every = 25
    config.train_steps = 10
    config.eval_every = 5000
    config.checkpoint_every = 25000
    config.log_every = 500
    config.embedding_size = 1024
    config.hidden_size = 256
    config.belief_size = 256
    config.state_size = 45
    config.dense_activation_function = "elu"
    config.cnn_activation_function = "relu"
    config.batch_size = 128
    config.chunk_size = 15
    config.horizon = 10
    config.gamma = 0.99
    config.gae_lambda = 0.95
    config.action_noise = 0.0
    config.action_ent_coef = 2e-3
    config.latent_ent_coef = 0.0
    config.free_nats = 3
    config.model_lr = 2e-4
    config.actor_lr = 8e-5
    config.value_lr = 4e-5 
    config.grad_clip_norm = 40
    config.load_checkpoint = False
    config.load_offline = False
    config.offline_dir = "data"
    config.offline_truncate_size = 1000000
    config.save_buffer = False
    config.source_dir = ""

    # RePo
    config.target_kl = 2.0
    config.beta_lr = 2e-4
    config.init_beta = 1e-4
    config.prior_train_steps = 5

    # Disagreement model
    config.disag_model = False
    config.ensemble_size = 6
    config.disag_lr = 3e-4
    config.disag_coef = 0.0

    # Inverse dynamics
    config.inv_dynamics = False
    config.inv_dynamics_lr = 3e-4
    config.inv_dynamics_hidden_size = 512

    # Multitask
    config.share_repr = False

    # TIA
    config.tia_obs_coef = 0.0
    config.tia_adv_coef = 0.0
    config.tia_reward_train_steps = 0

    return parse_arguments(config)

class AgentLoader:
    def __init__(self):
        self.agent = None
        self.belief = None
        self.posterior_state = None
        self.action_tensor = None
        self.env = None
        self.eval_env = None
        

    def load(self, seed, algo=None, task=None, ckpt_dir=None):
        config = get_config()
        config.algo = algo if algo is not None else config.algo
        config.env_id = task if task is not None else config.env_id
        config.source_dir = ckpt_dir if ckpt_dir is not None else config.source_dir
        config.seed = seed if seed is not None else config.seed
        set_seed(config.seed)
        set_device(config.use_gpu, config.gpu_id)

        # Logger
        logger = setup_logger(config)

        # Environment
        if "multitask" in config.algo:
            self.env = make_multitask_env(config.env_id, config.seed, config.pixel_obs)
            self.eval_env = make_multitask_env(config.env_id, config.seed, config.pixel_obs)
        else:
            self.env = make_env(config.env_id, config.seed, config.pixel_obs)
            self.eval_env = make_env(config.env_id, config.seed, config.pixel_obs)

        # Sync video distractors
        if getattr(self.eval_env.unwrapped, "_img_source", None) is not None:
            self.eval_env.unwrapped._bg_source = deepcopy(self.env.unwrapped._bg_source)

        # Agent
        if config.algo == "dreamer":
            self.agent = Dreamer(config, self.env, self.eval_env, logger)
        elif config.algo == "repo":
            self.agent = RePo(config, self.env, self.eval_env, logger)
        elif config.algo == "tia":
            self.agent = TIA(config, self.env, self.eval_env, logger)
        elif config.algo == "dreamer_multitask":
            self.agent = MultitaskDreamer(config, self.env, self.eval_env, logger)
        elif config.algo == "repo_multitask":
            self.agent = MultitaskRePo(config, self.env, self.eval_env, logger)
        else:
            raise NotImplementedError("Unsupported algorithm")
        self.agent.load_checkpoint(config.source_dir)
        self.agent.toggle_train(False)
        self.belief, self.posterior_state, self.action_tensor = self.agent.init_latent_and_action()

    def process_obs(self, obs):
        with torch.no_grad():
            obs_tensor = to_torch(preprocess(obs[None]))
            (
                belief,
                posterior_state,
                action_tensor,
            ) = self.agent.update_latent_and_select_action(
                self.belief, self.posterior_state, self.action_tensor, obs_tensor, False
            )
            action = to_np(action_tensor)[0]
            self.belief, self.posterior_state, self.action_tensor = belief, posterior_state, action_tensor
        return action

    def get_action(self, obs):
        self.agent.toggle_train(False)
        action = self.process_obs(obs)
        print("Action: ", action)
        return action
# (3, 64, 64)

if __name__ == "__main__":
    import cv2
    import imageio
    import numpy as np
    from pathlib import Path
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
    with torch.no_grad():
        while not (done or truncated):
            action = agent.get_action(obs)
            action_list.append(action)
            next_obs, reward, terminated, truncated, info = agent.eval_env.step(action)
            done = terminated or truncated
            video.record(next_obs)
            obs = next_obs
            episode_reward += reward
            episode_success += info.get("success", 0)
    np.save("action.npy", np.array(action_list))
    video.save("test.mp4")