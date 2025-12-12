import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["LAZY_LEGACY_OP"] = "0"
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["TORCH_LOGS"] = "+recompiles"
import warnings
import torch
import hydra
from termcolor import colored
warnings.filterwarnings("ignore")
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from common.parser import parse_cfg
from common.seed import set_seed
from storage.buffer import Buffer
from envs import make_env
from tdmpc2 import TDMPC2
from common.logger import Logger
from trainer.online_trainer import OnlineTrainer
from trainer.demo3_trainer import Demo3Trainer
from trainer.modem_trainer import ModemTrainer
from storage.ensemble_buffer import EnsembleBuffer
from storage.demo3_buffer import Demo3Buffer
from hydra import compose, initialize
from omegaconf import OmegaConf
torch.backends.cudnn.benchmark = True

torch.set_float32_matmul_precision("high")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_MODE"] = "offline"
hydra.utils.get_original_cwd = lambda: os.getcwd()
class AgentLoader:
    def __init__(self):
        self.train_agent = None
        self.t = 0

    def load(self, task="ms-pick-place-semi", checkpoint_path=None):
        with initialize(version_base=None, config_path="./config"):
            cfg = compose(
                config_name="demo3",  # 对应 @hydra.main(config_name="demo3")
                overrides=[
                    f"task={task}",
                    "steps=1000000",
                    "demo_path=/data2/user/sunhaowen/hw_mine/Intrinsic_Reward/ms-pick-place-semi_trajectories_10.pkl",
                    "enable_reward_learning=true"
                ]
            )
        # Config checks and processing
        assert torch.cuda.is_available()
        assert cfg.steps > 0, "Must train for at least 1 step."
        cfg.demo_sampling_ratio = cfg.get("demo_sampling_ratio", 0.0)
        cfg.use_demos = False
        cfg.num_envs = 1
        assert (
            cfg.demo_sampling_ratio >= 0.0 and cfg.demo_sampling_ratio <= 1.0
        ), f"Oversampling ratio {cfg.demo_sampling_ratio} is not between 0 and 1"
        cfg = parse_cfg(cfg)
        set_seed(cfg.seed)
        print(colored("Work dir:", "yellow", attrs=["bold"]), cfg.work_dir)
        # Initiallize elements
        env_ = make_env(cfg)
        if cfg.enable_reward_learning:
            # DEMO3
            cfg.algorithm = "DEMO3" if cfg.use_demos else "TD-MPC2 + Reward Learning"
            trainer_cls = Demo3Trainer
            cfg.n_stages = env_.n_stages
            buffer_cls = EnsembleBuffer if cfg.use_demos else Demo3Buffer
        elif cfg.use_demos:
            # MoDem
            cfg.algorithm = "Modem-v2"
            trainer_cls = ModemTrainer
            buffer_cls = EnsembleBuffer
        else:
            # TDMPC
            cfg.algorithm = "TDMPC2"
            trainer_cls = OnlineTrainer
            buffer_cls = Buffer

        buffer_ = buffer_cls(cfg)
        logger_ = Logger(cfg)

        # Training code
        self.train_agent = trainer_cls(
            cfg=cfg,
            env=env_,
            agent=TDMPC2(cfg),
            buffer=buffer_,
            logger=logger_,
        )
        payload = torch.load(checkpoint_path)
        self.train_agent.agent.model.load_state_dict(payload['model'])

    def get_action(self, obs):
        action = self.train_agent.agent.act(obs, t0=self.t == 0, eval_mode=True)
        self.t += 1
        return action[0]
    
# TensorDict(
# fields={
#     rgb_base: Tensor(shape=torch.Size([16, 3, 128, 128]), device=cuda:0, dtype=torch.float32, is_shared=True),
#     rgb_hand: Tensor(shape=torch.Size([16, 3, 128, 128]), device=cuda:0, dtype=torch.float32, is_shared=True),
#     state: Tensor(shape=torch.Size([16, 18]), device=cuda:0, dtype=torch.float32, is_shared=True)},
if __name__ == "__main__":
    agent = AgentLoader()
    agent.load(checkpoint_path = '/data2/user/sunhaowen/hw_mine/Intrinsic_Reward/agent_1000000.pt')
    from tensordict import TensorDict
    test_obs = TensorDict(
        {
            "rgb_base": torch.randn(1, 3, 128, 128).to('cuda:0'),
            "rgb_hand": torch.randn(1, 3, 128, 128).to('cuda:0'),
            "state": torch.randn(1, 18).to('cuda:0'),
        },
        batch_size=[1],
    )
    agent.get_action(obs=test_obs)
