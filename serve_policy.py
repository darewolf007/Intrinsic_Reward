import logging
import sys
import draccus
import numpy as np
from collections import deque
from termcolor import colored
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
from websocket import websocket_policy_server
from websocket import base_policy as _base_policy
from typing_extensions import override
from websocket import websocket_policy_server
# from Policy.demo3.demo3.load_agent import AgentLoader
from Policy.repo.run.load_repo import AgentLoader

class PolicyServer(_base_policy.BasePolicy):
    def __init__(self, agent):
        self.agent = agent

    @override
    def infer(self, obs):
        outputs = {}
        import cv2
        cv2.imwrite("test1.jpg",  np.transpose(obs, (1,2,0)))
        actions= self.agent.get_action(obs)
        action_queue = deque(maxlen=1)
        action_queue.extend([actions]) 
        action = action_queue.popleft()
        outputs["actions"] = action
        return outputs
    
def run():
    policy = PolicyServer(agent=AgentLoader())
    policy.agent.load(algo="repo", task="maniskill-PickCube", seed=3407, ckpt_dir = "/data2/user/sunhaowen/hw_mine/Intrinsic_Reward/Policy/repo/logdir/repo/maniskill-PickCube/test2_1280/3407")
    print(colored("Starting websocket policy server...", "green"))
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=8059,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    run()
