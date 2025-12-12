import time
start_time = time.time()
import cv2
import matplotlib.pyplot as plt
import torch
import RandomManiskill
import gymnasium as gym
# from RandomManiskill.random_pickcube import PickCubeEnv
print([id for id in gym.envs.registry.keys() if "PickCube" in id])
end_time = time.time()
print("Time taken:", end_time - start_time)
def center_crop_and_resize(img, target_size):
    H, W = img.shape[:2]
    short = min(H, W)
    y1 = (H - short) // 2
    x1 = (W - short) // 2
    cropped = img[y1:y1 + short, x1:x1 + short]
    resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return resized
torch.manual_seed(0)
import os
files = sorted(os.listdir("/data2/user/sunhaowen/hw_mine/Intrinsic_Reward/Simulator/test_background"), key=lambda x: x.lower())
import time
start_time = time.time()
env = gym.make("PickCube-v2", obs_mode="rgb+segmentation", 
    control_mode="pd_ee_delta_pose",
    camera_random=True,
    light_random=False,
    texture_random=False,
    background_random = False)
obs, _ = env.reset(seed=0)
end_time = time.time()
print("Time taken:", end_time - start_time)
obs, _ = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
random_obs = env.background_randomization(obs, target_size=128)
for cam_name, crop_image in random_obs.items():
    plt.imsave(f"/data2/user/sunhaowen/hw_mine/Intrinsic_Reward/Simulator/{cam_name}_out.png", crop_image[0].astype("uint8"))
# from RandomManiskill.random_pickcube import PickCubeEnv
# env = PickCubeEnv(num_envs=1, render_mode="sensors", camera_random=True,
#     light_random=False,
#     texture_random=False)
# from mani_skill.utils.wrappers import RecordEpisode
# env = RecordEpisode(env, "./videos", save_trajectory=False)
# env.reset(seed=0, options=dict(reconfigure=True))
# for _ in range(10):
#     # env.reset(seed=0)
#     env.step(env.action_space.sample())
# env.reset(seed=0, options=dict(reconfigure=True))
# for _ in range(10):
#     # env.reset(seed=0)
#     env.step(env.action_space.sample())
# env.reset(seed=0, options=dict(reconfigure=True))
# for _ in range(10):
#     # env.reset(seed=0)
#     env.step(env.action_space.sample())
# env.close()

# get obs from step
for i in range(3):
    obs, _ = env.reset(seed=i, options=dict(reconfigure=True))
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    random_obs = env.background_randomization(obs, target_size=128)
    import matplotlib.pyplot as plt
    for cam_name, crop_image in random_obs.items():
        plt.imsave(f"/data2/user/sunhaowen/hw_mine/Intrinsic_Reward/Simulator/{cam_name}_{i}_out.png", crop_image[0].astype("uint8"))
    end_time = time.time()
    print("Time taken:", end_time - start_time)