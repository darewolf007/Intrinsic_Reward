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
import os
files = sorted(os.listdir("/data2/user/sunhaowen/hw_mine/Intrinsic_Reward/Simulator/test_background"), key=lambda x: x.lower())
import time
start_time = time.time()
env = gym.make("PickCube-v2", obs_mode="rgb+segmentation", camera_random=False,
    light_random=False,
    texture_random=False,
    background_random = True)
obs, _ = env.reset()
end_time = time.time()
print("Time taken:", end_time - start_time)
obs, _ = env.reset()
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
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
your_image = cv2.cvtColor(cv2.resize(cv2.imread("/data2/user/sunhaowen/hw_mine/Intrinsic_Reward/Simulator/test_background/front.png"), (1280, 720)), cv2.COLOR_BGR2RGB)
green_screen_image = torch.from_numpy(your_image).to(env.device)
# only green-screen out the background and floor/ground in this case
seg_ids = torch.tensor([0], dtype=torch.int16, device=env.device)
seg_ids = torch.concatenate([seg_ids, env.unwrapped.scene.actors["ground"].per_scene_id]).to(env.device)
import time
start_time = time.time()
for cam_name in obs["sensor_data"].keys():
    camera_data = obs["sensor_data"][cam_name]
    seg = camera_data["segmentation"].to(env.device)
    mask = torch.zeros_like(seg, device=env.device)
    mask[torch.isin(seg, seg_ids)] = 1
    camera_data["rgb"] = camera_data["rgb"].to(env.device) * (1 - mask) + green_screen_image * mask
    plt.imsave(f"/data2/user/sunhaowen/hw_mine/Intrinsic_Reward/Simulator/{cam_name}_out.png", camera_data["rgb"].cpu().numpy()[0].astype("uint8"))
    # crop_image = center_crop_and_resize(camera_data["rgb"].cpu().numpy()[0], 128)
    # plt.imshow(crop_image)
    # plt.imsave(f"/data2/user/sunhaowen/hw_mine/Intrinsic_Reward/Simulator/{cam_name}_out.png", crop_image.astype("uint8"))
end_time = time.time()
print("Time taken:", end_time - start_time)