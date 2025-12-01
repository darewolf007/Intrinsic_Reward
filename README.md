# Intrinsic_Reward

python run/train_repo.py --algo repo --env_id maniskill-PickCube --expr_name benchmark --seed 0

python demo3/evaluate.py task=ms-pick-place-semi checkpoint=/data2/user/sunhaowen/hw_mine/Intrinsic_Reward/Maniskill3/PickCube.pt save_video=true save_trajectory=true obs="state" obs_save="rgb"


python demo3/train.py task=ms-pick-place-semi steps=1000000 demo_path=/data2/user/sunhaowen/hw_mine/Intrinsic_Reward/ms-pick-place-semi_trajectories_10.pkl  enable_reward_learning=true
