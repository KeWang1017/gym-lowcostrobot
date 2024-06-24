from pathlib import Path

from lerobot.scripts.push_dataset_to_hub import push_dataset_to_hub
import os

# LeRobot is still early days, no pip install, have to install from repo directly:
# pip install git+ssh://git@github.com/huggingface/lerobot.git@e67da1d7a665622c89d32cd2a58e3b4cc5fd6f4a

# You will need to have a HF_TOKEN in your environment

# https://github.com/huggingface/lerobot/blob/main/lerobot/common/datasets/push_dataset_to_hub/aloha_hdf5_format.py
# https://github.com/huggingface/lerobot/blob/main/lerobot/scripts/push_dataset_to_hub.py

DATA_DIR = '/home/ke/Documents/leap_mujoco_ws/gym-lowcostrobot/data' #os.path.join(os.path.dirname(__file__), "data")
HF_LEROBOT_VERSION: str = "v1.4"
HF_LEROBOT_BATCH_SIZE: int = 32
HF_LEROBOT_NUM_WORKERS: int = 8

# this looked very sketchy, likely this has changed
push_dataset_to_hub(
    raw_dir=DATA_DIR, # raw_dir
    # Fill in your dataset directory here
    # needs to contain "sim" and the directory should be called "_raw" but don't put that here
    # "sim_synth_demo", # dataset_id
    raw_format="aloha_hdf5", # raw_format
    repo_id="KeWangRobotics/sim_synth_demo", # repo_id

)
