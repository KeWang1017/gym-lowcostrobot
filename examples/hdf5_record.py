import argparse

import tqdm

from gym_lowcostrobot.envs import ReachCubeEnv, LiftCubeEnv
from gym_lowcostrobot.envs.wrappers.record_hdf5 import RecordHDF5Wrapper

from scripted_policy import LiftCubePolicy, displace_object
import mujoco
import numpy as np

def do_record_hdf5(args):
    env = LiftCubeEnv(render_mode=None, action_mode="ee", observation_mode="image")
    env = RecordHDF5Wrapper(env, hdf5_folder=args.folder, length=10000, name_prefix="lift")
    env.reset()
    NUM_EPISODES = 5
    cube_origin_pos = [0.03390873, 0.22571199, 0.04]
    for episode in range(NUM_EPISODES):
        env.reset()
        cube_pos = displace_object(env, square_size=0.1, invert_y=False, origin_pos=cube_origin_pos)
        # cube_pos = env.unwrapped.data.qpos[:3].astype(np.float32)
        ee_id = env.model.body("moving_side").id
        ee_pos = env.unwrapped.data.xpos[ee_id].astype(np.float32) # default [0.03390873 0.22571199 0.14506643]
        ee_orn = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(ee_orn, env.unwrapped.data.xmat[ee_id])
        # keep orientation constant
        init_pose = np.concatenate([ee_pos, ee_orn])
        meet_pose = np.concatenate([cube_pos, ee_orn])
        policy = LiftCubePolicy(init_pose=init_pose, meet_pose=meet_pose)
        episode_length = 1000
        for i in range(episode_length):
            action = env.action_space.sample()
            result = policy()
            ee_pos = env.unwrapped.data.xpos[ee_id].astype(np.float32)
            action[:3] = result[:3] - ee_pos
            action[3] = result[7]
            # Step the environment
            observation, reward, terminted, truncated, info = env.step(action)

            # Reset the environment if it's done
            if terminted or truncated:
                # env.reset()
                break

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trace video from HDF5 trace file")
    parser.add_argument("--folder", type=str, default="data/", help="Path to HDF5 folder")
    args = parser.parse_args()
    do_record_hdf5(args)
