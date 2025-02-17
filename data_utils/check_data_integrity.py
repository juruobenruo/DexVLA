from utils import find_all_hdf5, flatten_list
import os
path = "/media/rl/SanDisk PRO-BLAD/aloha_data/1_10"
import torch
import h5py
import numpy as np
from tqdm import tqdm
def get_norm_stats(dataset_path_list, rank0_print=print):
    """
    Calculate normalization statistics from a list of HDF5 dataset paths.

    Args:
        dataset_path_list : List of paths to HDF5 dataset files
        rank0_print : Function for printing messages. Defaults to built-in print.

    Returns:
        Tuple: A tuple containing:
            - stats : Dictionary with the following keys:
                - action_mean : Mean of actions
                - action_std : Standard deviation of actions
                - action_min : Minimum values of actions
                - action_max : Maximum values of actions
                - qpos_mean : Mean of joint positions
                - qpos_std : Standard deviation of joint positions
                - example_qpos : Example of joint positions
            - all_episode_len : List of episode lengths
    """
    all_qpos_data = []
    all_action_data = []
    all_episode_len = []

    for dataset_path in tqdm(dataset_path_list):
        try:
            with h5py.File(dataset_path, 'r') as root:
                qpos = root['/observations/qpos'][()]
                qvel = root['/observations/qvel'][()]
                action = root['/action'][()]
        except Exception as e:
            rank0_print(f'Error loading {dataset_path} in get_norm_stats')
            rank0_print(e)
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
        all_episode_len.append(len(qpos))
    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    # normalize action data
    action_mean = all_action_data.mean(dim=[0]).float()
    action_std = all_action_data.std(dim=[0]).float()
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0]).float()
    qpos_std = all_qpos_data.std(dim=[0]).float()
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    action_min = all_action_data.min(dim=0).values.float()
    action_max = all_action_data.max(dim=0).values.float()

    eps = 0.0001
    stats = {"action_mean": action_mean.numpy(), "action_std": action_std.numpy(),
             "action_min": action_min.numpy() - eps,"action_max": action_max.numpy() + eps,
             "qpos_mean": qpos_mean.numpy(), "qpos_std": qpos_std.numpy(),
             "example_qpos": qpos}

    return stats, all_episode_len


##################################################################################################################
tasks = os.listdir(path)

dataset_dir_l = [os.path.join(path, t) for t in tasks]
dataset_path_list_list = [find_all_hdf5(dataset_dir, skip_mirrored_data=True) for dataset_dir in dataset_dir_l]
dataset_path_list = flatten_list(dataset_path_list_list)

print(get_norm_stats(dataset_path_list))
