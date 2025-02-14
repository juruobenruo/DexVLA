import itertools
import os
import os.path as osp
import time
from collections import OrderedDict, defaultdict
from datetime import datetime
from tqdm import tqdm

import h5py

import torch
import collections
import tensorflow_datasets as tfds
import numpy as np
import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageTk
import time
import argparse

def get_image_list_np(img_rgb_dir_path, remove_index_list):
    cur_camera_rgb_list = []
    img_name_list = os.listdir(img_rgb_dir_path)
    img_name_list = sorted(img_name_list)

    for idx, img_name in enumerate(img_name_list):
        if idx in remove_index_list:
            continue

        img_path = os.path.join(img_rgb_dir_path, img_name)

        # (w 640, h 480)
        img_frame = Image.open(img_path).convert('RGB')
        # print(f'cur frame {img_frame} size: {img_frame.size}')
        # (480, 640, 3)
        img_np = np.array(img_frame)
        # print(f'cur np {img_path} size: {img_np.shape}')
        cur_camera_rgb_list.append(img_np)

    cur_camera_rgb_np = np.array(cur_camera_rgb_list)
    print('+++++++++++++++')
    print(f"img_rgb_dir_path: {img_rgb_dir_path}")
    print(f'cur_camera_rgb_np size: {cur_camera_rgb_np.shape}')

    return cur_camera_rgb_np


def plot_smooth_action(traj_act_xyz_np, fig_name):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    figure_name = ["x", "y", "z"]
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.plot(range(traj_act_xyz_np.shape[0]), traj_act_xyz_np[:, i], label='cur_action')
        # plt.plot(range(traj_act_xyz_np.shape[0]), traj_act_xyz_np[:, i], label='gt_action')
        plt.title(figure_name[i])
        plt.legend()
    plt.suptitle(f"Differences between predicted and target actions_traj")
    plt.tight_layout()

    # print(f"eval_traj_name: {eval_traj_name}")
    # work_dir = '/home/eai/wk/gitlab/real_franka_dataset/serl_data/serl_datasets/'
    # work_dir = '/data/team/wk/datasets/real_franka/ori_datasets'
    work_dir = '/home/jz08/wk/datasets/real_franka/ori_datasets'
    figure_dir_path = os.path.join(work_dir, f"smooth_action_results")
    os.makedirs(figure_dir_path, exist_ok=True)
    figure_path = os.path.join(figure_dir_path, f"{fig_name}.png")
    plt.savefig(figure_path)
    plt.clf()


def print_h5_structure(group, indent=0):
    for name in group:
        item = group[name]
        print(" " * indent + f"name: {name}")
        if isinstance(item, h5py.Group):
            print(" " * indent + f"Group: {name}")
            print_h5_structure(item, indent + 2)
        elif isinstance(item, h5py.Dataset):
            print(" " * indent + f"Dataset: {name} (Shape: {item.shape}, Dtype: {item.dtype})")
        else:
            print(" " * indent + f"Unknown item: {name}")


def print_dict_structure(cur_dict, indent=0):
    for name in cur_dict.keys():
        item = cur_dict[name]
        print(" " * indent + f"name: {name}")
        if isinstance(item, dict):
            print(" " * indent + f"Dict: {name}")
            print_dict_structure(item, indent + 2)
        elif isinstance(item, np.ndarray):
            print(" " * indent + f"Array: {name} (Shape: {item.shape}, Dtype: {item.dtype})")
        else:
            print(" " * indent + f"Unknown item: {name}")


def to_numpy(x):
    """
    Converts all torch tensors in nested dictionary or list or tuple to
    numpy (and leaves existing numpy arrays as-is), and returns
    a new nested structure.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """

    def f(tensor):
        if tensor.is_cuda:
            return tensor.detach().cpu().numpy()
        else:
            return tensor.detach().numpy()

    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: f,
            np.ndarray: lambda x: x,
            type(None): lambda x: x,
        }
    )


def recursive_dict_list_tuple_apply(x, type_func_dict):
    """
    Recursively apply functions to a nested dictionary or list or tuple, given a dictionary of
    {data_type: function_to_apply}.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        type_func_dict (dict): a mapping from data types to the functions to be
            applied for each data type.

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    assert (list not in type_func_dict)
    assert (tuple not in type_func_dict)
    assert (dict not in type_func_dict)

    if isinstance(x, (dict, collections.OrderedDict)):
        new_x = collections.OrderedDict() if isinstance(x, collections.OrderedDict) else dict()
        for k, v in x.items():
            new_x[k] = recursive_dict_list_tuple_apply(v, type_func_dict)
        return new_x
    elif isinstance(x, (list, tuple)):
        ret = [recursive_dict_list_tuple_apply(v, type_func_dict) for v in x]
        if isinstance(x, tuple):
            ret = tuple(ret)
        return ret
    else:
        for t, f in type_func_dict.items():
            if isinstance(x, t):
                return f(x)
        else:
            ## Pretty hacky fix to avoid error when strings get converted to tensors
            ## TODO (surajnair) try and clean this up at some point
            return x
            # raise NotImplementedError(
            #     'Cannot handle data type %s' % str(type(x)))


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)
    Returns:
        6D rotation representation, of size (*, 6)
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def euler_angles_to_rot_6d(euler_angles, convention="XYZ"):
    """
    Converts tensor with rot_6d representation to euler representation.
    """
    rot_mat = euler_angles_to_matrix(euler_angles, convention="XYZ")
    rot_6d = matrix_to_rotation_6d(rot_mat)
    return rot_6d


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def convert_h5py2np_dict(group, state_np_dict, indent=0):
    for name in group:
        item = group[name]
        print(" " * indent + f"name: {name}")
        if isinstance(item, h5py.Group):
            state_np_dict[name] = dict()
            sub_np_dict = state_np_dict[name]
            print(" " * indent + f"Group: {name}")
            convert_h5py2np_dict(item, sub_np_dict, indent + 2)
        elif isinstance(item, h5py.Dataset):
            state_np_dict[name] = item[...]
            tmp = state_np_dict[name]
            print(" " * indent + f"Dataset: {name} (Shape: {item.shape}, Dtype: {item.dtype})")
            print(" " * indent + f"Array: {name} (Shape: {tmp.shape}, Dtype: {tmp.dtype})")
        else:
            state_np_dict[name] = item
            print(" " * indent + f"Unknown item: {name}")


def print_name(name):
    print(name)

def generate_h5(obs_replay, action_replay, cfg, total_traj_cnt, act_root_dir_path, edit_flag):
    data_dict = {
        '/observations/qpos': obs_replay['qpos'],
        '/observations/qvel': obs_replay['qvel'],
        '/action': action_replay,
        'is_edited': np.array(edit_flag)
    }
    for cam_name in cfg['camera_names']:
        data_dict[f'/observations/images/{cam_name}'] = obs_replay['images'][cam_name]

    max_timesteps = len(data_dict['/observations/qpos'])
    print(f'max_timesteps: {max_timesteps}')

    data_dir = act_root_dir_path

    dataset_path = os.path.join(data_dir, f'episode_{total_traj_cnt}')
    # save the data, 2GB cache
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = True
        obs = root.create_group('observations')
        image = obs.create_group('images')
        for cam_name in cfg['camera_names']:
            _ = image.create_dataset(cam_name, (max_timesteps, cfg['cam_height'], cfg['cam_width'], 3), dtype='uint8',
                                     chunks=(1, cfg['cam_height'], cfg['cam_width'], 3), )
        qpos = obs.create_dataset('qpos', (max_timesteps, cfg['state_dim']))
        qvel = obs.create_dataset('qvel', (max_timesteps, cfg['state_dim']))
        # image = obs.create_dataset("image", (max_timesteps, 240, 320, 3), dtype='uint8', chunks=(1, 240, 320, 3))
        action = root.create_dataset('action', (max_timesteps, cfg['action_dim']))
        is_edited = root.create_dataset('is_edited', (1))
        # dt = h5py.special_dtype(vlen=str)
        # dt = h5py.string_dtype()
        # lang_intrs = root.create_dataset('lang_intrs', data=cfg['lang_intrs'], dtype=dt)
        # lang_intrs['/lang_intrs'][...] = cfg['lang_intrs']
        raw_lang = cfg['lang_intrs']
        distill_bert_lang = cfg['distill_bert_lang']
        # encoded_lang = cfg['lang_intrs_distilbert']
        root.create_dataset("language_raw", data=[raw_lang])
        root.create_dataset("distill_bert_lang", data=distill_bert_lang.cpu().detach().numpy())
        # root.create_dataset("language_distilbert", data=encoded_lang.cpu().detach().numpy())

        print(f'==== generate h5 ======')
        for name, array in data_dict.items():
            print(f"name: {name}")
            print(f"array: {array.shape}")
            root[name][...] = array

