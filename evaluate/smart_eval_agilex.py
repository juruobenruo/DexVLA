import os
from qwen2_vla.model_load_utils import load_model_for_eval

import torch
from torchvision import transforms
import cv2

import numpy as np
import time

from aloha_scripts.constants import FPS

from data_utils.utils import compute_dict_mean, set_seed, detach_dict, calibrate_linear_vel, \
    postprocess_base_action  # helper functions
from PIL import Image
from qwen_vl_utils import fetch_image
from einops import rearrange
import torch_utils as TorchUtils
# import matplotlib.pyplot as plt
import sys
from policy_heads import *
# from cv2 import aruco
from qwen2_vla.utils.image_processing_qwen2_vla import *


import copy


def pre_process(robot_state_value, key, stats):
    tmp = robot_state_value
    tmp = (tmp - stats[key + '_mean']) / stats[key + '_std']
    return tmp


def get_obs():
    """
    This function is used to get observations(images and robot states) in your robot environment.
    And you need to resize the images into the [320, 180] which is correspongding to training.
    """
    return None, None # images, states


def time_ms():
    return time.time_ns() // 1_000_000

class qwen2_vla_policy:
    def __init__(self, policy_config, data_args=None):
        super(qwen2_vla_policy).__init__()
        self.load_policy(policy_config)
        self.data_args = data_args

    def load_policy(self, policy_config):
        self.policy_config = policy_config
        # self.conv = conv_templates[policy_config['conv_mode']].copy()
        model_base = policy_config["model_base"] if policy_config[
            'enable_lora'] else None
        model_path = policy_config["model_path"]

        self.tokenizer, self.policy, self.multimodal_processor, self.context_len = load_model_for_eval(model_path=model_path,
                                                                                                    model_base=model_base, policy_config=policy_config)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ["[SOA]"]})

        self.config = AutoConfig.from_pretrained('/'.join(model_path.split('/')[:-1]), trust_remote_code=True)
    def datastruct_droid2qwen2vla(self, raw_lang):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": None,
                    },
                    {
                        "type": "image",
                        "image": None,
                    },
                    {
                        "type": "image",
                        "image": None,
                    },
                    {"type": "text", "text": f""},
                ],
            },
            # {"role": "assistant", "content": f''},
        ]

        messages[0]['content'][-1]['text'] = raw_lang

        return messages
    def process_batch_to_qwen2_vla(self, curr_image, robo_state, raw_lang):

        if len(curr_image.shape) == 5:  # 1,2,3,270,480
            curr_image = curr_image.squeeze(0)

        messages = self.datastruct_droid2qwen2vla(raw_lang)
        image_data = torch.chunk(curr_image, curr_image.shape[0], dim=0)  # left, right ,wrist
        image_list = []
        for i, each in enumerate(image_data):
            ele = {

            }
            each = Image.fromarray(each.cpu().squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8))
            ele['image'] = each
            ele['resized_height'] = 240
            ele['resized_width'] = 320

            image_list.append(torch.from_numpy(np.array(each)))
        # image_data = image_data / 255.0
        image_data = image_list
        text = self.multimodal_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        video_inputs = None
        model_inputs = self.multimodal_processor(
            text=text,
            images=image_data,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        data_dict = dict(states=robo_state)
        for k, v in model_inputs.items():
            data_dict[k] = v
        return data_dict


def eval_bc(policy, deploy_env, policy_config, raw_lang=None, select_one=False):
    assert raw_lang is not None, "raw lang is None!!!!!!"
    set_seed(0)

    rand_crop_resize = True
    model_config = policy.config.policy_head_config

    temporal_agg = policy_config['temp_agg']
    action_dim = model_config['input_dim']
    state_dim = model_config['state_dim']

    policy.policy.eval()

    import pickle

    ## 4. load data stats(min,max,mean....) and define post_process##############################################
    stats_path = os.path.join("/".join(policy_config['model_path'].split('/')[:-1]), f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    if policy_config["action_head"].lower() == 'act':
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    elif 'diffusion' in policy_config["action_head"]:
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
    #############################################################################################################
    env = deploy_env

    query_frequency = 16
    if temporal_agg:
        query_frequency = 1
        num_queries = int(query_frequency)
    else:
        query_frequency = int(query_frequency / 4)
        num_queries = query_frequency
        from collections import deque
        action_queue = deque(maxlen=num_queries)

    max_timesteps = int(1000 * 10)  # may increase for real-world tasks

    for rollout_id in range(1000):

        rollout_id += 0

        # env.reset(randomize=False)

        print(f"env has reset!")

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, action_dim],
                                           dtype=torch.bfloat16).cuda()
            # print(f'all_time_actions size: {all_time_actions.size()}')

        image_list = []  # for visualization

        with torch.inference_mode():
            time0 = time.time()
            DT = 1 / FPS
            for t in range(max_timesteps):

                obs = deploy_env.get_obs()

                ### 5. Realize the function of get_obs###################
                traj_rgb_np, robot_state = get_obs(obs, stats)
                #########################################################

                image_list.append(traj_rgb_np)

                robot_state = torch.from_numpy(robot_state).float().cuda()


                if t % query_frequency == 0:
                    ### 6. Augment the images################################
                    curr_image = torch.from_numpy(traj_rgb_np).float().cuda()
                    if rand_crop_resize:
                        print('rand crop resize is used!')
                        original_size = curr_image.shape[-2:]
                        ratio = 0.95
                        curr_image = curr_image[...,
                                     int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                                     int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
                        curr_image = curr_image.squeeze(0)
                        resize_transform = transforms.Resize(original_size, antialias=True)
                        curr_image = resize_transform(curr_image)
                        curr_image = curr_image.unsqueeze(0)
                        ######################################################

                # control_timestamps["policy_start"] = time_ms()
                if t == 0:
                    # warm up
                    for _ in range(2):
                        batch = policy.process_batch_to_qwen2_vla(curr_image, robot_state, raw_lang)
                        if policy_config['tinyvla']:
                            policy.policy.evaluate_tinyvla(**batch, is_eval=True, select_one=select_one, tokenizer=policy.tokenizer)
                        else:
                            all_actions, outputs = policy.policy.evaluate(**batch, is_eval=True, select_one=select_one, tokenizer=policy.tokenizer)
                            print("*" * 50)
                            print(outputs)
                    print('network warm up done')
                    time1 = time.time()

                if t % query_frequency == 0:
                    ###7. Process inputs and predict actions############################################################################################
                    batch = policy.process_batch_to_qwen2_vla(curr_image, robot_state, raw_lang)
                    if policy_config['tinyvla']:
                        all_actions, outputs = policy.policy.evaluate_tinyvla(**batch, is_eval=True, select_one=select_one, tokenizer=policy.tokenizer)
                    else:
                        all_actions, outputs = policy.policy.evaluate(**batch, is_eval=True, select_one=select_one, tokenizer=policy.tokenizer)
                    ####################################################################################################################################
                    if not temporal_agg:
                        action_queue.extend(
                            torch.chunk(all_actions, chunks=all_actions.shape[1], dim=1)[0:num_queries])

                else:
                    raw_action = action_queue.popleft()

                raw_action = raw_action.squeeze(0).cpu().to(dtype=torch.float32).numpy()
                ### 8. post process actions###########################
                action = post_process(raw_action)
                ######################################################
                print(f"after post_process action size: {action.shape}")
                print(f'step {t}, pred action: {outputs}{action}')
                if len(action.shape) == 2:
                    action = action[0]
                ##### Execute ######################################################################
                action_info = deploy_env.step(action.tolist(), mode=policy_config['control_mode'])
                ####################################################################################

            print(f'Avg fps {max_timesteps / (time.time() - time0)}')
            # plt.close()

    return


if __name__ == '__main__':
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>hyper parameters<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    action_head = 'dit_diffusion_policy'  # 'unet_diffusion_policy'
    model_size = '2B'
    policy_config = {

        #### 1. Specify path to trained DexVLA##########
        "model_path": "path/to/trained/DexVLA",
        ################################################

        "model_base": None, # only use for lora finetune

        "enable_lora": False,
        "conv_mode": "pythia",
        "temp_agg": False,
        "action_head": action_head,
        'model_size': model_size,
        'save_model': False,
        'control_mode': 'absolute', # absolute
        "tinyvla": False,
    }
    global im_size

    im_size = 320  #
    select_one = False  #

    raw_lang ='Fold t-shirt on the table.'
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>hyper parameters<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    policy = None
    #### 2. Initialize robot env##########
    agilex_bot = None
    ######################################

    #### 3. Load DexVLA####################
    policy = qwen2_vla_policy(policy_config)
    #######################################


    eval_bc(policy, agilex_bot, policy_config, raw_lang=raw_lang,
            select_one=select_one)

    print()
    exit()

