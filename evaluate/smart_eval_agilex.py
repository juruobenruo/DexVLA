from qwen2_vla.model_load_utils import load_model_for_eval
from torchvision import transforms
import pickle
import time
from data_utils.utils import set_seed

from policy_heads import *
from qwen2_vla.utils.image_processing_qwen2_vla import *


def pre_process(robot_state_value, key, stats):
    tmp = robot_state_value
    tmp = (tmp - stats[key + '_mean']) / stats[key + '_std']
    return tmp

def process_obs(obs, states, stats):
    """
    obs: three cameras' images
    states: Tensor, robot states
    stats: mean, std of robot states and actions
    This function is used to get observations(images and robot states) in your robot environment.
    """
    cur_left_wrist = obs['left_wrist']
    cur_right_wrist = obs['right_wrist']
    cur_top = obs['top']

    traj_rgb_np = np.array([cur_top, cur_left_wrist, cur_right_wrist]) # sequential must align with constants.py
    traj_rgb_np = np.expand_dims(traj_rgb_np, axis=1)
    traj_rgb_np = np.transpose(traj_rgb_np, (1, 0, 4, 2, 3))

    cur_state_np = pre_process(states, 'qpos', stats)
    cur_state = np.expand_dims(cur_state_np, axis=0)

    return traj_rgb_np, cur_state # images, states


def time_ms():
    return time.time_ns() // 1_000_000

class qwen2_vla_policy:
    def __init__(self, policy_config, data_args=None):
        super(qwen2_vla_policy).__init__()
        self.load_policy(policy_config)
        self.data_args = data_args

    def load_policy(self, policy_config):
        self.policy_config = policy_config
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
        image_data = torch.chunk(curr_image, curr_image.shape[0], dim=0)  # top, left_wrist, right_wrist
        image_list = []
        for i, each in enumerate(image_data):
            ele = {}
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


def eval_bc(policy, deploy_env, policy_config, raw_lang=None):

    assert raw_lang is not None, "raw lang is None!!!!!!"
    set_seed(0)
    rand_crop_resize = True

    policy.policy.eval()

    ## 4. load data stats(min,max,mean....) and define post_process####################################
    stats_path = os.path.join("/".join(policy_config['model_path'].split('/')[:-1]), f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    if policy_config["action_head"].lower() == 'act':
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    elif 'diffusion' in policy_config["action_head"]:
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
    #############################################################################################################

    query_frequency = 16

    query_frequency = int(query_frequency / 4)
    num_queries = query_frequency
    from collections import deque
    action_queue = deque(maxlen=num_queries)

    max_timesteps = int(1000 * 10)  # may increase for real-world tasks

    for rollout_id in range(1000):

        rollout_id += 0

        image_list = []  # for visualization

        with torch.inference_mode():
            time0 = time.time()
            for t in range(max_timesteps):

                obs,states = deploy_env.get_obs()

                ### 5. Realize the function of get_obs###################
                traj_rgb_np, robot_state = process_obs(obs, states, stats)
                #########################################################

                image_list.append(traj_rgb_np)

                robot_state = torch.from_numpy(robot_state).float().cuda()

                if t % query_frequency == 0:
                    ### 6. Augment the images##############################################################################################
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
                    #######################################################################################################################

                if t == 0:
                    # warm up
                    for _ in range(2):
                        batch = policy.process_batch_to_qwen2_vla(curr_image, robot_state, raw_lang)
                        if policy_config['tinyvla']:
                            policy.policy.evaluate_tinyvla(**batch, is_eval=True, tokenizer=policy.tokenizer)
                        else:
                            all_actions, outputs = policy.policy.evaluate(**batch, is_eval=True, tokenizer=policy.tokenizer)
                            print("*" * 50)
                            print(outputs)
                    print('network warm up done')
                    time1 = time.time()

                if t % query_frequency == 0:
                    ###7. Process inputs and predict actions############################################################################################
                    batch = policy.process_batch_to_qwen2_vla(curr_image, robot_state, raw_lang)
                    if policy_config['tinyvla']:
                        all_actions, outputs = policy.policy.evaluate_tinyvla(**batch, is_eval=True, tokenizer=policy.tokenizer)
                    else:
                        all_actions, outputs = policy.policy.evaluate(**batch, is_eval=True, tokenizer=policy.tokenizer)
                    ####################################################################################################################################

                    action_queue.extend(
                            torch.chunk(all_actions, chunks=all_actions.shape[1], dim=1)[0:num_queries])
                    raw_action = action_queue.popleft()

                else:
                    raw_action = action_queue.popleft()

                raw_action = raw_action.squeeze(0).cpu().to(dtype=torch.float32).numpy()
                ### 8. post process actions##########################################################
                action = post_process(raw_action)
                #####################################################################################
                print(f"after post_process action size: {action.shape}")
                print(f'step {t}, pred action: {outputs}{action}')
                if len(action.shape) == 2:
                    action = action[0]
                ##### Execute ######################################################################
                action_info = deploy_env.step(action.tolist())
                ####################################################################################

class FakeRobotEnv():
    """Fake robot environment used for testing model evaluation, please replace this to your real environment."""
    def __init__(self):
        pass

    def step(self, action):
        print("Execute action successfully!!!")

    def reset(self):
        print("Reset to home position.")

    def get_obs(self):
        img = np.zeros((480, 640, 3))
        obs = {
            'left_wrist': img,
            'right_wrist': img,
            'top': img,
        }
        states = np.zeros(14)
        return obs, states


if __name__ == '__main__':
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>hyper parameters<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    action_head = 'scale_dp_policy'  # or 'unet_diffusion_policy'
    policy_config = {
        #### 1. Specify path to trained DexVLA(Required)#############################
        "model_path": "root/path/to/DexVLA_qwen2_vl_stage2_folding/checkpoint-60000",
        #############################################################################
        "model_base": None, # only use for lora finetune
        "enable_lora": False, # only use for lora finetune
        "action_head": action_head,
        "tinyvla": False,
    }

    raw_lang ='Fold t-shirt on the table.'
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>hyper parameters<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    #### 2. Initialize robot env(Required)##########
    agilex_bot = FakeRobotEnv()
    ######################################
    agilex_bot.reset()

    #### 3. Load DexVLA####################
    policy = qwen2_vla_policy(policy_config)
    #######################################


    eval_bc(policy, agilex_bot, policy_config, raw_lang=raw_lang)


