import gc
import pickle

import os
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ['DEVICE'] = "cuda"
os.environ["WANDB_DISABLED"] = "true"

from data_utils.utils import load_data  # data functions
from data_utils.utils import set_seed  # helper functions
from policy_heads import *
from dataclasses import dataclass, field, fields, asdict
from typing import Dict, Optional, Sequence, List

from aloha_scripts.constants import TASK_CONFIGS
from qwen2_vla.utils.robot_data_processor import Qwen2VLAProcess
from transformers import AutoConfig, AutoModel, AutoProcessor
from qwen2_vla import QWen2VLATrainer
import transformers
import IPython
e = IPython.embed
from data_utils.data_collators import Qwen2VLADataCollatorForSupervisedDataset
from qwen2_vla import model_load_utils as ml_utils
import torch
local_rank = None
from aloha_scripts.utils import *
#  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>parameters<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
@dataclass
class ActionHeadArguments:
    policy_head_type: str = field(default="dit_diffusion_policy") # unet_diffusion_policy
    policy_head_size: str = field(default="DiT_B") # DiT_XL, DiT_L, DiT_B, DiT_S
    state_dim: int = 7 # state dimension
    action_dim: int = 10 # action dimension


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    model_pretrain: Optional[str] = field(default="") # pretrained model weights path
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)

    concat: str = field(default="None")
    policy_class: str = field(default="droid_diffusion")
    using_film: bool = field(default=True) # fusion modules, default using film

    load_pretrain_dit: bool = field(default=False) # whether to load weights of pretrained diffusion head
    pretrain_dit_path: Optional[str] = field(default=None) # path to pretrained diffusion head, used when load_pretrain_dit is True

    is_tinyvla: bool = field(default=False)

@dataclass
class DataArguments:
    lazy_preprocess: bool = False
    episode_first: bool = False  # batchsampler will samples episode index first and then samples timesteps
    select_seg_token_mask: bool = False
    use_reasoning: bool = False
    is_multimodal: bool = False
    image_aspect_ratio: str = 'square'
    task_name: str = field(default="stack_cube_2024_6_2") # task name corresponding to aloha_scripts/constants.py
    skip_mirrored_data: bool = field(default=False)
    chunk_size: int = field(default=16)
    delta_control: bool = field(default=False)
    image_size_stable: str = "480"  # image size of non-wrist camera
    image_size_wrist: str = "56" # image size of wrist camera
    history_images_length: int = 1 # length of history images

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    using_ema: bool = field(default=False) # whether to use ema update whole module, default to false

    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.98)
    adam_epsilon: float = field(default=1e-7)
    remove_unused_columns: bool = field(default=False)

    flash_attn: bool = field(default=False)
    freeze_vision_tower: bool = field(default=False)
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    resume_from_checkpoint: bool = field(default=False)
    llm_loss_weight: float = field(default=1.0)

    seed: int = field(default=0)

    # logger
    logging_dir: str = field(default='./logs')  # TensorBoard
    logging_strategy: str = field(default='steps')
    logging_steps: int = field(default=10)

    save_steps: int = field(default=10)
    num_train_epochs: int = field(default=3)
    max_steps: int = field(default=5000)

    # validate, unused
    do_eval: bool = field(default=False)
    evaluation_strategy: str = field(default="no")
    eval_steps: int = field(default=200)
    per_device_eval_batch_size: int = field(default=32)

    load_pretrain: bool = False # loading pretrained VLA (For stage 3 training)
    dataloader_pin_memory: bool = False

    # lora, used when lora_enable is True
    lora_enable: bool = False # using lora or not
    lora_module: str = "vit" # which part to lora finetune, used when lora_enable is True
    lora_task_type: str = 'CAUSAL_LM'
    lora_r: int = 64
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    non_lora_lr: Optional[float] = None

    group_by_modality_length: bool = field(default=False)

    model_max_length: int = field(
        default=2048,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )


#  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<parameters>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def parse_param():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, ActionHeadArguments))
    model_args, data_args, training_args, action_head_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type  # {'fp4', 'nf4'}
            )
        ))

    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **asdict(action_head_args))
    # initialize diffusion action head
    if action_head_args.policy_head_type == 'dit_diffusion_policy': # scaledp, using dit block
        config.policy_head_size = action_head_args.policy_head_size
        config.policy_head_config = AutoConfig.for_model(model_type=config.policy_head_type,
                                                       model_size=action_head_args.policy_head_size,
                                                       cond_dim=config.hidden_size, action_dim=action_head_args.action_dim,
                                                         prediction_horizon=data_args.chunk_size,
                                                       state_dim=action_head_args.state_dim,
                                                         is_tinyvla=model_args.is_tinyvla)
    elif action_head_args.policy_head_type == 'unet_diffusion_policy': # unet
        config.policy_head_config = AutoConfig.for_model(model_type=config.policy_head_type,
                                                       global_cond_dim=config.hidden_size, action_dim=action_head_args.action_dim,
                                                       state_dim=action_head_args.state_dim,
                                                         is_tinyvla=model_args.is_tinyvla)
    else:
        raise NotImplementedError(f"Unsupported policy head type {action_head_args.policy_head_type}")
    setattr(config.policy_head_config, "input_dim", asdict(action_head_args)['action_dim'])
    setattr(config.policy_head_config, "state_dim", asdict(action_head_args)['state_dim'])

    for k in ['concat', 'using_film']:
        setattr(config, k, asdict(model_args)[k])
    config.llm_loss_weight = training_args.llm_loss_weight

    if model_args.is_tinyvla:
        rank0_print(f"{RED} This is TinyVLA, Please Check Using_film equals False:Using_film {model_args.using_film} {RESET}")
        time.sleep(1)
    return model_args, data_args, training_args, action_head_args, config, bnb_model_from_pretrained_args
def train_bc(train_dataset=None, val_dataset=None, model=None, config=None, sampler_params=None, tokenizer=None, processor=None):

    set_seed(config['training_args'].seed)
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if config['training_args'].bf16 else torch.float32))
    if config['data_args'].history_images_length > 2:
        rank0_print(f"{RED} Using History and Turn to Video mode.{RESET}")
        video = True
    else:
        video = False

    data_collator = Qwen2VLADataCollatorForSupervisedDataset(multimodal_processor=processor, computed_type=compute_dtype, tokenizer=tokenizer, video=video)

    model.config.use_cache = True
    model.config.save_pretrained(config['training_args'].output_dir)

    data_module = dict(train_dataset=train_dataset,
                       data_collator=data_collator,
                       eval_dataset=val_dataset
                       )
    trainer = QWen2VLATrainer(model=model,
                                 tokenizer=tokenizer,
                                 args=config['training_args'],
                                 sampler_params=sampler_params,
                                 **data_module)

    trainer.train(resume_from_checkpoint=config['training_args'].resume_from_checkpoint)

    trainer.save_state()

    model.config.use_cache = True

    if config['training_args'].lora_enable:
        state_dict = ml_utils.get_peft_state_maybe_zero_3(
            model.named_parameters(), config['training_args'].lora_bias
        )
        non_lora_state_dict = ml_utils.get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=False
        )
        if config['training_args'].local_rank == 0 or config['training_args'].local_rank == -1:
            model.config.save_pretrained(config['training_args'].output_dir)
            model.save_pretrained(config['training_args'].output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict,
                       os.path.join(config['training_args'].output_dir, 'non_lora_trainables.bin'))
    else:
        ml_utils.safe_save_model_for_hf_trainer(trainer=trainer,
                                                  output_dir=config['training_args'].output_dir)



def main(all_config=None, model_config=None):
    set_seed(1)
    task_config = TASK_CONFIGS[all_config['data_args'].task_name]
    dataset_dir = task_config['dataset_dir']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    stats_dir = task_config.get('stats_dir', None)
    sample_weights = task_config.get('sample_weights', None)
    train_ratio = task_config.get('train_ratio', 0.999)
    name_filter = task_config.get('name_filter', lambda n: True)

    all_config['camera_names'] = camera_names
    all_config['episode_len'] = episode_len

    # load qwen2_vl tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        all_config['model_args'].model_name_or_path,
    )
    # load qwen2_vl input processor
    multimodal_processor = AutoProcessor.from_pretrained(all_config['model_args'].model_name_or_path)
    # load dexvla model
    model, data_args = ml_utils.load_model(config=all_config, qwen2_vla_config=model_config, rank0_print=rank0_print, tokenizer=tokenizer)

    rank0_print(f"{RED} Using Qwen2VLA as VLA backbone {RESET}")
    vla_process = Qwen2VLAProcess(tokenizer=tokenizer, multimodal_processor=multimodal_processor, data_args=all_config['data_args'], camera_names=camera_names)

    train_dataset, val_dataset, stats, sampler_params = load_data(dataset_dir, name_filter, camera_names, all_config['training_args'].per_device_train_batch_size,
                                                                  all_config['training_args'].per_device_eval_batch_size, all_config['data_args'].chunk_size,
                                                                  skip_mirrored_data=all_config['data_args'].skip_mirrored_data,
                                                                  config=all_config,
                                                                  stats_dir_l=stats_dir,
                                                                  rank0_print=rank0_print,
                                                                  policy_class=all_config['action_head_args'].policy_head_type,
                                                                  sample_weights=sample_weights, train_ratio=train_ratio, llava_pythia_process=vla_process)

    # exit(0)
    stats_path = os.path.join(all_config['training_args'].output_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataset=train_dataset, model=model, val_dataset=val_dataset, config=all_config, sampler_params=sampler_params, tokenizer=tokenizer, processor=multimodal_processor)
    # save dataset stats
    stats_path = os.path.join(all_config['training_args'].output_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)


if __name__ == '__main__':
    model_args, data_args, training_args, action_head_args, model_config, bnb_model_from_pretrained_args = parse_param()
    config = {
        'model_args':model_args,
        'data_args':data_args,
        'training_args':training_args,
        'action_head_args':action_head_args,
        'bnb_model_from_pretrained_args':bnb_model_from_pretrained_args
    }

    config_dict = {k:asdict(v) if not isinstance(v, dict) else v for k,v in config.items()}

    ckpt = os.path.join(config['training_args'].output_dir, f"checkpoint-{config['training_args'].save_steps}")

    # resume_from_training
    if os.path.exists(ckpt) and config['training_args'].resume_from_checkpoint:
        rank0_print(f"{RED}Resuming Training............{RESET}")
    main(all_config=config, model_config=model_config)
    pass


