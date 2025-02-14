import torch


import transformers
import copy
from dataclasses import dataclass, field, fields, asdict
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
from transformers import CLIPImageProcessor, SiglipImageProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, AutoProcessor
import warnings
import os
from aloha_scripts.utils import *
def find_all_linear_names(model, rank0_print, lora_module=None):
    cls = torch.nn.Linear
    lora_module_names = set()

    multimodal_keywords = ['multi_modal_projector', 'lm_head', 'xattn', 'input_action_proj', 'gt_film', 'gt_action_proj', 'reasoning_action_proj', 'reasoning_film', 'merger']
    if 'vit' not in lora_module:
        multimodal_keywords.append("vision_tower")
    if 'llm' not in lora_module:
        multimodal_keywords.append("language_model")
    if 'di_head' not in lora_module: # not lora finetune policy_head
        multimodal_keywords.append("policy_head")
    else: # lora policy_head
        multimodal_keywords.append("x_embedder")
        multimodal_keywords.append("cond_obs_emb")
        multimodal_keywords.append("norm_after_pool")


    rank0_print("##" * 20)

    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue

        if isinstance(module, cls):
            lora_module_names.add(name)

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')

    return list(lora_module_names)

def load_model(config=None, qwen2_vla_config=None, rank0_print=print, tokenizer=None):
    model_args = config['model_args']
    training_args = config['training_args']
    data_args = config['data_args']
    action_args = config['action_head_args']
    if training_args.load_pretrain: # loading pretrained weights
        pass
        kwargs = {"device_map": "cuda", "torch_dtype": torch.bfloat16}
        rank0_print(f"@@@@@@@Loading pretrain weights...@@@@@@@@@@")
        assert config['model_args'].model_pretrain is not "", "load pretrain weights need set the model_pretrain in DataArguments!!!!"
        # models = load_pretrained_model(config['model_args'].model_pretrain, config['model_args'].model_name_or_path, model_name, False, False)
        model_path = config['model_args'].model_pretrain
        model_base = config['model_args'].model_name_or_path
        path = model_path.split('/')[0:-1]
        root_path = '/'.join(path)
        # lora_cfg_pretrained = AutoConfig.from_pretrained(root_path)
        # config = lora_cfg_pretrained
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)  # default use_fast=False
        rank0_print(f"{RED}Loading pretrained <<{config['model_args'].model_pretrain}>> from base models...{RESET}")
        # model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=qwen2_vla_config,**kwargs)
        if config['training_args'].flash_attn:
            model = AutoModelForCausalLM.from_pretrained(
                model_base,
                config=qwen2_vla_config,
                cache_dir=config['training_args'].cache_dir,
                trust_remote_code=True,
                _fast_init=False,
                attn_implementation="flash_attention_2",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_base,
                config=qwen2_vla_config,
                cache_dir=config['training_args'].cache_dir,
                trust_remote_code=True,
                _fast_init=False,
                # attn_implementation="flash_attention_2",
            )
        rank0_print(f'Loading pretrained additional <<{model_path}/non_lora_trainables.bin>> weights...')
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
        else:
            raise f"there is no non_lora_trainables.bin in {model_path}"

        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in
                               non_lora_trainables.items()}
        if any(k.startswith('model.policy_head.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in
                                   non_lora_trainables.items()}

        # Delete the parameters related to Lora
        keys_to_del = []
        for k, v in non_lora_trainables.items():
            if 'lora' in k:
                keys_to_del.append(k)
        for key in keys_to_del:
            del non_lora_trainables[key]

        model.load_state_dict(non_lora_trainables, strict=False)

        from peft import PeftModel
        rank0_print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)
        rank0_print('Merging LoRA weights...')
        model = model.merge_and_unload()
        rank0_print('Model is loaded...')
        model.to(torch.bfloat16)
    
    else:
        kwargs = {"device_map": "cuda", "torch_dtype": torch.bfloat16}
        if config['training_args'].flash_attn:
            model = AutoModelForCausalLM.from_pretrained(
                config['model_args'].model_name_or_path,
                config=qwen2_vla_config,
                cache_dir=config['training_args'].cache_dir,
                trust_remote_code=True,
                _fast_init=False,
                attn_implementation="flash_attention_2",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                config['model_args'].model_name_or_path,
                config=qwen2_vla_config,
                cache_dir=config['training_args'].cache_dir,
                trust_remote_code=True,
                _fast_init=False,
                # attn_implementation="flash_attention_2",
                # **kwargs, # specified device map and dtype may cause nan initialize
            )

    if model_args.load_pretrain_dit and action_args.policy_head_type == 'dit_diffusion_policy' and not config['training_args'].resume_from_checkpoint:
        assert model_args.pretrain_dit_path is not None, "please specify a pretrained dit path when setting load_pretrain_dit==True"
        rank0_print(f'{RED}>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Loading pretrained dit weights...<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<{RESET}')
        pretrain_dit_weights = torch.load(model_args.pretrain_dit_path, map_location='cpu')

        rank0_print(f'{RED} << Loading Non-EMA weights>>{RESET}')
        pretrain_dit_weights = pretrain_dit_weights['nets']['nets']

        keys_to_del_dit = []
        pretrain_dit_weights = {k[7:] if k.startswith('policy.') else k: v for k, v in pretrain_dit_weights.items()}
        for k in pretrain_dit_weights.keys():
            if 'noise_pred' not in k: # del weights of vision backbones
                keys_to_del_dit.append(k)
            if 'cond_obs_emb' in k:
                keys_to_del_dit.append(k)
        pretrained_action_dim = pretrain_dit_weights['noise_pred_net.x_embedder.weight'].shape[-1]
        if config['action_head_args'].action_dim != pretrained_action_dim:
            rank0_print(
                f"{RED}Current Action Dim is {config['action_head_args'].action_dim} != Pretrained Action Dim {pretrained_action_dim}. Initializing a new head and state encoder.... {RESET}")
            # del pretrained weights of action head
            for each in ["noise_pred_net.x_embedder.weight", "noise_pred_net.final_layer.linear.weight", "noise_pred_net.final_layer.linear.bias", 'combine.0.weight', "combine.0.bias"]:
                if each not in keys_to_del_dit:
                    keys_to_del_dit.append(each)
        for k in keys_to_del_dit:
            del pretrain_dit_weights[k]
        pretrain_dit_weights = {k[15:] if k.startswith('noise_pred_net.') else k: v for k, v in pretrain_dit_weights.items()}

        model.policy_head.load_state_dict(pretrain_dit_weights, strict=False)


    model.config.use_cache = False

    model_args.freeze_backbone = training_args.freeze_backbone
    if model_args.freeze_backbone:
        model.requires_grad_(False)
    else:
        model.requires_grad_(True)

    model.visual.requires_grad_(True) # set to true first
    model.config.freeze_vision_tower = model_args.freeze_vision_tower = training_args.freeze_vision_tower
    if model_args.freeze_vision_tower:
        for n,p in model.visual.named_parameters():
            if not 'lora' in n.lower():
                p.requires_grad = False
    else:
        for p in model.visual.parameters():
            p.requires_grad = True


    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (
            torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # if training_args.lora_enable and (not training_args.load_pretrain):
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model, rank0_print, training_args.lora_module),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type=training_args.lora_task_type,
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("##" * 20)

        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config) # !!!only set lora weights to requires_grad True!!!
        rank0_print(model)
        model.print_trainable_parameters()
    elif training_args.load_pretrain:
        rank0_print("Already loaded pretrained weights which is based on lora, skipping LoRA initialize...")


    model.config.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter

    if not model_args.freeze_backbone:
        try:
            model.lm_head.requires_grad_(True)
        except Exception as e:
            print(e)
    # action head need to be trained
    model.policy_head.requires_grad_(True)

    if config['model_args'].using_film:
        model.input_action_proj.requires_grad_(True)
        model.reasoning_action_proj.requires_grad_(True)
        model.reasoning_film.requires_grad_(True)

    vision_tower = model.visual

    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
    model.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    for k, v in model.named_parameters():
        if v.requires_grad:
            if 'film' in k or 'action_proj' in k:
                rank0_print(f"{RED}{k}{RESET}", v.requires_grad, v.dtype)
            else:
                rank0_print(k, v.requires_grad, v.dtype)

    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    if training_args.bits in [4, 8]:
        model.multi_modal_projector.to(dtype=compute_dtype, device=training_args.device)

    # model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.non_lora_lr = training_args.non_lora_lr


    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    rank0_print("!"*100)
    lora_para = sum(p.numel() for n, p in model.named_parameters() if (p.requires_grad and 'lora' in n))
    all_para = sum(p.numel() for n, p in model.named_parameters())
    train_para = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
    rank0_print(f"{RED}Lora parameters/trainalbe parameters/all parameters:{lora_para/1000000}M/{train_para/1000000}M/{(all_para-lora_para)/1000000}M{RESET}")

    return model, data_args

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def load_merge_lora_weights(model_path=None, model_base=None, kwargs=None):
    path = model_path.split('/')[0:-1]
    root_path = '/'.join(path)
    lora_cfg_pretrained = AutoConfig.from_pretrained(root_path)
    # config = lora_cfg_pretrained
    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)  # default use_fast=False
    print('Loading QWen2-VLA from base model...')
    model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                 config=lora_cfg_pretrained, **kwargs)

    print('Loading additional QWen2-VLA weights expecially non-lora part(diffusion head)...')
    if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
        non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'),
                                         )
    else:
        # this is probably from HF Hub
        from huggingface_hub import hf_hub_download
        def load_from_hf(repo_id, filename, subfolder=None):
            cache_file = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                subfolder=subfolder)
            return torch.load(cache_file, map_location='cpu')

        non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
    non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in
                           non_lora_trainables.items()}
    if any(k.startswith('model.policy_head.') for k in non_lora_trainables):
        non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in
                               non_lora_trainables.items()}

    # Delete the parameters related to Lora
    keys_to_del = []
    for k, v in non_lora_trainables.items():
        if 'lora' in k:
            keys_to_del.append(k)
    for key in keys_to_del:
        del non_lora_trainables[key]

    model.load_state_dict(non_lora_trainables, strict=False)

    from peft import PeftModel
    print('Loading LoRA weights...')
    model = PeftModel.from_pretrained(model, model_path)
    print('Merging LoRA weights...')
    model = model.merge_and_unload()
    print('Model is loaded...')
    return model, tokenizer

def load_model_for_eval(model_path, model_base, load_8bit=False, load_4bit=False, device_map="cuda", policy_config=None):
    kwargs = {"device_map": device_map}
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.bfloat16
        if policy_config['save_model']:
            kwargs['torch_dtype'] = torch.bfloat16

    if '72B' in model_base:
        kwargs = {
            "device_map":"cpu",
            "max_memory":{0:"45GiB", 1:"45GiB", "cpu":"80GiB"},
            "offload_folder": "/home/eai/wjj/qwen2_vla/offload",
            "offload_state_dict": True,
        }
        with open(os.path.join(model_base, 'device_map.json'), 'r') as f:
            device_map = json.load(f)
        kwargs['device_map'] = device_map
    
    if 'qwen2' in model_path.lower():

        if 'lora' in model_path.lower() and model_base is None: # only for lora finetuning
            warnings.warn(
                'There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument.')
        if 'lora' in model_path.lower() and model_base is not None: # only for lora finetuning
            if policy_config['pretrain_path'] is not None:
                ps = model_path.split('/')
                if not os.path.exists(os.path.join(policy_config['pretrain_path'], 'pretrain_merge_weights')):
                    print("merging pretrained weights.......")
                    model, tokenizer = load_merge_lora_weights(model_path=policy_config['pretrain_path'], model_base=model_base, kwargs=kwargs)

                    os.makedirs(os.path.join(policy_config['pretrain_path'], 'pretrain_merge_weights'), exist_ok=True)
                    model.save_pretrained(
                        os.path.join(policy_config['pretrain_path'], 'pretrain_merge_weights'))
                    tokenizer.save_pretrained(os.path.join(policy_config['pretrain_path'], 'pretrain_merge_weights'))
                # multi_modal_processor = AutoProcessor.from_pretrained(parent_model_path, use_fast=False)
                # multi_modal_processor.save_pretrained(os.path.join(parent_model_path, 'pretrain_merge_weights'))
                print("loading pretrained weights as base model.......")
                model, tokenizer = load_merge_lora_weights(model_path=model_path, model_base=os.path.join(policy_config['pretrain_path'], 'pretrain_merge_weights'), kwargs=kwargs)

            else:
                model, tokenizer = load_merge_lora_weights(model_path=model_path, model_base=model_base, kwargs=kwargs)

            if policy_config['save_model']:
                print(f"#####################################Saving merged weights of model in {kwargs['torch_dtype']}.#####################################")
                os.makedirs(os.path.join(model_path, 'merge_weights'), exist_ok=True)
                model.save_pretrained(
                    os.path.join(model_path, 'merge_weights'))
                tokenizer.save_pretrained(os.path.join(model_path, 'merge_weights'))
                skip_params = [
                    "input_action_proj",
                    "policy_head",
                    "reasoning_action_proj",
                    "reasoning_film",
                ]
                head_param = {}
                for k,v in model.named_parameters():
                    if any(skip_param in k.lower() for skip_param in skip_params):
                        head_param[k] = v
                torch.save(head_param, os.path.join(model_path, 'merge_weights/head_params.bin'))
                multi_modal_processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
                multi_modal_processor.save_pretrained(os.path.join(model_path, 'merge_weights'))
                exit(0)

            # model = model.to(torch.bfloat16)
        elif model_base is not None:
            # this may be mm projector only
            print('Loading QWen2-VLA from base model...')
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            cfg_pretrained = AutoConfig.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained,
                                                         **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            print("load QWen2-VLA!!!")
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                use_safetensors=True,
                **kwargs).to("cuda")
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                                         device_map="auto")
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.bfloat16)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)


    multi_modal_processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    model.to(device="cuda")
    print(kwargs)
    # print(model)
    return tokenizer, model, multi_modal_processor, context_len

