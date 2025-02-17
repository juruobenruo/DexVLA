<h1 align="center">
DexVLA: Vision-Language Model with Plug-In Diffusion Expert for Visuomotor Policy Learning</h1>


* **DexVLA: Vision-Language Model with Plug-In Diffusion Expert for Visuomotor Policy Learning** <br>
  [![arXiv](https://img.shields.io/badge/Arxiv-2502.05855-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2502.05855)
  


## 📰 News
* **`Feb. 17th, 2025`**: **DexVLA** is out! **Paper** can be found [here](https://arxiv.org/abs/2502.05855). The **project web** can be found [here](https://dex-vla.github.io/).

## Contents
- [Install](#install)
- [Data Preparation](#data-preparation)
- [Download Pretrained VLM](#Download-Pretrained-VLM)
- [Train](#train)
- [Evaluation](#evaluation)

## Install

1. Clone this repository and navigate to diffusion-vla folder
```bash
git clone https://github.com/juruobenruo/dexvla.git
```
Install Packages
```Shell
conda create -n dexvla python=3.10 -y
conda activate dexvla
pip install --upgrade pip  # 
pip install -r requirements.txt
cd policy_heads
pip install -e .
```
For training acceleration, please install [flash_attention](https://github.com/Dao-AILab/flash-attention).
```shell
pip install flash-attn --no-build-isolation
```

## Data Preparation
1. Our data format is the same as [act](https://github.com/MarkFzp/act-plus-plus), so you need to transfer your data into h5py format. You can refer to function "generate_h5" in [data_preprocess_scripts/rlds_to_h5py.py](https://github.com/juruobenruo/DexVLA/tree/main/data_preprocess_scripts) which is used to transfer the data from rlds format to h5py format.
```angular2html
# h5 data structure
root
  |-action (100,10)
  |-language_raw (1,)
  |-substep_reasonings (100,)
  |-observations
      |-images # multi-view
          |-left (100,480,640,3)
          |-right (100,480,640,3)
          |-wrist (100,480,640,3)
      |-joint_positions (100,7)
      |-qpos (100,7)
      |-qvel (100,7)
```
2. You have to add one entry in [constants.py](https://github.com/juruobenruo/DexVLA/blob/main/aloha_scripts/constants.py) to specify the path of your data as follows.
```python
    'example_task_name': { # for local debug
        'dataset_dir': [
            '/path/to/task1', # define the path of the dataset
        ],
        'episode_len': 1000,  
        'camera_names': ['left', 'right', 'wrist'] # keys corresponding to below h5 data structure
    }
```

## Download Pretrained VLM
We construct the VLM backbone by integrating Qwen2-VL-2B, a powerful and efficient model, into our framework. 
The Qwen2-VL 2B serves as the core of our architecture, providing robust capabilities 
for vision-language tasks. We use off-the-shelf Qwen2-VL model proposed 
in [Qwen2-VL](https://arxiv.org/pdf/2409.12191) without any post training on VLM itself. You can download the official weights from this link:

| Model               | Link                                                           |
|---------------------|----------------------------------------------------------------|
| Qwen2-VL (~2B)      | [huggingface](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) |

!!! After downloading the standard weights, you have to replace the official "config.json"
with our "docs/config.json" designed for VLA.

## Train
The training script are "scripts/stage2_train.sh" and "scripts/stage3_train.sh". And you need to change following parameters:
1. **OUTPUT** :refers to the save directory for training, which must include the keyword "qwen2"(and optionally "lora"). If LoRA training is used, the name must include "lora" (e.g., "qwen2_lora").
2. **task_name** :refers to the tasks used for training, which should be corresponded to "your_task_name" in aloha_scripts/constant.py
3. **model_name_or_path** :path to the pretrained VLM weights

Other hyperparameters like "batch_size", "save_steps" could be customized according to your computation resources.
Start training by following commands:

Train stage2. Training on large amount of tasks.
And following hyper-parameters must be set as:
1. **load_pretrain_dit** : True
2. **DIT_PRETRAIN** :Path to pretrained policy head(ScaleDP).
3. **MNOP** :Path to official Qwen2_vl weights(VLM backbone).

```shell
./scripts/stage2_train.sh 
```
Train stage3. Post-training on target dexterous tasks. 
And following hyper-parameters must be set as:
1. **MNOP** :Path to trained DexVLA of Stage2.

```shell
./scripts/stage3_train.sh 
```

## Evaluation
!!! Make sure your trained checkpoint dir has two files: "preprocessor_config.json" and "chat_template.json".
If not, please copy them from downloaded Qwen2_vl weights or this [link](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/tree/main).

You can refer to our evaluation script [smart_eval_agilex.py](https://github.com/lesjie-wen/dexvla/blob/main/evaluate/smart_eval_agilex.py) to evaluate your DexVLA.

## Trouble Shooting
1. "TypeError: _batch_encode_plus() got an unexpected keyword argument 'images'". 
Copy "preprocessor_config.json" and "chat_template.json" into your own trained 
DexVLA dir. And must be put in target "checkpoint-XXXX" dir.
~~~
Traceback (most recent call last):
  File "/media/rl/HDD/projects/open_dexvla_preview/train_vla.py", line 320, in <module>
    main(all_config=config, model_config=model_config)
  File "/media/rl/HDD/projects/open_dexvla_preview/train_vla.py", line 282, in main
    train_dataset, val_dataset, stats, sampler_params = load_data(dataset_dir, name_filter, camera_names, all_config['training_args'].per_device_train_batch_size,
  File "/media/rl/HDD/projects/open_dexvla_preview/data_utils/utils.py", line 337, in load_data
    train_dataset = EpisodicDataset(dataset_path_list, camera_names, norm_stats, train_episode_ids, train_episode_len, chunk_size, policy_class, robot=robot, llava_pythia_process=llava_pythia_process, data_args=config['data_args'])
  File "/media/rl/HDD/projects/open_dexvla_preview/data_utils/utils.py", line 43, in __init__
    a=self.__getitem__(0) # initialize self.is_sim and self.transformations
  File "/media/rl/HDD/projects/open_dexvla_preview/data_utils/utils.py", line 191, in __getitem__
    return self.llava_pythia_process.forward_process(sample, use_reasoning=self.data_args.use_reasoning)
  File "/media/rl/HDD/projects/open_dexvla_preview/qwen2_vla/utils/robot_data_processor.py", line 87, in forward_process
    model_inputs = self.multimodal_processor(
  File "/home/rl/miniconda3/envs/opendexvla/lib/python3.8/site-packages/transformers/tokenization_utils_base.py", line 3016, in __call__
    encodings = self._call_one(text=text, text_pair=text_pair, **all_kwargs)
  File "/home/rl/miniconda3/envs/opendexvla/lib/python3.8/site-packages/transformers/tokenization_utils_base.py", line 3126, in _call_one
    return self.encode_plus(
  File "/home/rl/miniconda3/envs/opendexvla/lib/python3.8/site-packages/transformers/tokenization_utils_base.py", line 3202, in encode_plus
    return self._encode_plus(
  File "/home/rl/miniconda3/envs/opendexvla/lib/python3.8/site-packages/transformers/tokenization_utils_fast.py", line 603, in _encode_plus
    batched_output = self._batch_encode_plus(
TypeError: _batch_encode_plus() got an unexpected keyword argument 'images'
~~~

## Acknowledgement
We build our project based on:
- [LLaVA](https://github.com/haotian-liu/LLaVA): an amazing open-sourced project for vision language assistant
- [act-plus-plus](https://github.com/haotian-liu/LLaVA): an amazing open-sourced project for robotics visuomotor learning
- [Miphi](https://github.com/zhuyiche/llava-phi): an amazing open-sourced project for tiny vision language model

## Citation

If you find DexVLA useful for your research and applications, please cite using this BibTeX:
```bibtex
@misc{wen2025dexvlavisionlanguagemodelplugin,
      title={DexVLA: Vision-Language Model with Plug-In Diffusion Expert for General Robot Control}, 
      author={Junjie Wen and Yichen Zhu and Jinming Li and Zhibin Tang and Chaomin Shen and Feifei Feng},
      year={2025},
      eprint={2502.05855},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2502.05855}, 
}
```
