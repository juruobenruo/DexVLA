<h1 align="center">
DexVLA: Vision-Language Model with Plug-In Diffusion Expert for Visuomotor Policy Learning</h1>


* **DexVLA: Vision-Language Model with Plug-In Diffusion Expert for Visuomotor Policy Learning** <br>
  [![arXiv](https://img.shields.io/badge/Arxiv-2502.05855-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2502.05855)
  


## 📰 News
* **`Feb. 09th, 2025`**: **DexVLA** is out! **Paper** can be found [here](https://arxiv.org/abs/2502.05855). The **project web** can be found [here](https://dex-vla.github.io/).

## Contents
- [Install](#install)
- [Data Preparation](#data-preparation)
- [Download Pretrained VLM](#Download-Pretrained-VLM)
- [Train](#train)
- [Evaluation](#evaluation)

## Install

1. Clone this repository and navigate to diffusion-vla folder
```bash
git clone https://github.com/lesjie-wen/dexvla.git
```

2. Install Package
1. flash_attn
2. cd open_dex_vla pip install
3. cd policy_heads pip install
```Shell
conda create -n dexvla python=3.10 -y
conda activate dexvla
pip install --upgrade pip  # 
pip install -r requirements.txt
cd policy_heads
pip install -e .
```

## Data Preparation
1. Our data format is the same as [act](https://github.com/MarkFzp/act-plus-plus), so you need to transfer your data into h5py format. You can refer to the [rlds_to_h5py.py](https://github.com/lesjie-wen/tinyvla/blob/main/data_utils/rlds_to_h5py.py) which is used to transfer the data from rlds format to h5py format.
```angular2html
# h5 data structure
root
  |-action (100,10)
  |-language_raw (1,)
  |-observations
      |-images # multi-view
          |-left (100,480,640,3)
          |-right (100,480,640,3)
          |-wrist (100,480,640,3)
      |-joint_positions (100,7)
      |-qpos (100,7)
      |-qvel (100,7)
```
2. You have to add one entry in [constants.py](https://github.com/lesjie-wen/dexvla/blob/main/aloha_scripts/constants.py) to specify the path of your data as follows.
```python
    'example_task_name': { # for local debug
        'dataset_dir': [
            DATA_DIR + '/your_task_path', # define the path of the dataset
        ],
        'episode_len': 1000,  
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    }
```

## Download Pretrained VLM
We construct the VLM backbone by integrating Qwen2-VL 2B, a powerful and efficient model, into our framework. The Qwen2-VL 2B serves as the core of our architecture, providing robust capabilities for vision-language tasks. We follow the standard training pipeline and data setup used in [Qwen-VL](https://github.com/QwenLM/Qwen-VL), ensuring the integration is seamless. The weights of Qwen2-VL 2B used in our paper are listed as follows: 

| Model               | Link                                                           |
|---------------------|----------------------------------------------------------------|
| Qwen2-VL (~2B)      | [huggingface](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) |


## Train
The training script are "scripts/stage2_train.sh" and "scripts/stage3_train.sh". And you need to change following parameters:
1. **OUTPUT** :refers to the save directory for training, which must include the keyword "qwen2"(and optionally "lora"). If LoRA training is used, the name must include "lora" (e.g., "qwen2_lora").
2. **task_name** :refers to the tasks used for training, which should be corresponded to "your_task_name" in aloha_scripts/constant.py
3. **model_name_or_path** :path to the pretrained VLM weights
Other hyperparameters like "batch_size", "save_steps" could be customized according to your computation resources.

Start training by following commands:
```shell
./scripts/stage2_train.sh
./scripts/stage3_train.sh
```

## Evaluation
You can refer to our evaluation script [smart_eval_agilex.py](https://github.com/lesjie-wen/dexvla/blob/main/evaluate/smart_eval_agilex.py).

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
