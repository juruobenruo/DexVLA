<h1 align="center">
DexVLA: Vision-Language Model with Plug-In Diffusion Expert for Visuomotor Policy Learning</h1>


* **DexVLA: Vision-Language Model with Plug-In Diffusion Expert for Visuomotor Policy Learning** <br>
  [![arXiv](https://img.shields.io/badge/Arxiv-2402.03766-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2409.12514)
  


## 📰 News
* **`Nov. 19th, 2024`**: **DexVLA** is out! **Paper** can be found [here](https://arxiv.org/abs/2502.05855). The **project web** can be found [here](https://dex-vla.github.io/).

## Contents
- [Install](#install)
- [Data Preparation](#data-preparation)
- [Download Pretrained VLM](#Download-Pretrained-VLM)
- [Train](#train)
- [Evaluation](#evaluation)

## Install

1. Clone this repository and navigate to diffusion-vla folder
```bash
git clone https://github.com/lesjie-wen/tinyvla.git
```

2. Install Package
1. flash_attn
2. cd qwen2_vla pip install
3. cd policy_heads pip install
```Shell
conda create -n tinyvla python=3.10 -y
conda activate tinyvla
pip install --upgrade pip  # 
pip install -r requirements.txt
cd detr
pip install -e . 
# install llava-pythia
cd ../llava-pythia
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
2. You have to add one entry in [constants.py](https://github.com/lesjie-wen/tinyvla/blob/main/aloha_scripts/constants.py) to specify the path of your data as follows.
```python
    'your_task_name':{
        'dataset_dir': DATA_DIR + '/your_task_path', # define the path of the dataset
        'episode_len': 1000, #max length of the episode,
        'camera_names': ['front', 'wrist'] # define the camera names which are used as the key when reading data
    }
```
## Download Pretrained VLM
We construct the VLM backbone by integrating a series of tiny LLM([Pythia](https://github.com/EleutherAI/pythia)) into [Llava](https://github.com/haotian-liu/LLaVA) framework. We follow the standard training pipe line and data provided by [Llava](https://github.com/haotian-liu/LLaVA). All the weights of VLM used in our paper are listed as following: 

| Model               | Usage         | Link                                                           |
|---------------------|---------------|----------------------------------------------------------------|
| Llava-Pythia(~400M) | For TinyVLA-S | [huggingface](https://huggingface.co/lesjie/Llava-Pythia-400M) |
| Llava-Pythia(~700M) | For TinyVLA-B | [huggingface](https://huggingface.co/lesjie/Llava-Pythia-700M) |
| Llava-Pythia(~1.3B) | For TinyVLA-H | [huggingface](https://huggingface.co/lesjie/Llava-Pythia-1.3B) |


## Train
The training script is "scripts/train.sh". And you need to change following parameters:
1. **OUTPUT** :refers to the save dir for training
2. **task_name** :refers to the tasks used for training, which should be corresponded to "your_task_name" in aloha_scripts/constant.py
3. **model_name_or_path** :path to the pretrained VLM weights
Other hyperparameters like "batch_size", "save_steps" could be customized according to your computation resources.

Start training by following commands:
```shell
./scripts/train.sh
```

## Evaluation
Before evaluation, we provide a post process script to generate a usable and smaller weights.
The process script is "scripts/process_ckpts.sh". And you need to change following parameters:
1.  **source_dir** :path to trained VLA dir equals to **OUTPUT** in train.sh
2. **target_dir** :path to save processed VLA weights

You can refer to our evaluation script [eval_real_franka.py](https://github.com/lesjie-wen/tinyvla/blob/main/eval_real_franka.py).
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
