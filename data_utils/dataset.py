from dataclasses import dataclass
from typing import Dict, Sequence
import torch
import transformers
import gc
import numpy as np
import os

@dataclass
class Qwen2VLADataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    multimodal_processor: transformers.AutoProcessor=None
    computed_type: torch.dtype=None
    tokenizer: transformers.AutoTokenizer=None
    video: bool=False

    # @profile
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.flip(instance['input_ids'].squeeze(0), dims=[0]) for instance in instances]
        attention_mask = [torch.flip(instance['attention_mask'].squeeze(0), dims=[0]) for instance in instances]
        labels = [torch.flip(instance['labels'].squeeze(0), dims=[0]) for instance in instances]
        if self.video:
            video_grid_thw = torch.stack([instances['video_grid_thw'] for instances in instances])
            pixel_values_videos = torch.stack([instances['pixel_values_videos'] for instances in instances])
            pixel_values = None
            image_grid_thw=None
        else:
            image_grid_thw = torch.stack([instances['image_grid_thw'] for instances in instances])
            pixel_values = torch.stack([instances['pixel_values'] for instances in instances])
            pixel_values_videos = None
            video_grid_thw = None

        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=-100)
        labels = torch.flip(labels, dims=[1])
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids,
                                                    batch_first=True,
                                                    padding_value=self.tokenizer.pad_token_id)
        input_ids = torch.flip(input_ids, dims=[1])
        b = input_ids.shape[0]
        if self.video:
            video_grid_thw = video_grid_thw.reshape(b * video_grid_thw.shape[1], video_grid_thw.shape[2])
            pixel_values_videos = pixel_values_videos.reshape(b * pixel_values_videos.shape[1], pixel_values_videos.shape[2])

        else:
            image_grid_thw = image_grid_thw.reshape(b * image_grid_thw.shape[1], image_grid_thw.shape[2])
            pixel_values = pixel_values.reshape(b * pixel_values.shape[1], pixel_values.shape[2])

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
            
        if not isinstance(instances[0]['action'], torch.Tensor):
            actions = torch.tensor(np.array([instance['action'] for instance in instances]))
            states = torch.tensor(np.array([instance['state'] for instance in instances]))
        else:
            actions = torch.stack([instance['action'] for instance in instances])
            states = torch.stack([instance['state'] for instance in instances])

        is_pad_all = torch.stack([instance['is_pad'] for instance in instances])
        
        #exit(0)
        batch = dict(
            input_ids=input_ids,
            attention_mask=attention_mask[0],
            labels=labels,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            actions=actions,
            states=states,
            video_grid_thw=video_grid_thw,
            pixel_values=pixel_values,
            is_pad=is_pad_all
        )
        del input_ids
        del attention_mask
        del labels
        del pixel_values_videos
        del pixel_values
        del actions
        del states
        del video_grid_thw
        del image_grid_thw
        del is_pad_all
        gc.collect()
        torch.cuda.empty_cache()
        return batch
