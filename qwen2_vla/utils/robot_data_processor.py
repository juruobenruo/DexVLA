from PIL import Image
import numpy as np
from torchvision.transforms.functional import to_pil_image, to_tensor
import torchvision.transforms as transforms
import torch
from qwen_vl_utils import process_vision_info
from qwen_vl_utils import *
class Qwen2VLAProcess:
    def __init__(
            self,
            language=None,
            tokenizer=None,
            max_seq_len=512,
            multimodal_processor=None,
            camera_names=None,
            data_args=None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.camera_names = camera_names
        # self.language = language
        self.multimodal_processor = multimodal_processor
        self.data_args = data_args

    def preprocess_image(self, image, size=224):
        # Model has been trained to handle images of different aspects ratios
        # resized to 224x224 in the range [-1, 1]. Bilinear and antialias resize
        # options are helpful to improve quality in some tasks.
        image = np.asarray(image)
        if image.ndim == 2:  # Convert image without last channel into greyscale.
            image = np.stack((image,) * 3, axis=-1)
        image = image[..., :3]  # Remove alpha layer.
        assert image.shape[-1] == 3

        image_pil = to_pil_image(image)

        # Step 2: Define the resize transformation
        resize_transform = transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR)

        # Step 3: Apply the resize transformation
        image_resized_pil = resize_transform(image_pil)

        # Step 4: Convert back to tensor if needed
        image_resized = to_tensor(image_resized_pil)
        return image.numpy() / 127.5 - 1.0  # [0, 255]->[-1,1]

    def qwen2_image_preprocess(self, each, camera_name):
        ele = {
        }
        each = Image.fromarray(each.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8))
        ele['image'] = each
        if 'wrist' in camera_name:
            w, h = eval(self.data_args.image_size_wrist)
            ele['resized_height'] = h
            ele['resized_width'] = w
        else:
            ele['resized_height'] = each.height
            ele['resized_width'] = each.width
        each = fetch_image(ele)
        return torch.from_numpy(np.array(each))

    def forward_process(self, sample, use_reasoning=True):
        video = False
        messages = self.datastruct_droid2llava(sample, video=video)

        data_dict = dict(
            messages=messages,
            images=None
        )

        image_data = torch.chunk(sample['image'], sample['image'].shape[0], 0)

        images_list = []

        for i, each in enumerate(image_data):
            img_pil = self.qwen2_image_preprocess(each, self.camera_names[i])
            images_list.append(img_pil)

        image_data = images_list
        video_inputs = None

        text = self.multimodal_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        model_inputs = self.multimodal_processor(
            text=text,
            images=image_data,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        input_labels = torch.ones_like(model_inputs['input_ids']) * -100
        if use_reasoning:
            answer = sample['reasoning'] + "Next action:" + '<|im_end|>'
        else:
            answer = '' + '<|im_end|>'

        output_text = self.tokenizer(answer, padding=True, return_tensors="pt")
        output_labels = output_text['input_ids']
        model_inputs['input_ids'] = torch.cat((model_inputs['input_ids'], output_text['input_ids']), dim=-1)
        model_inputs['attention_mask'] = torch.cat((model_inputs['attention_mask'], output_text['attention_mask']), dim=-1)
        labels = torch.cat((input_labels, output_labels), dim=-1)
        data_dict['state'] = sample['state']
        data_dict['action'] = sample['action']
        data_dict['is_pad'] = sample['is_pad']
        data_dict['labels'] = labels
        for k, v in model_inputs.items():
            data_dict[k] = v
        return data_dict

    def datastruct_droid2llava(self, sample, video=False):
        len_image = sample['image'].shape[0]

        messages = [
            {
                "role": "user",
                "content": [],
            },
        ]

        for i in range(len_image):
            if video:
                messages[0]['content'].append({
                    "type": "video",
                    "video": None,
                })
            else:
                messages[0]['content'].append({
                            "type": "image",
                            "image": None,
                        })
        messages[0]['content'].append({"type": "text", "text": f""})
        messages[0]['content'][-1]['text'] = sample['raw_lang']

        return messages