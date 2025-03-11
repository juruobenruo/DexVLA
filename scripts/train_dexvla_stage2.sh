#!/bin/bash
LLM=qwen2_vl
LLM_MODEL_SIZE=2B

ACTION_HEAD=scale_dp_policy  #unet_diffusion_policy or scale_dp_policy

DIT_PRETRAIN=/path/to/pretrained/ScaleDP
MNOP=/path/to/pretrained/qwen2_vl # official qwen2_vl weights

TASKNAME=example_tasks

OUTPUT=/path/to/save/dir

deepspeed --master_port 29604 --num_gpus=8 --num_nodes=1 ./train_vla.py \
  --deepspeed scripts/zero2.json \
  --use_reasoning True \
  --lora_enable False \
  --action_dim 14 \
  --state_dim 14 \
  --flash_attn True \
  --chunk_size 50 \
  --load_pretrain_dit True \
  --pretrain_dit_path $DIT_PRETRAIN \
  --policy_head_type $ACTION_HEAD \
  --policy_head_size "ScaleDP_H" \
  --image_size_stable "(320,240)" \
  --image_size_wrist "(320,240)" \
  --task_name ${TASKNAME} \
  --model_name_or_path $MNOP \
  --version v0 \
  --tune_mm_mlp_adapter True \
  --freeze_vision_tower False \
  --freeze_backbone False \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --bf16 True \
  --output_dir $OUTPUT \
  --max_steps 100000 \
  --per_device_train_batch_size 12 \
  --gradient_accumulation_steps 1 \
  --save_strategy "steps" \
  --save_steps 10000 \
  --save_total_limit 50 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "constant" \
  --logging_steps 50 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 8 \
  --lazy_preprocess True \
  --policy_class $ACTION_HEAD \
  --concat "token_cat" \
  --report_to tensorboard \
  --logging_dir $OUTPUT/log | tee $OUTPUT/log.log

for dir in "$OUTPUT"/*/ ; do
    if [[ "$(basename "$dir")" == *"checkpoint"* ]]; then
        cp ${MNOP}/preprocessor_config.json $dir
        cp ${MNOP}/chat_template.json $dir
    fi
done
echo $OUTPUT
