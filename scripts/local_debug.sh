#!/bin/bash
LLM=qwen2_vl
LLM_MODEL_SIZE=2B

ACTION_HEAD=dit_diffusion_policy  #act #unet_diffusion_policy dit_diffusion_policy

DIT_PRETRAIN=/path/to/pretrained/ScaleDP
MNOP=/media/rl/HDD/data/multi_head_train_results/aloha_qwen2_vla/qwen2_vl_2B/qwen2_vl_4_cameras_1_17_all_data_pretrain_4w_DiT_H_1_17_full_param_stage_1_50_raw_lang/checkpoint-60000
TASKNAME=example_tasks

OUTPUT=/home/rl/Downloads/output/test

python ./train_vla.py \
  --use_reasoning False \
  --lora_enable False \
  --action_dim 14 \
  --state_dim 14 \
  --flash_attn False \
  --chunk_size 50 \
  --lora_module "vit llm" \
  --using_film True \
  --using_ema False \
  --policy_head_type $ACTION_HEAD \
  --policy_head_size "DiT_H" \
  --image_size_stable "(320,240)" \
  --image_size_wrist "(320,240)" \
  --episode_first False \
  --task_name $TASKNAME \
  --model_name_or_path $MNOP \
  --version v0 \
  --tune_mm_mlp_adapter True \
  --freeze_vision_tower False \
  --freeze_backbone False \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length False \
  --bf16 True \
  --output_dir $OUTPUT \
  --max_steps 80000 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --save_strategy "steps" \
  --save_steps 5 \
  --save_total_limit 50 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.001 \
  --lr_scheduler_type "cosine" \
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

mv ./60030.log $OUTPUT
echo $OUTPUT