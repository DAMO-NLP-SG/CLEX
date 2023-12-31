torchrun --nproc_per_node=8 \
--master_port=2349 \
train/train_lm.py \
    --model_name_or_path /path/to/llama2  \
    --data_path /path/to/lm_data.json \
    --bf16 True \
    --output_dir new_ckpts/llama2/tmp \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 250 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --ddp_find_unused_parameters True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --lazy_preprocess True \
    --do_train True \
    --do_eval False \
    --do_predict True \
    --log_scale False