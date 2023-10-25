export model_path=/mnt/workspace/ckpts/CLEX/CLEX-7b-hf-v1

for length in 8 16
do
torchrun --nproc_per_node=4 \
--master_port=2341 \
train/train_lm.py \
    --model_name_or_path $model_path \
    --data_path /mnt/workspace/Projects/fastchat_rwkv/fastchat/data/sharegpt/sharegpt_final_en_new.json \
    --output_dir tmp/test_final_$length \
    --bf16_full_eval \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length $(($length * 1024)) \
    --do_train False \
    --do_eval False \
    --do_predict True \
    --log_scale True
done
