export model_path=/path/to/clexmodel

for length in 256
do
torchrun --nproc_per_node=4 \
--master_port=2341 \
train/eval_lm.py \
    --model_name_or_path $model_path \
    --data_path /path/to/eval_data \
    --output_dir /path/to/save_dir \
    --bf16_full_eval \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --tf32 True \
    --model_max_length $(($length * 1024)) \
    --do_train False \
    --do_eval False \
    --do_predict True \
    --log_scale True
done
