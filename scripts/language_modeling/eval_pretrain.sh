accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes 2 \
    --gpu_ids 4,5 \
    --main_process_port 29667 \
    eval.py \
    --config config/language_modeling/pretrain.yaml \
    --checkpoint_path ./wandb/latest-run/files/checkpoint/last \
    --max_eval_samples 100 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 128 \
    --output_file eval_results.json
