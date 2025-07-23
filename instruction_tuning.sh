# 指令微调
accelerate launch\
     --mixed_precision bf16 \
     --num_machines 1 \
     --num_processes 8 \
     --gpu_ids 0,1,2,3,4,5,6,7 \
     --main_process_port 29666 \
     -m src.language_modeling.train \
     --config config/language_modeling/finetune.yaml \
     --per_device_train_batch_size 4 \
     --gradient_accumulation_steps 2 \
     --model_name_or_path wandb/latest-run/files/checkpoint/last \
     --retriever_name_or_path Salesforce/SFR-Embedding-Mistral
