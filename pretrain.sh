# 预训练
accelerate launch\
     --mixed_precision bf16 \
     --num_machines 1 \
     --num_processes 8 \
     --gpu_ids 0,1,2,3,4,5,6,7 \
     --main_process_port 29666 \
     -m src.language_modeling.train \
     --config config/language_modeling/pretrain.yaml \
     --per_device_train_batch_size 12 \
     --gradient_accumulation_steps 4 \
     --model_name_or_path mistralai/mistral-7b-instruct-v0.2 \
     --retriever_name_or_path Salesforce/SFR-Embedding-Mistral
