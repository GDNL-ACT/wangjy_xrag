#!/bin/bash

# 定义数据集列表
datasets=("nq_open" "hotpotqa" "triviaqa" "webqa" "truthfulqa" "factkg")

# 定义GPU列表（根据你的实际GPU数量调整）
gpus=(4 5 6 7)

# 获取GPU数量
num_gpus=${#gpus[@]}

# 循环执行每个数据集的测试
for i in "${!datasets[@]}"; do
    dataset=${datasets[$i]}
    # 轮流选择GPU
    gpu_id=${gpus[$((i % num_gpus))]}
    
    echo "Testing dataset: $dataset on GPU: $gpu_id"
    
    # 在后台运行测试
    CUDA_VISIBLE_DEVICES=$gpu_id python -m src.eval.run_eval \
        --data $dataset \
        --model_name_or_path wandb/latest-run/files/checkpoint/last \
        --retriever_name_or_path Salesforce/SFR-Embedding-Mistral \
        --use_rag \
        --save_dir eval_res/SFR-Embedding-Mistral &
    
    # 可选：添加延迟避免同时启动过多进程
    sleep 5
done

# 等待所有后台任务完成
wait

echo "All evaluations completed!"
