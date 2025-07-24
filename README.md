# xRAG
## 运行说明
1. 创建conda环境：
```
conda create -n xrag python=3.9 -y
conda activate xrag
```

2. 安装依赖：
```
pip install -r requirements.txt
```
由于flash-attn构建较慢，使用预下载的whl包安装，如有版本问题可下载对应版本：（https://github.com/Dao-AILab/flash-attention/releases?expanded=true&page=5&q=）

3. 配置环境变量：
```
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$PWD/huggingface  # huggingface存储位置
export WANDB_MODE=offline
export WANDB_DIR=$PWD/wandb  # wandb存储位置
```

4. 下载预训练的wiki数据集并分割，下载结果存储在data/pretrain/wikipedia中；下载指令微调的数据集，下载内容存储在huggingface缓存中，最终数据集存储在data/instruction_tuning/processed/context_aware_instrution_tuning_data.jsonl：
```
bash prepare_data.sh
```

5. 预训练，参数中包含使用的卡数、batch size、梯度积累等：
```
bash pretrain.sh
```
训练日志和checkpoints由wandb保存在相关目录中，tokenize结果会缓存在huggingface目录中。

6. 指令微调，默认加载wandb记录的最新训练的预训练模型，可手动更改model_name_or_path指定模型
```
bash instruction_tuning.sh
```

7. 模型评估，多卡跑多个测试任务，结果存在eval_res/SFR-Embedding-Mistral下，不过目前数据集还需要重新选择：
```
bash eval_all.sh
```


以下为原始readme
## Introduction
Official repo for [xRAG: Extreme Context Compression for Retrieval-augmented Generation with One Token](https://arxiv.org/abs/2405.13792)

## Get Started
Refer to `Dockerfile` for required packages

Configure `wandb` and `accelerate`
```bash
wandb login
accelerate config
```

## Pretrained Checkpoints
HuggingFace
| Model                 | Backbone | Download                                                                    |
|-----------------------|-----------------|-----------------------------------------------------------------------------|
| xRAG-7b | [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)            | [🤗 Hugging Face](https://huggingface.co/Hannibal046/xrag-7b) |
| xRAG-MoE | [mistralai/Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)            | [🤗 Hugging Face](https://huggingface.co/Hannibal046/xrag-moe) |


## Tutorial

We provide a tutorial for xRAG in `tutorial.ipynb`. Check it out!

## Data
- download [enwiki-dec2021](https://github.com/facebookresearch/atlas?tab=readme-ov-file#models) as pretraining data and corpus for retrieval
- prepare instruction tuning data in `prepare_data.ipynb`
- download [TriviaQA](https://drive.google.com/drive/folders/1lFFTklW_0HuR53hLpFdLClgfSAhXn_2f)
- using [ColBERT-v2](https://github.com/stanford-futuredata/ColBERT.git) to conduct retrieval

## Training
Training scripts in `scripts/`, for example, to train a Mistral-7b with SFR:
```bash
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes 8 \
    --main_process_port 29666 \
    -m \
    src.language_modeling.train \
    --config config/language_modeling/pretrain.yaml \
```
## Evaluation
The evaluation code is in `src/eval`. For example, to evaluate on TriviaQA:

without retrieval augmentation:
```bash
CUDA_VISIBLE_DEVICES=0 python -m src.eval.run_eval \
        --data triviaqa \
        --model_name_or_path Hannibal046/xrag-7b
```

with retrieval augmentation:
```bash
CUDA_VISIBLE_DEVICES=0 python -m src.eval.run_eval \
        --data triviaqa \
        --model_name_or_path Hannibal046/xrag-7b \
        --use_rag
```

with xRAG:
```bash
CUDA_VISIBLE_DEVICES=0 python -m src.eval.run_eval \
        --data triviaqa \
        --model_name_or_path Hannibal046/xrag-7b \
        --retriever_name_or_path /data1/models/BAAI/bge-m3 \
        --use_rag
```

## Benchmark
To benchmark xRAG, we provide the code in `src/language_modeling/profiler.py`.
```
python -m src.language_modeling.profiler --instruction_length 54 --generation_length 30 --dataset triviaqa --use_xrag
python -m src.language_modeling.profiler --instruction_length 54 --generation_length 30 --dataset triviaqa
```
