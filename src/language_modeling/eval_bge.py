#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import json
import torch
import torch.distributed as dist
from functools import partial
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from transformers import AutoTokenizer, LlamaTokenizer, LlamaTokenizerFast
from tokenizers import AddedToken
import numpy as np
from collections import Counter
import re

# 导入自定义模块
from src.model import (
    XMistralForCausalLM,
    XMistralConfig,
    XMixtralForCausalLM,
    XMixtralConfig,
    BGE,
)

from src.language_modeling.utils import (
    XRAG_TOKEN,
    get_retrieval_embeds,
)

from src.language_modeling.preprocessing import (
    encode_with_chat_format_pretrain,
)

from src.utils import get_yaml_file

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate pretrained model with BLEU score")
    
    # 基本配置
    parser.add_argument("--config", type=str, required=True, help="config file to launch the evaluation")
    parser.add_argument("--checkpoint_path", type=str, default=None, 
                       help="path to the checkpoint directory")
    parser.add_argument("--workdir", type=str, help="working directory")
    parser.add_argument("--dev_file", type=str, default=None, help="A csv or a json file containing the dev data.")
    
    # 模型配置
    parser.add_argument("--base_model", help='base LLM load')
    parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model or model identifier")
    parser.add_argument("--retriever_name_or_path", type=str, help="Path to pretrained retriever model")
    parser.add_argument("--use_fast_tokenizer", type=eval, default=True)
    parser.add_argument("--use_flash_attn", type=eval, default=False)
    parser.add_argument("--chat_format", choices=['mistral','tulu','mixtral','qwen','yi','gemma'])
    
    # 生成配置
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for nucleus sampling")
    parser.add_argument("--do_sample", type=eval, default=False, help="Whether to use sampling for generation")
    
    # 评估配置
    parser.add_argument("--per_device_eval_batch_size", type=int, default=None, help="Batch size for evaluation")
    parser.add_argument("--retrieval_context_length", type=int, default=None, help="Max token number for retriever")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Maximum number of samples to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--preprocessing_num_workers", type=int, default=4)
    parser.add_argument("--overwrite_cache", type=eval, default=False)
    
    # 输出配置
    parser.add_argument("--output_file", type=str, default="eval_results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    # 从YAML配置文件加载配置
    if args.config:
        yaml_config = get_yaml_file(args.config)
        for k, v in yaml_config.items():
            if hasattr(args, k) and getattr(args, k) is None:
                setattr(args, k, v)
    
    # 处理路径
    if args.workdir:
        if args.dev_file and not os.path.isabs(args.dev_file):
            args.dev_file = os.path.join(args.workdir, args.dev_file)
        if args.retriever_name_or_path and os.path.isdir(args.retriever_name_or_path):
            args.retriever_name_or_path = os.path.join(args.workdir, args.retriever_name_or_path)
        if not os.path.isabs(args.checkpoint_path):
            args.checkpoint_path = os.path.join(args.workdir, args.checkpoint_path)
    
    return args

def collator(samples, llm_tokenizer, retriever_tokenizer=None, retrieval_context_length=180):
    """数据整理函数，与训练时保持一致"""
    
    def padding(input_ids, labels=None, padding_side='right'):
        def _padding(ids, padding_value, padding_side='right'):
            if padding_side == 'right':
                return torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=padding_value)
            elif padding_side == 'left':
                flipped_ids = [torch.flip(x, dims=[0]) for x in ids]  
                return torch.flip(
                    torch.nn.utils.rnn.pad_sequence(flipped_ids, batch_first=True, padding_value=padding_value),
                    dims=[1],
                )
        
        input_ids = _padding(input_ids, padding_value=llm_tokenizer.pad_token_id, padding_side=padding_side)
        attention_mask = (input_ids != llm_tokenizer.pad_token_id).long()
        if labels is not None:
            labels = _padding(labels, padding_value=-100, padding_side=padding_side)
        return input_ids, attention_mask, labels

    xrag_input_ids, xrag_attention_mask, xrag_labels = padding(
        input_ids=[x['xrag_input_ids'] for x in samples],
        labels=[x['xrag_labels'] for x in samples] if 'xrag_labels' in samples[0].keys() else None,
        padding_side=llm_tokenizer.padding_side
    )

    ret = {
        "xrag_input_ids": xrag_input_ids,
        "xrag_attention_mask": xrag_attention_mask,
        "xrag_labels": xrag_labels,
    }

    if 'retriever_input_text' in samples[0].keys():
        retriever_input_text = [x['retriever_input_text'] for x in samples]
        assert isinstance(retriever_input_text[0], list)
        retriever_input_text = [x for y in retriever_input_text for x in y]
        
        # 处理不同检索器的分词问题
        if retriever_tokenizer.name_or_path == "intfloat/e5-large-v2":
            retriever_input_text = ["passage: " + x for x in retriever_input_text]
        elif retriever_tokenizer.name_or_path == 'intfloat/e5-mistral-7b-instruct':
            retriever_input_text = [x + retriever_tokenizer.eos_token for x in retriever_input_text]

        tokenized_retrieval_text = retriever_tokenizer(
            retriever_input_text, 
            max_length=retrieval_context_length,
            padding=True, truncation=True, return_tensors="pt"
        )
        
        ret['retriever_input_ids'] = tokenized_retrieval_text['input_ids']
        ret['retriever_attention_mask'] = tokenized_retrieval_text['attention_mask']
    
    return ret

def simple_tokenize(text):
    """简单的分词函数，使用正则表达式"""
    # 将文本转换为小写并使用正则表达式分词
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens

def compute_single_bleu_score(prediction, reference):
    """计算单个样本的BLEU分数"""
    pred_tokens = simple_tokenize(prediction)
    ref_tokens = simple_tokenize(reference)
    
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    
    # 计算n-gram精度 (BLEU-4)
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    def calculate_precision(pred_tokens, ref_tokens, n):
        pred_ngrams = get_ngrams(pred_tokens, n)
        ref_ngrams = get_ngrams(ref_tokens, n)
        
        if len(pred_ngrams) == 0:
            return 0.0
        
        pred_counter = Counter(pred_ngrams)
        ref_counter = Counter(ref_ngrams)
        
        overlap = 0
        for ngram in pred_counter:
            overlap += min(pred_counter[ngram], ref_counter[ngram])
        
        return overlap / len(pred_ngrams)
    
    # 计算1-4gram精度
    precisions = []
    for n in range(1, 5):
        prec = calculate_precision(pred_tokens, ref_tokens, n)
        precisions.append(prec)
    
    # 如果所有精度都为0，BLEU为0
    if all(p == 0 for p in precisions):
        return 0.0
    
    # 计算几何平均
    log_sum = sum(np.log(p) if p > 0 else -float('inf') for p in precisions)
    if log_sum == -float('inf'):
        geometric_mean = 0.0
    else:
        geometric_mean = np.exp(log_sum / 4)
    
    # 简化的长度惩罚
    bp = min(1.0, np.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1)))
    
    bleu = bp * geometric_mean
    return bleu

@torch.no_grad()
def evaluate_model(model, dataloader, tokenizer, retriever, accelerator, args):
    """评估模型"""
    model.eval()
    
    all_results = []  # 存储每条数据的完整信息
    
    progress_bar = tqdm(dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process)
    
    sample_idx = 0
    for batch_idx, batch in enumerate(progress_bar):
        try:
            # 获取检索嵌入
            retrieval_kwargs = {}
            if retriever is not None:
                retrieval_kwargs['retrieval_embeds'] = get_retrieval_embeds(
                    model=retriever,
                    input_ids=batch['retriever_input_ids'],
                    attention_mask=batch['retriever_attention_mask'],
                )
            
            # 准备输入（去除标签部分，只保留prompt）
            input_ids = batch['xrag_input_ids']
            attention_mask = batch['xrag_attention_mask']
            labels = batch['xrag_labels']
            
            # 找到每个样本中第一个非-100的标签位置作为生成起始点
            generation_start_indices = []
            for i in range(labels.shape[0]):
                valid_label_mask = labels[i] != -100
                if valid_label_mask.any():
                    first_valid_idx = torch.where(valid_label_mask)[0][0].item()
                    generation_start_indices.append(first_valid_idx)
                else:
                    generation_start_indices.append(input_ids.shape[1])
            
            # 为每个样本准备输入和生成
            batch_size = input_ids.shape[0]
            
            for i in range(batch_size):
                sample_result = {
                    "sample_id": sample_idx,
                    "input": "",
                    "reference": "",
                    "prediction": "",
                    "bleu_score": 0.0,
                    "generation_success": False,
                    "error_message": None
                }
                
                try:
                    # 获取prompt部分
                    start_idx = generation_start_indices[i]
                    prompt_ids = input_ids[i:i+1, :start_idx]
                    prompt_attention = attention_mask[i:i+1, :start_idx]
                    
                    # 获取输入文本
                    input_text = tokenizer.decode(prompt_ids[0], skip_special_tokens=True).strip()
                    sample_result["input"] = input_text
                    
                    # 获取参考答案
                    reference_ids = input_ids[i][labels[i] != -100]
                    reference_text = tokenizer.decode(reference_ids, skip_special_tokens=True).strip()
                    sample_result["reference"] = reference_text
                    
                    # 准备检索嵌入（如果需要）
                    sample_retrieval_kwargs = {}
                    if 'retrieval_embeds' in retrieval_kwargs:
                        # 假设每个样本对应一个检索嵌入
                        sample_retrieval_kwargs['retrieval_embeds'] = retrieval_kwargs['retrieval_embeds'][i:i+1]
                    
                    # 使用unwrap_model来获取原始模型进行生成
                    unwrapped_model = accelerator.unwrap_model(model)
                    
                    # 生成文本
                    generated_ids = unwrapped_model.generate(
                        input_ids=prompt_ids,
                        attention_mask=prompt_attention,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature if args.do_sample else None,
                        top_p=args.top_p if args.do_sample else None,
                        do_sample=args.do_sample,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        **sample_retrieval_kwargs
                    )
                    
                    # 直接解码生成的文本（不需要截取，因为generate输出的就是新token）
                    generated_text = tokenizer.decode(
                        generated_ids[0], 
                        skip_special_tokens=True
                    ).strip()
                    
                    sample_result["prediction"] = generated_text
                    sample_result["generation_success"] = True
                    
                    # 计算BLEU分数
                    bleu_score = compute_single_bleu_score(generated_text, reference_text)
                    sample_result["bleu_score"] = bleu_score
                    
                except Exception as e:
                    error_msg = f"Generation failed for sample {sample_idx}: {str(e)}"
                    logger.warning(error_msg)
                    sample_result["error_message"] = error_msg
                
                all_results.append(sample_result)
                sample_idx += 1
                
        except Exception as e:
            logger.error(f"Batch {batch_idx} failed: {str(e)}")
            # 添加空结果以保持一致性
            batch_size = batch['xrag_input_ids'].shape[0]
            for i in range(batch_size):
                sample_result = {
                    "sample_id": sample_idx,
                    "input": "",
                    "reference": "",
                    "prediction": "",
                    "bleu_score": 0.0,
                    "generation_success": False,
                    "error_message": f"Batch processing failed: {str(e)}"
                }
                all_results.append(sample_result)
                sample_idx += 1
    
    # 收集所有进程的结果
    if accelerator.num_processes > 1:
        # 使用gather_for_metrics来收集结果
        gathered_results = accelerator.gather_for_metrics(all_results)
        
        if accelerator.is_main_process:
            all_results = gathered_results
    
    # 只在主进程计算统计信息
    if accelerator.is_main_process:
        # 计算总体统计
        successful_samples = [r for r in all_results if r["generation_success"]]
        bleu_scores = [r["bleu_score"] for r in successful_samples]
        
        if bleu_scores:
            avg_bleu = np.mean(bleu_scores)
            std_bleu = np.std(bleu_scores)
            max_bleu = np.max(bleu_scores)
            min_bleu = np.min(bleu_scores)
        else:
            avg_bleu = std_bleu = max_bleu = min_bleu = 0.0
        
        summary_stats = {
            "total_samples": len(all_results),
            "successful_samples": len(successful_samples),
            "failed_samples": len(all_results) - len(successful_samples),
            "success_rate": len(successful_samples) / len(all_results) if all_results else 0.0,
            "average_bleu": avg_bleu,
            "std_bleu": std_bleu,
            "max_bleu": max_bleu,
            "min_bleu": min_bleu
        }
        
        return {
            "summary": summary_stats,
            "samples": all_results
        }
    else:
        return None

def main():
    args = parse_args()
    set_seed(args.seed)
    
    accelerator = Accelerator()
    
    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=True)
    
    # 加载检索器
    retriever = None
    retriever_hidden_size = -1
    retrieval_embed_length = 0
    retriever_tokenizer = None
    
    if args.retriever_name_or_path is not None:
        retriever = BGE(
            args.retriever_name_or_path, 
            device=accelerator.device
        )
        retriever_tokenizer = AutoTokenizer.from_pretrained(args.retriever_name_or_path)
        
        retrieval_embed_length = retriever.get_embed_length()
        retriever_hidden_size = retriever.get_embed_dim()
    
    # 加载数据集
    if args.dev_file is None:
        raise ValueError("dev_file must be provided for evaluation")
    
    raw_datasets = load_dataset("json", data_files={"dev": args.dev_file})
    
    # 限制评估样本数量
    if args.max_eval_samples is not None and len(raw_datasets['dev']) > args.max_eval_samples:
        raw_datasets['dev'] = raw_datasets['dev'].select(range(args.max_eval_samples))
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path if os.path.exists(args.checkpoint_path) else args.model_name_or_path,
        use_fast=args.use_fast_tokenizer,
    )
    
    # 设置模型类
    if args.chat_format == 'mixtral':
        MODEL_CLASS, CONFIG_CLASS = XMixtralForCausalLM, XMixtralConfig
        tokenizer.padding_side = 'left'    
    elif args.chat_format == 'mistral':
        MODEL_CLASS, CONFIG_CLASS = XMistralForCausalLM, XMistralConfig
        tokenizer.padding_side = 'left'
    
    # 加载模型配置和模型
    config = CONFIG_CLASS.from_pretrained(
        args.checkpoint_path if os.path.exists(args.checkpoint_path) else args.model_name_or_path,
        retriever_hidden_size=retriever_hidden_size
    )
    
    model = MODEL_CLASS.from_pretrained(
        args.checkpoint_path if os.path.exists(args.checkpoint_path) else args.model_name_or_path,
        config=config,
        use_flash_attention_2=args.use_flash_attn,
        torch_dtype=torch.bfloat16 if accelerator.mixed_precision == 'bf16' else 'auto',
        device_map=accelerator.device
    )
    
    # 添加特殊token
    num_added_tokens = 0
    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens({"pad_token": "<pad>"})
    
    num_added_tokens += tokenizer.add_tokens([AddedToken(XRAG_TOKEN, lstrip=False, rstrip=False)])
    xrag_token_id = tokenizer.convert_tokens_to_ids(XRAG_TOKEN)
    model.set_xrag_token_id(xrag_token_id)
    
    if num_added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
    
    # 数据预处理
    encode_function = partial(
        encode_with_chat_format_pretrain,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        retrieval_embed_length=retrieval_embed_length,
        chat_format=args.chat_format,
    )
    
    with accelerator.main_process_first():
        lm_datasets = raw_datasets.map(
            encode_function,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            remove_columns=[name for name in raw_datasets["dev"].column_names 
                          if name not in ["input_ids", "labels", "attention_mask"]],
            desc="Tokenizing and reformatting data",
        )
        lm_datasets.set_format(type="pt")
    
    eval_dataset = lm_datasets["dev"]
    
    # 创建数据加载器
    collate_fn = partial(
        collator,
        llm_tokenizer=tokenizer,
        retriever_tokenizer=retriever_tokenizer,
        retrieval_context_length=args.retrieval_context_length,
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.per_device_eval_batch_size
    )
    
    # 准备模型和数据加载器
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
    
    logger.info("***** Running evaluation *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Batch size = {args.per_device_eval_batch_size}")
    
    # 开始评估
    eval_results = evaluate_model(model, eval_dataloader, tokenizer, retriever, accelerator, args)
    
    # 输出结果（只在主进程）
    if accelerator.is_main_process and eval_results is not None:
        logger.info("***** Evaluation Results *****")
        logger.info(f"Total samples: {eval_results['summary']['total_samples']}")
        logger.info(f"Successful samples: {eval_results['summary']['successful_samples']}")
        logger.info(f"Success rate: {eval_results['summary']['success_rate']:.2%}")
        logger.info(f"Average BLEU Score: {eval_results['summary']['average_bleu']:.4f}")
        logger.info(f"BLEU Score Std: {eval_results['summary']['std_bleu']:.4f}")
        logger.info(f"Max BLEU Score: {eval_results['summary']['max_bleu']:.4f}")
        logger.info(f"Min BLEU Score: {eval_results['summary']['min_bleu']:.4f}")
        
        # 保存结果
        output_data = {
            "evaluation_config": {
                "checkpoint_path": args.checkpoint_path,
                "dev_file": args.dev_file,
                "max_eval_samples": args.max_eval_samples,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "do_sample": args.do_sample,
                "per_device_eval_batch_size": args.per_device_eval_batch_size
            },
            "summary": eval_results['summary'],
            "samples": eval_results['samples']
        }
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {args.output_file}")
        
        # 显示一些示例
        logger.info("\n***** Sample Results *****")
        successful_samples = [s for s in eval_results['samples'] if s['generation_success']]
        for i, sample in enumerate(successful_samples[:3]):
            logger.info(f"\n--- Sample {sample['sample_id']} ---")
            logger.info(f"Input: {sample['input'][:200]}...")
            logger.info(f"Reference: {sample['reference']}")
            logger.info(f"Prediction: {sample['prediction']}")
            logger.info(f"BLEU Score: {sample['bleu_score']:.4f}")

if __name__ == "__main__":
    main()
