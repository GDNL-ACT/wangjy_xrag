#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
from transformers import AutoTokenizer
import numpy as np
from collections import Counter
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate ROUGE-L scores from evaluation results")
    
    parser.add_argument("--input_file", type=str, required=True, 
                       help="Input JSON file with evaluation results")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Output JSON file with ROUGE-L scores")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                       help="Path to the tokenizer (model checkpoint or HuggingFace model)")
    parser.add_argument("--use_fast_tokenizer", type=eval, default=True,
                       help="Whether to use fast tokenizer")
    
    return parser.parse_args()

def load_tokenizer(tokenizer_path, use_fast=True):
    """加载分词器"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=use_fast)
        logger.info(f"Loaded tokenizer from {tokenizer_path}")
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise

def tokenize_text(text, tokenizer):
    """使用模型分词器对文本进行分词"""
    if not text or not text.strip():
        return []
    
    # 使用tokenizer编码然后解码每个token
    tokens = tokenizer.tokenize(text)
    # 过滤掉特殊token
    tokens = [token for token in tokens if not token.startswith('<') or not token.endswith('>')]
    return tokens

def lcs_length(X, Y):
    """计算最长公共子序列的长度"""
    m, n = len(X), len(Y)
    
    # 创建DP表
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 填充DP表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i-1] == Y[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

def compute_rouge_l(prediction, reference, tokenizer):
    """计算ROUGE-L得分"""
    if not prediction.strip() or not reference.strip():
        return {
            'rouge_l_precision': 0.0,
            'rouge_l_recall': 0.0,
            'rouge_l_f1': 0.0
        }
    
    # 使用模型tokenizer进行分词
    pred_tokens = tokenize_text(prediction, tokenizer)
    ref_tokens = tokenize_text(reference, tokenizer)
    
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return {
            'rouge_l_precision': 0.0,
            'rouge_l_recall': 0.0,
            'rouge_l_f1': 0.0
        }
    
    # 计算最长公共子序列长度
    lcs_len = lcs_length(pred_tokens, ref_tokens)
    
    # 计算精确率和召回率
    precision = lcs_len / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
    recall = lcs_len / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
    
    # 计算F1分数
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    return {
        'rouge_l_precision': precision,
        'rouge_l_recall': recall,
        'rouge_l_f1': f1
    }

def calculate_rouge_scores(input_data, tokenizer):
    """为所有样本计算ROUGE-L得分"""
    samples = input_data.get('samples', [])
    
    logger.info(f"Calculating ROUGE-L scores for {len(samples)} samples...")
    
    updated_samples = []
    rouge_l_scores = []
    
    for i, sample in enumerate(samples):
        if i % 100 == 0:
            logger.info(f"Processed {i}/{len(samples)} samples")
        
        # 复制原始样本数据
        updated_sample = sample.copy()
        
        # 只为成功生成的样本计算ROUGE-L
        if sample.get('generation_success', False):
            prediction = sample.get('prediction', '')
            reference = sample.get('reference', '')
            
            rouge_scores = compute_rouge_l(prediction, reference, tokenizer)
            updated_sample.update(rouge_scores)
            rouge_l_scores.append(rouge_scores['rouge_l_f1'])
        else:
            # 失败的样本设置为0
            updated_sample.update({
                'rouge_l_precision': 0.0,
                'rouge_l_recall': 0.0,
                'rouge_l_f1': 0.0
            })
            rouge_l_scores.append(0.0)
        
        updated_samples.append(updated_sample)
    
    # 计算总体统计
    successful_rouge_scores = [score for score in rouge_l_scores if score > 0]
    
    if successful_rouge_scores:
        avg_rouge_l = np.mean(successful_rouge_scores)
        std_rouge_l = np.std(successful_rouge_scores)
        max_rouge_l = np.max(successful_rouge_scores)
        min_rouge_l = np.min(successful_rouge_scores)
    else:
        avg_rouge_l = std_rouge_l = max_rouge_l = min_rouge_l = 0.0
    
    # 更新统计信息
    updated_summary = input_data.get('summary', {}).copy()
    updated_summary.update({
        'average_rouge_l': avg_rouge_l,
        'std_rouge_l': std_rouge_l,
        'max_rouge_l': max_rouge_l,
        'min_rouge_l': min_rouge_l
    })
    
    return {
        'evaluation_config': input_data.get('evaluation_config', {}),
        'summary': updated_summary,
        'samples': updated_samples
    }

def main():
    args = parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_file):
        logger.error(f"Input file {args.input_file} does not exist")
        return
    
    # 加载分词器
    logger.info("Loading tokenizer...")
    tokenizer = load_tokenizer(args.tokenizer_path, args.use_fast_tokenizer)
    
    # 读取输入JSON文件
    logger.info(f"Loading evaluation results from {args.input_file}")
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load input file: {e}")
        return
    
    # 验证输入数据格式
    if 'samples' not in input_data:
        logger.error("Input file does not contain 'samples' field")
        return
    
    logger.info(f"Found {len(input_data['samples'])} samples in input file")
    
    # 计算ROUGE-L得分
    logger.info("Calculating ROUGE-L scores...")
    updated_data = calculate_rouge_scores(input_data, tokenizer)
    
    # 添加ROUGE计算的配置信息
    updated_data['rouge_config'] = {
        'tokenizer_path': args.tokenizer_path,
        'use_fast_tokenizer': args.use_fast_tokenizer,
        'metric': 'ROUGE-L'
    }
    
    # 保存结果
    logger.info(f"Saving results to {args.output_file}")
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(updated_data, f, indent=2, ensure_ascii=False)
        logger.info("Results saved successfully")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        return
    
    # 输出统计信息
    logger.info("***** ROUGE-L Results *****")
    summary = updated_data['summary']
    logger.info(f"Total samples: {summary.get('total_samples', 0)}")
    logger.info(f"Successful samples: {summary.get('successful_samples', 0)}")
    logger.info(f"Average ROUGE-L F1: {summary.get('average_rouge_l', 0):.4f}")
    logger.info(f"ROUGE-L F1 Std: {summary.get('std_rouge_l', 0):.4f}")
    logger.info(f"Max ROUGE-L F1: {summary.get('max_rouge_l', 0):.4f}")
    logger.info(f"Min ROUGE-L F1: {summary.get('min_rouge_l', 0):.4f}")
    
    # 显示一些示例
    logger.info("\n***** Sample ROUGE-L Scores *****")
    successful_samples = [s for s in updated_data['samples'] if s.get('generation_success', False)]
    for i, sample in enumerate(successful_samples[:3]):
        logger.info(f"\n--- Sample {sample.get('sample_id', i)} ---")
        logger.info(f"Input: {sample.get('input', '')[:100]}...")
        logger.info(f"Reference: {sample.get('reference', '')}")
        logger.info(f"Prediction: {sample.get('prediction', '')}")
        logger.info(f"BLEU Score: {sample.get('bleu_score', 0):.4f}")
        logger.info(f"ROUGE-L Precision: {sample.get('rouge_l_precision', 0):.4f}")
        logger.info(f"ROUGE-L Recall: {sample.get('rouge_l_recall', 0):.4f}")
        logger.info(f"ROUGE-L F1: {sample.get('rouge_l_f1', 0):.4f}")

if __name__ == "__main__":
    main()
