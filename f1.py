#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import re
from collections import Counter
import numpy as np
from transformers import AutoTokenizer
import torch

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate F1 scores from evaluation results")
    
    parser.add_argument("--input_file", type=str, required=True, 
                       help="Input JSON file containing evaluation results")
    parser.add_argument("--output_file", type=str, default="f1_results.json",
                       help="Output JSON file for F1 results")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                       help="Path to the tokenizer (checkpoint or model path)")
    parser.add_argument("--use_fast_tokenizer", type=eval, default=True,
                       help="Whether to use fast tokenizer")
    
    return parser.parse_args()

def load_tokenizer(tokenizer_path, use_fast=True):
    """加载分词器"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            use_fast=use_fast,
        )
        logger.info(f"Successfully loaded tokenizer from {tokenizer_path}")
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load tokenizer from {tokenizer_path}: {str(e)}")
        raise

def tokenize_text(text, tokenizer, method='model'):
    """使用不同方法对文本进行分词"""
    if method == 'model':
        # 使用模型分词器
        tokens = tokenizer.tokenize(text.lower())
        # 过滤掉特殊token
        tokens = [token for token in tokens if not token.startswith('<') or not token.endswith('>')]
        return tokens
    elif method == 'regex':
        # 使用正则表达式分词（备用方法）
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    else:
        raise ValueError(f"Unknown tokenization method: {method}")

def calculate_f1_score(prediction, reference, tokenizer, tokenization_method='model'):
    """计算单个样本的F1得分"""
    if not prediction.strip() or not reference.strip():
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'pred_tokens': 0,
            'ref_tokens': 0,
            'common_tokens': 0
        }
    
    try:
        # 分词
        pred_tokens = tokenize_text(prediction, tokenizer, tokenization_method)
        ref_tokens = tokenize_text(reference, tokenizer, tokenization_method)
        
        if len(pred_tokens) == 0 and len(ref_tokens) == 0:
            return {
                'precision': 1.0,
                'recall': 1.0,
                'f1': 1.0,
                'pred_tokens': 0,
                'ref_tokens': 0,
                'common_tokens': 0
            }
        
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'pred_tokens': len(pred_tokens),
                'ref_tokens': len(ref_tokens),
                'common_tokens': 0
            }
        
        # 计算token重叠
        pred_counter = Counter(pred_tokens)
        ref_counter = Counter(ref_tokens)
        
        # 计算共同token数量
        common_tokens = 0
        for token in pred_counter:
            if token in ref_counter:
                common_tokens += min(pred_counter[token], ref_counter[token])
        
        # 计算precision和recall
        precision = common_tokens / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
        recall = common_tokens / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
        
        # 计算F1得分
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'pred_tokens': len(pred_tokens),
            'ref_tokens': len(ref_tokens),
            'common_tokens': common_tokens
        }
        
    except Exception as e:
        logger.warning(f"Error calculating F1 for prediction: {str(e)}")
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'pred_tokens': 0,
            'ref_tokens': 0,
            'common_tokens': 0,
            'error': str(e)
        }

def calculate_exact_match(prediction, reference):
    """计算精确匹配得分"""
    pred_clean = prediction.strip().lower()
    ref_clean = reference.strip().lower()
    return 1.0 if pred_clean == ref_clean else 0.0

def calculate_token_level_metrics(prediction, reference, tokenizer):
    """计算token级别的详细指标"""
    pred_tokens = tokenize_text(prediction, tokenizer, 'model')
    ref_tokens = tokenize_text(reference, tokenizer, 'model')
    
    pred_set = set(pred_tokens)
    ref_set = set(ref_tokens)
    
    # 计算unique token的precision/recall
    if len(pred_set) == 0 and len(ref_set) == 0:
        unique_precision = unique_recall = unique_f1 = 1.0
    elif len(pred_set) == 0 or len(ref_set) == 0:
        unique_precision = unique_recall = unique_f1 = 0.0
    else:
        common_unique = len(pred_set & ref_set)
        unique_precision = common_unique / len(pred_set)
        unique_recall = common_unique / len(ref_set)
        unique_f1 = 2 * unique_precision * unique_recall / (unique_precision + unique_recall) if (unique_precision + unique_recall) > 0 else 0.0
    
    return {
        'unique_tokens_precision': unique_precision,
        'unique_tokens_recall': unique_recall,
        'unique_tokens_f1': unique_f1,
        'pred_unique_tokens': len(pred_set),
        'ref_unique_tokens': len(ref_set),
        'common_unique_tokens': len(pred_set & ref_set)
    }

def process_evaluation_results(input_data, tokenizer):
    """处理评估结果，计算F1得分"""
    samples = input_data.get('samples', [])
    
    if not samples:
        logger.error("No samples found in input data")
        return None
    
    logger.info(f"Processing {len(samples)} samples...")
    
    processed_samples = []
    all_f1_scores = []
    all_precision_scores = []
    all_recall_scores = []
    all_exact_match_scores = []
    all_unique_f1_scores = []
    
    successful_samples = 0
    
    for i, sample in enumerate(samples):
        if i % 100 == 0:
            logger.info(f"Processing sample {i+1}/{len(samples)}")
        
        sample_result = {
            'sample_id': sample.get('sample_id', i),
            'input': sample.get('input', ''),
            'reference': sample.get('reference', ''),
            'prediction': sample.get('prediction', ''),
            'generation_success': sample.get('generation_success', False),
            'bleu_score': sample.get('bleu_score', 0.0),
            'error_message': sample.get('error_message', None)
        }
        
        # 只对成功生成的样本计算F1
        if sample.get('generation_success', False):
            prediction = sample.get('prediction', '')
            reference = sample.get('reference', '')
            
            # 计算F1得分
            f1_result = calculate_f1_score(prediction, reference, tokenizer)
            sample_result.update(f1_result)
            
            # 计算精确匹配
            exact_match = calculate_exact_match(prediction, reference)
            sample_result['exact_match'] = exact_match
            
            # 计算token级别指标
            token_metrics = calculate_token_level_metrics(prediction, reference, tokenizer)
            sample_result.update(token_metrics)
            
            # 收集统计数据
            all_f1_scores.append(f1_result['f1'])
            all_precision_scores.append(f1_result['precision'])
            all_recall_scores.append(f1_result['recall'])
            all_exact_match_scores.append(exact_match)
            all_unique_f1_scores.append(token_metrics['unique_tokens_f1'])
            
            successful_samples += 1
        else:
            # 失败的样本设置默认值
            sample_result.update({
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'exact_match': 0.0,
                'pred_tokens': 0,
                'ref_tokens': 0,
                'common_tokens': 0,
                'unique_tokens_precision': 0.0,
                'unique_tokens_recall': 0.0,
                'unique_tokens_f1': 0.0,
                'pred_unique_tokens': 0,
                'ref_unique_tokens': 0,
                'common_unique_tokens': 0
            })
        
        processed_samples.append(sample_result)
    
    # 计算总体统计
    if all_f1_scores:
        summary_stats = {
            'total_samples': len(samples),
            'successful_samples': successful_samples,
            'failed_samples': len(samples) - successful_samples,
            'success_rate': successful_samples / len(samples),
            
            # F1相关统计
            'average_f1': np.mean(all_f1_scores),
            'std_f1': np.std(all_f1_scores),
            'max_f1': np.max(all_f1_scores),
            'min_f1': np.min(all_f1_scores),
            'median_f1': np.median(all_f1_scores),
            
            # Precision/Recall统计
            'average_precision': np.mean(all_precision_scores),
            'std_precision': np.std(all_precision_scores),
            'average_recall': np.mean(all_recall_scores),
            'std_recall': np.std(all_recall_scores),
            
            # 精确匹配统计
            'average_exact_match': np.mean(all_exact_match_scores),
            'exact_match_count': sum(all_exact_match_scores),
            
            # Unique token F1统计
            'average_unique_f1': np.mean(all_unique_f1_scores),
            'std_unique_f1': np.std(all_unique_f1_scores),
            
            # 原始BLEU统计（如果存在）
            'original_bleu_stats': input_data.get('summary', {})
        }
    else:
        summary_stats = {
            'total_samples': len(samples),
            'successful_samples': 0,
            'failed_samples': len(samples),
            'success_rate': 0.0,
            'average_f1': 0.0,
            'std_f1': 0.0,
            'max_f1': 0.0,
            'min_f1': 0.0,
            'median_f1': 0.0,
            'average_precision': 0.0,
            'std_precision': 0.0,
            'average_recall': 0.0,
            'std_recall': 0.0,
            'average_exact_match': 0.0,
            'exact_match_count': 0,
            'average_unique_f1': 0.0,
            'std_unique_f1': 0.0,
            'original_bleu_stats': input_data.get('summary', {})
        }
    
    return {
        'summary': summary_stats,
        'samples': processed_samples
    }

def main():
    args = parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        return
    
    # 加载分词器
    logger.info(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = load_tokenizer(args.tokenizer_path, args.use_fast_tokenizer)
    
    # 读取输入数据
    logger.info(f"Loading evaluation results from {args.input_file}")
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load input file: {str(e)}")
        return
    
    # 处理数据计算F1得分
    logger.info("Calculating F1 scores...")
    results = process_evaluation_results(input_data, tokenizer)
    
    if results is None:
        logger.error("Failed to process evaluation results")
        return
    
    # 输出统计信息
    logger.info("\n***** F1 Evaluation Results *****")
    summary = results['summary']
    logger.info(f"Total samples: {summary['total_samples']}")
    logger.info(f"Successful samples: {summary['successful_samples']}")
    logger.info(f"Success rate: {summary['success_rate']:.2%}")
    logger.info(f"Average F1 Score: {summary['average_f1']:.4f}")
    logger.info(f"F1 Score Std: {summary['std_f1']:.4f}")
    logger.info(f"Max F1 Score: {summary['max_f1']:.4f}")
    logger.info(f"Min F1 Score: {summary['min_f1']:.4f}")
    logger.info(f"Median F1 Score: {summary['median_f1']:.4f}")
    logger.info(f"Average Precision: {summary['average_precision']:.4f}")
    logger.info(f"Average Recall: {summary['average_recall']:.4f}")
    logger.info(f"Average Exact Match: {summary['average_exact_match']:.4f}")
    logger.info(f"Exact Match Count: {summary['exact_match_count']}")
    logger.info(f"Average Unique Token F1: {summary['average_unique_f1']:.4f}")
    
    # 保存结果
    output_data = {
        'evaluation_config': {
            'input_file': args.input_file,
            'tokenizer_path': args.tokenizer_path,
            'use_fast_tokenizer': args.use_fast_tokenizer,
            'original_config': input_data.get('evaluation_config', {})
        },
        'summary': summary,
        'samples': results['samples']
    }
    
    logger.info(f"Saving results to {args.output_file}")
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Results successfully saved to {args.output_file}")
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")
        return
    
    # 显示一些示例
    logger.info("\n***** Sample F1 Results *****")
    successful_samples = [s for s in results['samples'] if s['generation_success']]
    for i, sample in enumerate(successful_samples[:3]):
        logger.info(f"\n--- Sample {sample['sample_id']} ---")
        logger.info(f"Input: {sample['input'][:100]}...")
        logger.info(f"Reference: {sample['reference']}")
        logger.info(f"Prediction: {sample['prediction']}")
        logger.info(f"F1 Score: {sample['f1']:.4f}")
        logger.info(f"Precision: {sample['precision']:.4f}")
        logger.info(f"Recall: {sample['recall']:.4f}")
        logger.info(f"Exact Match: {sample['exact_match']:.0f}")
        logger.info(f"BLEU Score: {sample['bleu_score']:.4f}")

if __name__ == "__main__":
    main()
