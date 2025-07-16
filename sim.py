#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import time
from typing import List, Dict, Any, Optional
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
# import tiktoken
from tqdm import tqdm

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate semantic similarity using OpenAI embeddings")
    
    parser.add_argument("--input_file", type=str, required=True, 
                       help="Input JSON file with evaluation results")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output JSON file (default: add '_with_similarity' to input filename)")
    parser.add_argument("--openai_api_key", type=str, default=None,
                       help="OpenAI API key (can also be set via OPENAI_API_KEY env var)")
    parser.add_argument("--openai_base_url", type=str, default=None,
                       help="OpenAI API base URL (optional, for custom endpoints)")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-ada-002",
                       help="OpenAI embedding model to use")
    parser.add_argument("--batch_size", type=int, default=100,
                       help="Batch size for API calls")
    parser.add_argument("--max_retries", type=int, default=3,
                       help="Maximum number of retries for failed API calls")
    parser.add_argument("--retry_delay", type=float, default=1.0,
                       help="Delay between retries in seconds")
    parser.add_argument("--rate_limit_delay", type=float, default=0.1,
                       help="Delay between API calls to avoid rate limiting")
    parser.add_argument("--min_text_length", type=int, default=1,
                       help="Minimum text length to calculate similarity (skip if either text is shorter)")
    
    return parser.parse_args()

def count_tokens(text: str, model: str = "text-embedding-ada-002") -> int:
    """计算文本的token数量"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        # 如果模型不支持，使用cl100k_base编码作为fallback
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

def is_valid_text(text: str, min_length: int = 1) -> bool:
    """检查文本是否有效（非空且长度足够）"""
    return text is not None and text.strip() and len(text.strip()) >= min_length

def get_embeddings_batch(client: OpenAI, texts: List[str], model: str, max_retries: int = 3, retry_delay: float = 1.0) -> List[Optional[List[float]]]:
    """批量获取文本嵌入，返回None表示无效文本"""
    if not texts:
        return []
    
    # 检查哪些文本是有效的
    valid_texts = []
    valid_indices = []
    
    for i, text in enumerate(texts):
        if is_valid_text(text):
            valid_texts.append(text.strip())
            valid_indices.append(i)
    
    if not valid_texts:
        return [None] * len(texts)
    
    # 获取有效文本的嵌入
    embeddings_result = [None] * len(texts)
    
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                input=valid_texts,
                model=model
            )
            
            # 将嵌入结果放回对应位置
            for i, embedding_data in enumerate(response.data):
                original_index = valid_indices[i]
                embeddings_result[original_index] = embedding_data.embedding
            
            return embeddings_result
            
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))  # 指数退避
            else:
                logger.error(f"Failed to get embeddings after {max_retries} attempts")
                raise e

def calculate_semantic_similarity(embedding1: Optional[List[float]], embedding2: Optional[List[float]]) -> float:
    """计算两个嵌入向量之间的余弦相似度"""
    # 如果任一嵌入为None或空，返回0
    if not embedding1 or not embedding2:
        return 0.0
    
    try:
        # 转换为numpy数组并重塑
        emb1 = np.array(embedding1).reshape(1, -1)
        emb2 = np.array(embedding2).reshape(1, -1)
        
        # 计算余弦相似度
        similarity = cosine_similarity(emb1, emb2)[0][0]
        return float(similarity)
    except Exception as e:
        logger.warning(f"Failed to calculate similarity: {str(e)}")
        return 0.0

def process_samples_with_embeddings(samples: List[Dict[Any, Any]], client: OpenAI, args: argparse.Namespace) -> List[Dict[Any, Any]]:
    """处理样本，添加语义相似度信息"""
    
    # 收集需要计算嵌入的文本
    predictions = []
    references = []
    valid_indices = []
    
    for i, sample in enumerate(samples):
        prediction = sample.get('prediction', '').strip()
        reference = sample.get('reference', '').strip()
        
        # 检查文本有效性
        pred_valid = is_valid_text(prediction, args.min_text_length)
        ref_valid = is_valid_text(reference, args.min_text_length)
        
        # 只有当两个文本都有效时才计算相似度
        if pred_valid and ref_valid:
            predictions.append(prediction)
            references.append(reference)
            valid_indices.append(i)
    
    if not valid_indices:
        logger.warning("No valid text pairs found for embedding calculation")
        # 为所有样本添加默认值
        updated_samples = samples.copy()
        for sample in updated_samples:
            sample.update({
                'semantic_similarity': 0.0,
                'prediction_embedding_available': False,
                'reference_embedding_available': False,
                'embedding_model_used': args.embedding_model,
                'similarity_skip_reason': 'invalid_text_pair'
            })
        return updated_samples
    
    logger.info(f"Processing {len(valid_indices)} valid text pairs for semantic similarity calculation")
    
    # 检查token数量
    # total_tokens = 0
    # for text in predictions + references:
    #     total_tokens += count_tokens(text, args.embedding_model)
    
    # logger.info(f"Total tokens to process: {total_tokens:,}")
    
    # 批量获取嵌入
    all_embeddings = []
    all_texts = predictions + references
    
    progress_bar = tqdm(range(0, len(all_texts), args.batch_size), desc="Getting embeddings")
    
    for start_idx in progress_bar:
        end_idx = min(start_idx + args.batch_size, len(all_texts))
        batch_texts = all_texts[start_idx:end_idx]
        
        try:
            batch_embeddings = get_embeddings_batch(
                client, batch_texts, args.embedding_model, 
                args.max_retries, args.retry_delay
            )
            all_embeddings.extend(batch_embeddings)
            
            # 速率限制
            if args.rate_limit_delay > 0:
                time.sleep(args.rate_limit_delay)
                
        except Exception as e:
            logger.error(f"Failed to process batch {start_idx}-{end_idx}: {str(e)}")
            # 为失败的批次添加None嵌入
            all_embeddings.extend([None for _ in batch_texts])
    
    # 分离预测和参考的嵌入
    prediction_embeddings = all_embeddings[:len(predictions)]
    reference_embeddings = all_embeddings[len(predictions):]
    
    # 计算相似度并更新样本
    updated_samples = samples.copy()
    semantic_similarities = []
    
    for i, (pred_emb, ref_emb) in enumerate(zip(prediction_embeddings, reference_embeddings)):
        sample_idx = valid_indices[i]
        
        # 计算语义相似度
        similarity = calculate_semantic_similarity(pred_emb, ref_emb)
        if similarity > 0:  # 只有成功计算的相似度才加入统计
            semantic_similarities.append(similarity)
        
        # 更新样本信息
        updated_samples[sample_idx].update({
            'semantic_similarity': similarity,
            'prediction_embedding_available': pred_emb is not None,
            'reference_embedding_available': ref_emb is not None,
            'embedding_model_used': args.embedding_model,
            'prediction_text_length': len(predictions[i]),
            'reference_text_length': len(references[i])
        })
    
    # 为跳过的样本添加默认值
    processed_indices = set(valid_indices)
    for i, sample in enumerate(updated_samples):
        if i not in processed_indices:
            prediction = sample.get('prediction', '').strip()
            reference = sample.get('reference', '').strip()
            
            # 确定跳过原因
            skip_reason = []
            if not is_valid_text(prediction, args.min_text_length):
                skip_reason.append('invalid_prediction')
            if not is_valid_text(reference, args.min_text_length):
                skip_reason.append('invalid_reference')
            if not sample.get('generation_success', False):
                skip_reason.append('generation_failed')
            
            sample.update({
                'semantic_similarity': 0.0,
                'prediction_embedding_available': False,
                'reference_embedding_available': False,
                'embedding_model_used': args.embedding_model,
                'similarity_skip_reason': '_'.join(skip_reason) if skip_reason else 'unknown',
                'prediction_text_length': len(prediction),
                'reference_text_length': len(reference)
            })
    
    logger.info(f"Calculated semantic similarities for {len(semantic_similarities)} samples")
    if semantic_similarities:
        logger.info(f"Average semantic similarity: {np.mean(semantic_similarities):.4f}")
        logger.info(f"Semantic similarity std: {np.std(semantic_similarities):.4f}")
        logger.info(f"Max semantic similarity: {np.max(semantic_similarities):.4f}")
        logger.info(f"Min semantic similarity: {np.min(semantic_similarities):.4f}")
    
    return updated_samples

def update_summary_with_semantic_metrics(summary: Dict[str, Any], samples: List[Dict[Any, Any]]) -> Dict[str, Any]:
    """更新摘要信息，添加语义相似度统计"""
    
    # 收集有效的语义相似度分数
    semantic_similarities = []
    embedding_available_count = 0
    skip_reasons = {}
    
    for sample in samples:
        if sample.get('prediction_embedding_available', False) and sample.get('reference_embedding_available', False):
            similarity = sample.get('semantic_similarity', 0.0)
            if similarity > 0:  # 只统计有效的相似度
                semantic_similarities.append(similarity)
                embedding_available_count += 1
        
        # 统计跳过原因
        skip_reason = sample.get('similarity_skip_reason')
        if skip_reason:
            skip_reasons[skip_reason] = skip_reasons.get(skip_reason, 0) + 1
    
    # 更新摘要
    updated_summary = summary.copy()
    
    if semantic_similarities:
        updated_summary.update({
            'semantic_similarity_stats': {
                'samples_with_valid_similarity': embedding_available_count,
                'average_semantic_similarity': float(np.mean(semantic_similarities)),
                'std_semantic_similarity': float(np.std(semantic_similarities)),
                'max_semantic_similarity': float(np.max(semantic_similarities)),
                'min_semantic_similarity': float(np.min(semantic_similarities)),
                'median_semantic_similarity': float(np.median(semantic_similarities)),
                'skip_reasons': skip_reasons
            }
        })
    else:
        updated_summary.update({
            'semantic_similarity_stats': {
                'samples_with_valid_similarity': 0,
                'average_semantic_similarity': 0.0,
                'std_semantic_similarity': 0.0,
                'max_semantic_similarity': 0.0,
                'min_semantic_similarity': 0.0,
                'median_semantic_similarity': 0.0,
                'skip_reasons': skip_reasons
            }
        })
    
    return updated_summary

def main():
    args = parse_args()
    
    # 设置OpenAI客户端
    api_key = args.openai_api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key must be provided via --openai_api_key or OPENAI_API_KEY environment variable")
    
    client_kwargs = {'api_key': api_key}
    if args.openai_base_url:
        client_kwargs['base_url'] = args.openai_base_url
    
    client = OpenAI(**client_kwargs)
    
    # 设置输出文件名
    if args.output_file is None:
        input_name, input_ext = os.path.splitext(args.input_file)
        args.output_file = f"{input_name}_with_similarity{input_ext}"
    
    # 读取输入文件
    logger.info(f"Reading input file: {args.input_file}")
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read input file: {str(e)}")
        return
    
    # 验证数据格式
    if 'samples' not in data:
        logger.error("Input file must contain 'samples' field")
        return
    
    samples = data['samples']
    logger.info(f"Found {len(samples)} samples in input file")
    
    # 处理样本，添加语义相似度
    try:
        updated_samples = process_samples_with_embeddings(samples, client, args)
    except Exception as e:
        logger.error(f"Failed to process samples: {str(e)}")
        return
    
    # 更新摘要信息
    updated_summary = update_summary_with_semantic_metrics(
        data.get('summary', {}), updated_samples
    )
    
    # 准备输出数据
    output_data = data.copy()
    output_data['samples'] = updated_samples
    output_data['summary'] = updated_summary
    
    # 添加处理信息
    output_data['semantic_similarity_processing'] = {
        'embedding_model': args.embedding_model,
        'processing_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'min_text_length': args.min_text_length,
        'total_samples_processed': len([s for s in updated_samples if s.get('prediction_embedding_available', False) and s.get('reference_embedding_available', False)]),
        'skipped_samples': len([s for s in updated_samples if s.get('similarity_skip_reason')])
    }
    
    # 保存结果
    logger.info(f"Saving results to: {args.output_file}")
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info("Results saved successfully")
    except Exception as e:
        logger.error(f"Failed to save output file: {str(e)}")
        return
    
    # 输出统计信息
    logger.info("\n***** Semantic Similarity Results *****")
    semantic_stats = updated_summary.get('semantic_similarity_stats', {})
    logger.info(f"Samples with valid similarity: {semantic_stats.get('samples_with_valid_similarity', 0)}")
    logger.info(f"Average semantic similarity: {semantic_stats.get('average_semantic_similarity', 0.0):.4f}")
    logger.info(f"Semantic similarity std: {semantic_stats.get('std_semantic_similarity', 0.0):.4f}")
    logger.info(f"Max semantic similarity: {semantic_stats.get('max_semantic_similarity', 0.0):.4f}")
    logger.info(f"Min semantic similarity: {semantic_stats.get('min_semantic_similarity', 0.0):.4f}")
    
    # 显示跳过原因统计
    skip_reasons = semantic_stats.get('skip_reasons', {})
    if skip_reasons:
        logger.info("\n***** Skip Reasons *****")
        for reason, count in skip_reasons.items():
            logger.info(f"{reason}: {count}")
    
    # 显示一些示例
    logger.info("\n***** Sample Results with Semantic Similarity *****")
    valid_samples = [s for s in updated_samples if s.get('prediction_embedding_available', False) and s.get('reference_embedding_available', False)]
    for i, sample in enumerate(valid_samples[:3]):
        logger.info(f"\n--- Sample {sample['sample_id']} ---")
        logger.info(f"BLEU Score: {sample.get('bleu_score', 0.0):.4f}")
        logger.info(f"Semantic Similarity: {sample.get('semantic_similarity', 0.0):.4f}")
        logger.info(f"Prediction ({sample.get('prediction_text_length', 0)} chars): {sample.get('prediction', '')[:100]}...")
        logger.info(f"Reference ({sample.get('reference_text_length', 0)} chars): {sample.get('reference', '')[:100]}...")

if __name__ == "__main__":
    main()
