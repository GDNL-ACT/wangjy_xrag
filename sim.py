#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import time
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate semantic similarity using BGE-M3 embeddings")
    
    parser.add_argument("--input_file", type=str, required=True, 
                       help="Input JSON file with evaluation results")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output JSON file (default: add '_with_similarity' to input filename)")
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-m3",
                       help="BGE embedding model path or name")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for embedding computation")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length for the model")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu), auto-detect if not specified")
    parser.add_argument("--min_text_length", type=int, default=1,
                       help="Minimum text length to calculate similarity (skip if either text is shorter)")
    parser.add_argument("--use_fp16", action="store_true",
                       help="Use half precision for faster inference")
    
    return parser.parse_args()

class BGEEmbedder:
    """BGE-M3嵌入模型包装器"""
    
    def __init__(self, model_path: str, device: str = None, max_length: int = 512, use_fp16: bool = False):
        self.model_path = model_path
        self.max_length = max_length
        self.use_fp16 = use_fp16
        
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Loading BGE model from {model_path} on {self.device}")
        
        # 加载分词器和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        
        # 移动到指定设备
        self.model = self.model.to(self.device)
        
        # 设置为评估模式
        self.model.eval()
        
        # 如果使用半精度
        if self.use_fp16 and self.device.type == 'cuda':
            self.model = self.model.half()
        
        logger.info(f"Model loaded successfully on {self.device}")
    
    def encode(self, texts: List[str], batch_size: int = 32, normalize_embeddings: bool = True) -> np.ndarray:
        """
        将文本列表编码为嵌入向量
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            normalize_embeddings: 是否归一化嵌入向量
        
        Returns:
            嵌入向量数组，形状为 (len(texts), embedding_dim)
        """
        all_embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
                batch_texts = texts[i:i + batch_size]
                
                # 分词
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # 移动到设备
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # 如果使用半精度，确保输入也是半精度
                if self.use_fp16 and self.device.type == 'cuda':
                    input_ids = input_ids.half()
                    attention_mask = attention_mask.half()
                
                # 获取模型输出
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # 使用CLS token的嵌入或平均池化
                # BGE模型通常使用CLS token
                embeddings = outputs.last_hidden_state[:, 0]  # CLS token
                
                # 归一化
                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                # 转换为numpy并添加到结果列表
                embeddings_np = embeddings.cpu().float().numpy()
                all_embeddings.append(embeddings_np)
        
        # 合并所有批次的结果
        return np.vstack(all_embeddings)

def is_valid_text(text: str, min_length: int = 1) -> bool:
    """检查文本是否有效（非空且长度足够）"""
    return text is not None and text.strip() and len(text.strip()) >= min_length

def get_embeddings_batch(embedder: BGEEmbedder, texts: List[str], batch_size: int = 32) -> List[Optional[np.ndarray]]:
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
    
    try:
        # 计算嵌入
        embeddings = embedder.encode(valid_texts, batch_size=batch_size)
        
        # 将嵌入结果放回对应位置
        for i, embedding in enumerate(embeddings):
            original_index = valid_indices[i]
            embeddings_result[original_index] = embedding
        
        return embeddings_result
        
    except Exception as e:
        logger.error(f"Failed to get embeddings: {str(e)}")
        raise e

def calculate_semantic_similarity(embedding1: Optional[np.ndarray], embedding2: Optional[np.ndarray]) -> float:
    """计算两个嵌入向量之间的余弦相似度"""
    # 如果任一嵌入为None或空，返回0
    if embedding1 is None or embedding2 is None:
        return 0.0
    
    try:
        # 重塑为二维数组
        emb1 = embedding1.reshape(1, -1)
        emb2 = embedding2.reshape(1, -1)
        
        # 计算余弦相似度
        similarity = cosine_similarity(emb1, emb2)[0][0]
        return float(similarity)
    except Exception as e:
        logger.warning(f"Failed to calculate similarity: {str(e)}")
        return 0.0

def process_samples_with_embeddings(samples: List[Dict[Any, Any]], embedder: BGEEmbedder, args: argparse.Namespace) -> List[Dict[Any, Any]]:
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
    
    # 批量获取嵌入
    logger.info("Computing prediction embeddings...")
    prediction_embeddings = get_embeddings_batch(embedder, predictions, args.batch_size)
    
    logger.info("Computing reference embeddings...")
    reference_embeddings = get_embeddings_batch(embedder, references, args.batch_size)
    
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
    
    # 初始化BGE嵌入器
    try:
        embedder = BGEEmbedder(
            model_path=args.embedding_model,
            device=args.device,
            max_length=args.max_length,
            use_fp16=args.use_fp16
        )
    except Exception as e:
        logger.error(f"Failed to initialize BGE embedder: {str(e)}")
        return
    
    # 处理样本，添加语义相似度
    try:
        updated_samples = process_samples_with_embeddings(samples, embedder, args)
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
        'max_length': args.max_length,
        'batch_size': args.batch_size,
        'device': str(embedder.device),
        'use_fp16': args.use_fp16,
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
