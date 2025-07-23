#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ColBERTSearcher:
    def __init__(self, index_path: str):
        """加载索引并初始化搜索器"""
        with open(index_path, "rb") as f:
            index_data = pickle.load(f)
        
        self.embeddings = index_data["embeddings"]
        self.metadata = index_data["metadata"]
        self.model_name = index_data["model_name"]
        self.max_query_length = index_data["max_query_length"]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        if "[Q]" not in self.tokenizer.vocab:
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["[Q]", "[D]"]})
    
    def encode_query(self, query: str) -> torch.Tensor:
        """编码查询"""
        query_with_prefix = f"[Q] {query}"
        
        encoded = self.tokenizer(
            query_with_prefix,
            padding=True,
            truncation=True,
            max_length=self.max_query_length,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            query_embeddings = outputs.last_hidden_state[0]
            
            # 只保留有效token
            valid_mask = attention_mask[0].bool()
            valid_embeddings = query_embeddings[valid_mask]
            
            # L2归一化
            valid_embeddings = torch.nn.functional.normalize(valid_embeddings, p=2, dim=1)
            
        return valid_embeddings.cpu()
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, dict]]:
        """搜索相似文档"""
        query_embeddings = self.encode_query(query)
        
        scores = []
        for i, doc_embeddings in enumerate(self.embeddings):
            # 计算最大相似度分数
            similarity_matrix = torch.mm(query_embeddings, doc_embeddings.T)
            max_similarities = torch.max(similarity_matrix, dim=1)[0]
            score = torch.sum(max_similarities).item()
            
            scores.append((score, i))
        
        # 排序并返回top-k结果
        scores.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for score, idx in scores[:top_k]:
            results.append((score, self.metadata[idx]))
        
        return results

def main():
    parser = argparse.ArgumentParser(description="ColBERT搜索示例")
    parser.add_argument("index_path", help="索引文件路径")
    parser.add_argument("query", help="搜索查询")
    parser.add_argument("--top_k", "-k", type=int, default=5, help="返回结果数量")
    
    args = parser.parse_args()
    
    searcher = ColBERTSearcher(args.index_path)
    results = searcher.search(args.query, args.top_k)
    
    print(f"\n搜索查询: {args.query}")
    print("=" * 50)
    
    for i, (score, metadata) in enumerate(results, 1):
        print(f"\n{i}. 分数: {score:.4f}")
        print(f"   标题: {metadata['title']}")
        print(f"   ID: {metadata['id']}")
        print(f"   章节: {metadata['section']}")
        print(f"   文本: {metadata['original_text'][:200]}...")

if __name__ == "__main__":
    main()
