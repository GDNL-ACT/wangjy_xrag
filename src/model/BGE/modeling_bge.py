import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, TaskType


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class BGE:
    def __init__(self, model_name_or_path='BAAI/bge-m3', device='cpu', enable_lora=False):
        self.model = AutoModel.from_pretrained(
            model_name_or_path, 
            torch_dtype=torch.bfloat16, 
            device_map=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.enable_lora = enable_lora
        
        if enable_lora:
            # 配置LoRA参数
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,  # 特征提取任务
                inference_mode=False,  # 训练模式
                r=16,  # LoRA rank
                lora_alpha=32,  # LoRA alpha
                lora_dropout=0.1,  # LoRA dropout
                target_modules=["query", "key", "value", "dense"],  # 目标模块
            )
            
            # 应用LoRA到模型
            self.model = get_peft_model(self.model, lora_config)
            self.model.train()  # 设置为训练模式
            print("LoRA已启用，模型处于训练模式")
        else:
            self.model.eval()  # 设置为评估模式
            print("模型处于评估模式")
    
    def set_training_mode(self, training=True):
        """手动切换训练/评估模式"""
        if self.enable_lora:
            if training:
                self.model.train()
            else:
                self.model.eval()
        else:
            print("警告：未启用LoRA，无法切换到训练模式")
    
    def get_embed_dim(self):
        if self.enable_lora:
            return self.model.base_model.config.hidden_size
        else:
            return self.model.config.hidden_size
    
    def get_embed_length(self):
        return 1
    
    def get_embedding(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = average_pool(outputs.last_hidden_state, attention_mask)
        # BGE-M3使用L2归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def get_doc_embedding(self, input_ids, attention_mask):
        return self.get_embedding(input_ids, attention_mask)
    
    def get_query_embedding(self, input_ids, attention_mask):
        return self.get_embedding(input_ids, attention_mask)
    
    def encode(self, texts, max_length=512):
        """便捷的编码方法"""
        if isinstance(texts, str):
            texts = [texts]
        
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # 如果启用LoRA且处于训练模式，不使用no_grad
        if self.enable_lora and self.model.training:
            embeddings = self.get_embedding(
                encoded['input_ids'],
                encoded['attention_mask']
            )
        else:
            with torch.no_grad():
                embeddings = self.get_embedding(
                    encoded['input_ids'],
                    encoded['attention_mask']
                )
        
        return embeddings
    
    def get_trainable_parameters(self):
        """获取可训练参数信息（仅在启用LoRA时有效）"""
        if self.enable_lora:
            return self.model.print_trainable_parameters()
        else:
            print("未启用LoRA，所有参数均为冻结状态")
