import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerDecoderLayer(nn.Module):
    """简化的Transformer解码器层，用于演示KV Cache"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # TODO: 自己实现带kv_cache的self_attention
        
        
    def forward(self, x, past_key_value=None, use_cache=False):
        # TODO: 实现带KV Cache的前向传播
        pass

class TransformerModel(nn.Module):
    """简化的Transformer模型"""
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead) for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.num_layers = num_layers
        
    def forward(self, x, past_key_values=None, use_cache=False):
        # TODO: 实现模型的前向传播，管理所有层的KV Cache
        pass

def generate(
    model,
    input_ids,
    max_length=50,
    temperature=1.0,
    top_k=None,
    pad_token_id=0,
    eos_token_id=None
):
    """
    使用KV Cache实现高效文本生成
    
    参数:
        model: 预训练的Transformer模型
        input_ids: 输入序列的token ids，形状为 [batch_size, seq_len]
        max_length: 生成文本的最大长度
        temperature: 生成时的温度参数
        top_k: top-k采样的k值，若为None则使用贪婪搜索
        pad_token_id: padding token的ID
        eos_token_id: 结束符token的ID
        
    返回:
        生成的token序列，形状为 [batch_size, max_length]
    """
    # TODO: 实现带KV Cache的生成函数
    # 1. 初始化KV Cache
    # 2. 实现自回归生成循环
    # 3. 在每一步更新KV Cache
    # 4. 处理提前结束的情况（如生成了EOS token）
    pass