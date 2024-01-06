import torch
import math
import torch.nn as nn
from torch.nn import functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        B, _, _ = x.shape
        x = x + self.pe.repeat(B, 1, 1)
        return x
        # return self.dropout(x)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.init_parameters()
    
    def init_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)

class Block(nn.Module):
    def __init__(self, cfg):
        super(Block, self).__init__()
        self.ln_q = LayerNorm(cfg.input_dim)
        self.ln_k = LayerNorm(cfg.input_dim)
        self.self_attention = nn.MultiheadAttention(cfg.input_dim, cfg.num_heads)
        self.cross_attention = nn.MultiheadAttention(cfg.input_dim, cfg.num_heads)
        self.mlp = Mlp(cfg.input_dim, cfg.mlp_hidden_features, cfg.input_dim)
        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.self_attention.in_proj_weight)
        nn.init.xavier_uniform_(self.self_attention.out_proj.weight)
        nn.init.xavier_uniform_(self.cross_attention.in_proj_weight)
        nn.init.xavier_uniform_(self.cross_attention.out_proj.weight)
        # nn.init.xavier_uniform_(self.mlp.weight)  # 使用 Xavier初始化
        # nn.init.zeros_(self.mlp.bias)  # 将偏置初始化为零

    def forward(self, query, feature, self_attention_mask=None, cross_attention_mask=None):
        query = query.permute(1, 0, 2)
        feature = feature.permute(1, 0, 2)
        query = self.ln_q(query)
        feature = self.ln_k(feature)
        query, _ = self.self_attention(query, query, query, attn_mask=self_attention_mask)
        query, _ = self.cross_attention(query, feature, feature, attn_mask=cross_attention_mask)
        query = self.mlp(query)
        query = query.permute(1, 0, 2)
        return query