"""
attention_head.py
=================
Multi-Head Self-Attention from scratch.

Formule : Attention(Q,K,V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    - Q, K, V : projections linéaires de l'entrée
    - sqrt(d_k) : normalisation pour éviter la saturation du softmax
    - num_heads : plusieurs têtes en parallèle dans des sous-espaces distincts
"""

import torch
import torch.nn as nn
import math


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = math.sqrt(self.head_dim)   # 1/sqrt(d_k)

        # Projections Q, K, V en une seule matrice (plus efficace)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj  = nn.Linear(embed_dim, embed_dim)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape   # batch, séquence, embed_dim

        # Projeter et reshape pour les têtes multiples
        qkv = self.qkv_proj(x)                              # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                   # (3, B, heads, N, head_dim)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # Scores d'attention
        attn = (Q @ K.transpose(-2, -1)) / self.scale       # (B, heads, N, N)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Agrégation pondérée des valeurs
        out = attn @ V                                        # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, self.embed_dim)  # (B, N, embed_dim)
        return self.out_proj(out)
