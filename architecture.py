"""
architecture.py
===============
Mini Vision Transformer (Mini-ViT) complet pour classification CIFAR-10.

Pipeline complet :
    Image (B,3,32,32)
    → PatchEmbedding          → (B, 64, 128)   [64 patches de dim 128]
    → + CLS token             → (B, 65, 128)   [token de classification]
    → + Positional Embedding  → (B, 65, 128)   [info spatiale]
    → Dropout
    → 6x TransformerBlock     → (B, 65, 128)
    → LayerNorm
    → CLS token [:, 0, :]     → (B, 128)
    → Linear head             → (B, 10)

CLS token : vecteur appris ajouté en début de séquence (inspiré de BERT).
            Après les blocs, il agrège l'info globale de l'image.

Positional Embedding : appris (pas sinusoïdal), donne au modèle l'info spatiale.
    Sans lui, l'attention est invariante à la permutation des patches.
"""

import torch
import torch.nn as nn
from src.model.patch_embedding import PatchEmbedding
from src.model.transformer_block import TransformerBlock
from src.utils.initialization import initialize_weights


class MiniViT(nn.Module):
    def __init__(
        self,
        img_size=32, patch_size=4, in_channels=3, num_classes=10,
        embed_dim=128, depth=6, num_heads=4, mlp_ratio=4.0, dropout=0.1,
    ):
        super().__init__()

        # 1. Découpage de l'image en tokens
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches          # 64

        # 2. CLS token — vecteur appris (1,1,embed_dim), répliqué sur le batch au forward
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 3. Positional embedding appris — shape (1, 65, 128)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.pos_dropout = nn.Dropout(dropout)

        # 4. Blocs Transformer empilés
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # 5. Normalisation finale + tête de classification
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # 6. Initialisation des poids
        self.apply(initialize_weights)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)                            # (B, 64, 128)

        cls = self.cls_token.expand(B, -1, -1)             # (B, 1, 128)
        x   = torch.cat([cls, x], dim=1)                   # (B, 65, 128)

        x = x + self.pos_embed                             # ajout position
        x = self.pos_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_out = x[:, 0, :]                               # (B, 128) — CLS token seulement
        return self.head(cls_out)                          # (B, 10)
