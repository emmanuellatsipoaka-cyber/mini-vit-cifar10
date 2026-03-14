"""
transformer_block.py
====================
Bloc Transformer complet : Pre-Norm + Attention + FFN + connexions résiduelles.

Architecture (Pre-Norm, utilisée dans ViT) :
    x = x + Attention(LayerNorm(x))   ← résiduelle 1
    x = x + FFN(LayerNorm(x))         ← résiduelle 2

Connexion résiduelle contre le Vanishing Gradient :
    Sans résiduelle : gradient = ∏ ∂F_l/∂x  → peut tendre vers 0
    Avec résiduelle : gradient = 1 + ∂F/∂x  → jamais nul (terme +1 constant)

Pourquoi GELU et pas ReLU ?
    ReLU(x) = max(0, x) : gradient nul pour x < 0 → "dying neurons"
    GELU(x) = x * Φ(x)  : différentiable partout, gradient non-nul pour tout x
    → convergence plus stable, empiriquement meilleur sur les Transformers (BERT, GPT, ViT)

Pourquoi LayerNorm et pas BatchNorm ?
    BatchNorm normalise sur le batch (dépend de la taille du batch).
    LayerNorm normalise sur la dimension embed_dim indépendamment.
    → plus stable pour les Transformers où la longueur de séquence varie.
"""

import torch.nn as nn
from src.model.attention_head import MultiHeadSelfAttention


class FeedForwardBlock(nn.Module):
    """
    MLP 2 couches avec GELU.
    Linear(embed_dim → hidden_dim) → GELU → Dropout → Linear(hidden_dim → embed_dim) → Dropout

    mlp_ratio=4 : hidden_dim = 4 * embed_dim = 512 (standard ViT)
    """

    def __init__(self, embed_dim=128, mlp_ratio=4, dropout=0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Un bloc Transformer complet avec Pre-LayerNorm.

    Args:
        embed_dim (int)   : Dimension de l'embedding.
        num_heads (int)   : Nombre de têtes d'attention.
        mlp_ratio (float) : Facteur d'expansion du FFN.
        dropout   (float) : Taux de dropout.
    """

    def __init__(self, embed_dim=128, num_heads=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()

        # LayerNorm avant chaque sous-couche (Pre-Norm)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.attention = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.ffn = FeedForwardBlock(
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    def forward(self, x):
        # Résiduelle 1 : autour de l'attention
        # ∂x_out/∂x_in = 1 + ∂Attention/∂x  → gradient ≥ 1
        x = x + self.attention(self.norm1(x))

        # Résiduelle 2 : autour du FFN
        x = x + self.ffn(self.norm2(x))

        return x
