"""
patch_embedding.py
==================
Première étape du Vision Transformer : transformer une image en séquence de tokens.

Idée :
    Une image 32x32 avec patch_size=4 donne (32/4)^2 = 64 patches.
    Chaque patch (4x4x3 = 48 pixels) est projeté vers un vecteur de dim embed_dim.
    On obtient 64 "tokens" — exactement comme des mots dans un Transformer NLP.

Pourquoi Conv2d comme projection ?
    Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    est mathématiquement équivalente à une projection linéaire sur chaque patch
    non-chevauchant. Plus efficace qu'un nn.Linear appliqué manuellement.
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Découpe une image en patches et projette chaque patch dans l'espace d'embedding.

    Args:
        img_size    (int) : Taille de l'image carrée. Ex: 32 pour CIFAR-10.
        patch_size  (int) : Taille d'un patch. Ex: 4 → patches 4x4 pixels.
        in_channels (int) : Canaux d'entrée. 3 pour RGB.
        embed_dim   (int) : Dimension de l'embedding (= d_model du Transformer).

    Attribut calculé :
        num_patches (int) : (img_size // patch_size) ** 2  → 64 pour CIFAR-10, patch=4
    """

    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=128):
        super().__init__()

        assert img_size % patch_size == 0, (
            f"img_size ({img_size}) doit être divisible par patch_size ({patch_size})"
        )

        self.patch_size  = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 64

        # Conv2d stride=patch_size : chaque filtre couvre exactement un patch
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        """
        Args:
            x : (B, C, H, W)
        Returns:
            (B, num_patches, embed_dim)  — séquence de tokens prête pour le Transformer
        """
        x = self.projection(x)   # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)         # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)   # (B, num_patches, embed_dim)
        return x
