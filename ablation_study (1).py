"""
ablation_study.py
=================
Étude d'ablation : mesurer l'impact de chaque composant du Mini-ViT.

Méthode : retirer un composant à la fois, comparer les performances.
Cela prouve que chaque composant contribue réellement.

Variantes testées (5 époques chacune — suffisant pour comparer les tendances) :
    1. Modèle COMPLET            ← baseline de référence
    2. Sans Positional Encoding  ← le modèle ne sait plus où est chaque patch
    3. Sans CLS token            ← agrégation par mean pooling à la place
    4. Depth=2                   ← seulement 2 blocs Transformer au lieu de 6

Usage :
    python src/experiments/ablation_study.py
"""

import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.model.patch_embedding import PatchEmbedding
from src.model.transformer_block import TransformerBlock
from src.utils.dataset_loader import get_cifar10_loaders
from src.utils.metrics import compute_accuracy, AverageMeter
from src.utils.initialization import initialize_weights


# ─── Modèle flexible pour l'ablation ─────────────────────────────────────────

class MiniViTAblation(nn.Module):
    """
    Version du Mini-ViT avec flags on/off pour désactiver des composants.

    Flags :
        use_pos_embed (bool) : False → pas d'encodage positionnel
        use_cls_token (bool) : False → mean pooling à la place du CLS token
        depth         (int)  : nombre de blocs Transformer
    """

    def __init__(
        self,
        img_size=32, patch_size=4, in_channels=3, num_classes=10,
        embed_dim=128, depth=6, num_heads=4, mlp_ratio=4.0, dropout=0.1,
        use_pos_embed=True,
        use_cls_token=True,
    ):
        super().__init__()
        self.use_pos_embed = use_pos_embed
        self.use_cls_token = use_cls_token

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches  # 64

        # CLS token : seulement si use_cls_token=True
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            seq_len = num_patches + 1
        else:
            self.cls_token = None
            seq_len = num_patches

        # Positional embedding : seulement si use_pos_embed=True
        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

        self.pos_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self.apply(initialize_weights)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        if self.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)
            x   = torch.cat([cls, x], dim=1)

        if self.use_pos_embed:
            x = x + self.pos_embed

        x = self.pos_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Agrégation finale : CLS token ou moyenne de tous les tokens
        features = x[:, 0, :] if self.use_cls_token else x.mean(dim=1)
        return self.head(features)


# ─── Entraînement rapide pour l'ablation ─────────────────────────────────────

def quick_train(model, train_loader, val_loader, device, epochs=5, lr=1e-3):
    """5 époques suffisent pour comparer les tendances entre variantes."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    history   = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        lm, am = AverageMeter(), AverageMeter()
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lm.update(loss.item(), imgs.size(0))
            am.update(compute_accuracy(out, lbls), imgs.size(0))

        # Validation
        model.eval()
        vlm, vam = AverageMeter(), AverageMeter()
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                vlm.update(criterion(out, lbls).item(), imgs.size(0))
                vam.update(compute_accuracy(out, lbls), imgs.size(0))

        history["train_loss"].append(lm.avg)
        history["val_loss"].append(vlm.avg)
        history["train_acc"].append(am.avg)
        history["val_acc"].append(vam.avg)
        print(f"    Ep {epoch}/{epochs} | ValLoss={vlm.avg:.4f} | ValAcc={vam.avg:.4f}")

    return history


# ─── Visualisation ────────────────────────────────────────────────────────────

def plot_ablation(all_histories, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    colors = ["steelblue", "coral", "green", "purple"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for (name, hist), color in zip(all_histories.items(), colors):
        eps = range(1, len(hist["val_loss"]) + 1)
        ax1.plot(eps, hist["val_loss"], label=name, color=color, linewidth=2)
        ax2.plot(eps, hist["val_acc"],  label=name, color=color, linewidth=2)

    ax1.set(xlabel="Époque", ylabel="Val Loss",     title="Ablation — Val Loss")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.set(xlabel="Époque", ylabel="Val Accuracy", title="Ablation — Val Accuracy")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "ablation_results.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Graphique → {path}")


def print_summary(all_histories):
    print("\n" + "=" * 62)
    print(f"  {'Variante':<32} {'Val Acc':>10} {'Val Loss':>10}")
    print("=" * 62)
    for name, hist in all_histories.items():
        print(f"  {name:<32} {hist['val_acc'][-1]:>10.4f} {hist['val_loss'][-1]:>10.4f}")
    print("=" * 62)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Ablation Study — Mini-ViT sur CIFAR-10")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")

    train_loader, val_loader = get_cifar10_loaders(
        data_dir="./data", batch_size=256, num_workers=2
    )

    EPOCHS = 5

    variants = {
        "1. Complet (baseline)":    dict(use_pos_embed=True,  use_cls_token=True,  depth=6),
        "2. Sans Pos. Encoding":    dict(use_pos_embed=False, use_cls_token=True,  depth=6),
        "3. Sans CLS (mean pool)":  dict(use_pos_embed=True,  use_cls_token=False, depth=6),
        "4. Depth=2 (peu profond)": dict(use_pos_embed=True,  use_cls_token=True,  depth=2),
    }

    all_histories = {}

    for name, kwargs in variants.items():
        print(f"\n{'─'*60}")
        print(f"  Variante : {name}")
        print(f"{'─'*60}")
        model = MiniViTAblation(
            embed_dim=128, num_heads=4, num_classes=10, **kwargs
        ).to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f"  Paramètres : {params:,}")
        all_histories[name] = quick_train(
            model, train_loader, val_loader, device, epochs=EPOCHS
        )

    print_summary(all_histories)
    plot_ablation(all_histories, "./results")
    print("\n  Ablation terminée.")


if __name__ == "__main__":
    main()
