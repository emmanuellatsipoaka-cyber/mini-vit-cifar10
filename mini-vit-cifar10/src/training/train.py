"""
train.py
========
Boucle d'entraînement complète du Mini-ViT sur CIFAR-10.

Usage :
    python src/training/train.py

Ce script :
    1. Charge CIFAR-10
    2. Instancie MiniViT
    3. Configure AdamW + Cosine LR avec warmup
    4. Lance train + validation sur `epochs` époques
    5. Sauvegarde le meilleur modèle dans results/best_model.pth
    6. Génère results/training_curves.png
"""

import os, sys, math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.model.architecture import MiniViT
from src.utils.dataset_loader import get_cifar10_loaders
from src.utils.metrics import compute_accuracy, AverageMeter

# ─── Hyperparamètres ────────────────────────────────────────────────────────
CONFIG = {
    "img_size": 32, "patch_size": 4, "embed_dim": 128,
    "depth": 6, "num_heads": 4, "mlp_ratio": 4.0,
    "dropout": 0.1, "num_classes": 10,
    "epochs": 30, "batch_size": 128,
    "lr": 1e-3, "weight_decay": 0.05, "warmup_epochs": 5,
    "data_dir": "./data", "results_dir": "./results",
    "save_path": "./results/best_model.pth", "num_workers": 2,
}

def get_device():
    if torch.cuda.is_available():
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("  CPU mode")
    return torch.device("cpu")

def build_optimizer(model, cfg):
    """
    AdamW : weight decay découplé des moments (meilleur que Adam pour Transformers).
    On exclut les biais et LayerNorm du weight decay (pas de sens de les pénaliser).
    """
    decay     = [p for n, p in model.named_parameters() if p.ndim >= 2 and p.requires_grad]
    no_decay  = [p for n, p in model.named_parameters() if p.ndim <  2 and p.requires_grad]
    return torch.optim.AdamW([
        {"params": decay,    "weight_decay": cfg["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=cfg["lr"])

def build_scheduler(optimizer, cfg):
    """Warmup linéaire puis Cosine Annealing."""
    def lr_lambda(epoch):
        if epoch < cfg["warmup_epochs"]:
            return float(epoch + 1) / float(max(1, cfg["warmup_epochs"]))
        progress = (epoch - cfg["warmup_epochs"]) / (cfg["epochs"] - cfg["warmup_epochs"])
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_m, acc_m = AverageMeter(), AverageMeter()
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(images)
        loss   = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # évite les explosions de gradient
        optimizer.step()
        loss_m.update(loss.item(), images.size(0))
        acc_m.update(compute_accuracy(logits, labels), images.size(0))
    return loss_m.avg, acc_m.avg

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_m, acc_m = AverageMeter(), AverageMeter()
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(images)
        loss_m.update(criterion(logits, labels).item(), images.size(0))
        acc_m.update(compute_accuracy(logits, labels), images.size(0))
    return loss_m.avg, acc_m.avg

def plot_curves(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train", color="steelblue")
    ax1.plot(epochs, history["val_loss"],   label="Val",   color="coral")
    ax1.set(xlabel="Époque", ylabel="Loss", title="Cross-Entropy Loss")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train", color="steelblue")
    ax2.plot(epochs, history["val_acc"],   label="Val",   color="coral")
    ax2.set(xlabel="Époque", ylabel="Accuracy", title="Accuracy")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle("Mini-ViT — CIFAR-10", fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Courbes → {path}")

def main():
    print("=" * 60)
    print("  Mini-ViT — Entraînement CIFAR-10")
    print("=" * 60)
    device = get_device()
    os.makedirs(CONFIG["results_dir"], exist_ok=True)

    print("\n[1/4] Chargement CIFAR-10...")
    train_loader, val_loader = get_cifar10_loaders(
        CONFIG["data_dir"], CONFIG["batch_size"], CONFIG["num_workers"]
    )
    print(f"  Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    print("\n[2/4] Initialisation du modèle...")
    model = MiniViT(
        img_size=CONFIG["img_size"], patch_size=CONFIG["patch_size"],
        embed_dim=CONFIG["embed_dim"], depth=CONFIG["depth"],
        num_heads=CONFIG["num_heads"], mlp_ratio=CONFIG["mlp_ratio"],
        dropout=CONFIG["dropout"], num_classes=CONFIG["num_classes"],
    ).to(device)
    print(f"  Paramètres : {sum(p.numel() for p in model.parameters()):,}")

    print("\n[3/4] Configuration optimisation...")
    # CrossEntropyLoss avec label smoothing : réduit la confiance excessive → meilleure généralisation
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = build_optimizer(model, CONFIG)
    scheduler = build_scheduler(optimizer, CONFIG)

    print("\n[4/4] Boucle d'entraînement...")
    history    = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc   = 0.0

    for epoch in range(1, CONFIG["epochs"] + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        for k, v in zip(["train_loss","val_loss","train_acc","val_acc"],
                         [tr_loss, vl_loss, tr_acc, vl_acc]):
            history[k].append(v)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Ep {epoch:3d}/{CONFIG['epochs']} | "
              f"TrainLoss={tr_loss:.4f} TrainAcc={tr_acc:.4f} | "
              f"ValLoss={vl_loss:.4f} ValAcc={vl_acc:.4f} | LR={lr_now:.6f}")

        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_acc": vl_acc, "config": CONFIG}, CONFIG["save_path"])
            print(f"    ✓ Meilleur modèle sauvegardé (ValAcc={vl_acc:.4f})")

    print(f"\n  Meilleure Val Accuracy : {best_acc:.4f}")
    plot_curves(history, CONFIG["results_dir"])
    print("  Entraînement terminé.")

if __name__ == "__main__":
    main()
