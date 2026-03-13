"""
metrics.py
==========
Utilitaires de mesure des performances du modèle.
"""

import torch


def compute_accuracy(logits, targets):
    """
    Accuracy top-1.

    Args:
        logits  : (B, num_classes) — sorties brutes du modèle
        targets : (B,) — labels vrais (entiers 0..9)

    Returns:
        accuracy (float) dans [0, 1]

    Note : argmax(logits) = argmax(softmax(logits)) car softmax est monotone.
    Pas besoin d'appliquer softmax avant argmax.
    """
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        return (preds == targets).float().mean().item()


class AverageMeter:
    """
    Accumule une métrique sur une époque entière et calcule la moyenne pondérée.

    Usage :
        meter = AverageMeter()
        for batch in dataloader:
            meter.update(loss.item(), n=batch_size)
        print(meter.avg)  # moyenne sur toute l'époque
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum   = 0.0
        self.count = 0

    def update(self, val, n=1):
        """
        Args:
            val (float) : valeur de la métrique pour ce batch
            n   (int)   : nombre d'exemples dans le batch (pour pondérer correctement)
        """
        self.sum   += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else 0.0
