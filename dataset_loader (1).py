"""
dataset_loader.py
=================
Chargement et prétraitement de CIFAR-10.

CIFAR-10 :
    60 000 images RGB 32x32 | 10 classes | 50 000 train / 10 000 test
    Classes : airplane, automobile, bird, cat, deer,
              dog, frog, horse, ship, truck

Normalisation :
    mean = (0.4914, 0.4822, 0.4465)  — moyenne par canal sur le train set
    std  = (0.2470, 0.2435, 0.2616)  — écart-type par canal sur le train set
    → entrées centrées autour de 0, compatible avec Xavier init (suppose x ~ N(0,1))

Augmentation (train seulement) :
    RandomCrop(32, padding=4)  : recadrage aléatoire → invariance position
    RandomHorizontalFlip()     : symétrie → invariance orientation
    Ces augmentations évitent l'overfitting sans modifier les labels.
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_MEAN    = (0.4914, 0.4822, 0.4465)
CIFAR10_STD     = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def get_cifar10_loaders(data_dir="./data", batch_size=128, num_workers=2):
    """
    Retourne les DataLoaders train et val pour CIFAR-10.

    Args:
        data_dir    (str) : Dossier où télécharger CIFAR-10.
        batch_size  (int) : Images par batch.
        num_workers (int) : Threads parallèles de chargement.

    Returns:
        train_loader, val_loader
    """

    # Transformations entraînement : augmentation + normalisation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # Transformations validation : normalisation seulement (évaluation déterministe)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # download=True : télécharge automatiquement si absent
    train_ds = datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_transform)
    val_ds   = datasets.CIFAR10(data_dir, train=False, download=True, transform=val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader
