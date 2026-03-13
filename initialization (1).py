"""
initialization.py
=================
Stratégie d'initialisation des poids — justification mathématique.

Problème :
    Si les poids sont trop grands → activations explosent → loss = NaN
    Si les poids sont trop petits → activations s'effondrent → gradients nuls

Objectif : maintenir Var(sortie) ≈ Var(entrée) à travers toutes les couches.

────────────────────────────────────────────────────────
XAVIER (Glorot) — pour activations symétriques / linéaires
────────────────────────────────────────────────────────
Pour y = Wx avec n_in entrées :
    Var(y_i) = n_in * Var(w) * Var(x)

On veut Var(y) = Var(x), soit Var(w) = 1/n_in.
En tenant compte aussi de la rétropropagation :
    Var(w) = 2 / (n_in + n_out)

→ W ~ Uniform(-√(6/(n_in+n_out)), +√(6/(n_in+n_out)))

Adapté à GELU (proche d'une activation linéaire pour x > 0).

────────────────────────────────────────────────────────
HE (Kaiming) — pour ReLU
────────────────────────────────────────────────────────
ReLU annule 50% des valeurs → coupe la variance par 2.
Correction : Var(w) = 2 / n_in
→ W ~ Normal(0, √(2/n_in))

────────────────────────────────────────────────────────
Notre choix : Xavier pour les couches Linear/Conv2d
────────────────────────────────────────────────────────
GELU est quasi-linéaire pour x > 0 → Xavier est adapté.
CLS token et pos_embed : trunc_normal(std=0.02) pour stabilité initiale.
"""

import torch.nn as nn


def initialize_weights(module):
    """
    Appliquée via model.apply(initialize_weights).
    Parcourt récursivement tous les sous-modules du modèle.
    """

    if isinstance(module, nn.Linear):
        # Xavier uniform : gain=1 (adapté à GELU)
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Conv2d):
        # Conv2d utilisée comme projection linéaire des patches
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.LayerNorm):
        # gamma=1 (pas de scaling), beta=0 (pas de shift) → normalisation neutre
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
