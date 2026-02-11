"""
Adapter Module — Bottleneck Adapters pour le Transfer Learning Multilingue

Ce module implémente les adaptateurs bottleneck utilisés dans MMS (Massively
Multilingual Speech) pour adapter un modèle wav2vec 2.0 pré-entraîné à de
nouvelles langues avec un minimum de paramètres entraînables.

Architecture Bottleneck Adapter :
    Input (d_model)
      ↓
    Down-projection : Linear(d_model → bottleneck_dim)
      ↓
    Non-linéarité : ReLU
      ↓
    Up-projection : Linear(bottleneck_dim → d_model)
      ↓
    Connexion résiduelle : output = input + adapter(input)

Avantage : Seuls les paramètres de l'adaptateur sont entraînés (~2-5% du total),
le reste du modèle est gelé. Cela permet d'adapter le modèle à une nouvelle
langue avec très peu de données.

Deux implémentations :
1. From Scratch (NumPy) — multiplication matricielle explicite
2. PyTorch (nn.Module) — code professionnel avec insertion dans un Transformer gelé
"""

import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn


# ============================================
# IMPLEMENTATION 1: From Scratch (NumPy)
# ============================================
# Objectif: Comprendre le formalisme mathématique


def adapter_from_scratch(
    x: np.ndarray,
    W_down: np.ndarray,
    W_up: np.ndarray
) -> np.ndarray:
    """Adaptateur bottleneck from scratch.

    Étapes explicites :
    1. Down-projection : h = x @ W_down   (d_model → bottleneck_dim)
    2. Non-linéarité :   h = relu(h)
    3. Up-projection :   h = h @ W_up     (bottleneck_dim → d_model)
    4. Connexion résiduelle : output = x + h

    Args:
        x: Tensor d'entrée, shape (..., d_model)
        W_down: Matrice de down-projection, shape (d_model, bottleneck_dim)
        W_up: Matrice de up-projection, shape (bottleneck_dim, d_model)

    Returns:
        Tensor de sortie, shape (..., d_model) — même shape que l'entrée.
    """
    # Étape 1 : Down-projection (d_model → bottleneck_dim)
    h = np.matmul(x, W_down)

    # Étape 2 : Non-linéarité ReLU
    h = np.maximum(0, h)

    # Étape 3 : Up-projection (bottleneck_dim → d_model)
    h = np.matmul(h, W_up)

    # Étape 4 : Connexion résiduelle
    output = x + h

    return output


# ============================================
# IMPLEMENTATION 2: PyTorch (nn.Module)
# ============================================
# Objectif: Code professionnel et efficace


class BottleneckAdapter(nn.Module):
    """Module adaptateur bottleneck pour le transfer learning.

    Architecture : Linear(d_model → bottleneck_dim) → ReLU → Linear(bottleneck_dim → d_model) + Résiduel

    Le bottleneck réduit la dimensionnalité, forçant le réseau à apprendre
    une représentation compacte de l'adaptation nécessaire. La connexion
    résiduelle garantit que la sortie a la même shape que l'entrée.

    Méthodes PyTorch expliquées :
    - nn.Linear() : Projection linéaire (y = xW^T + b)
    - nn.ReLU() : Activation (max(0, x))
    - Connexion résiduelle : output = input + adapter(input)

    Args:
        d_model: Dimension du Transformer (ex: 768 pour wav2vec 2.0 base).
        bottleneck_dim: Dimension réduite de l'adaptateur (ex: 64).
    """

    def __init__(self, d_model: int = 768, bottleneck_dim: int = 64):
        super().__init__()
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim

        # Down-projection : d_model → bottleneck_dim
        self.down_proj = nn.Linear(d_model, bottleneck_dim)

        # Non-linéarité
        self.activation = nn.ReLU()

        # Up-projection : bottleneck_dim → d_model
        self.up_proj = nn.Linear(bottleneck_dim, d_model)

        # Initialisation near-zero pour que l'adaptateur commence comme identité
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass avec connexion résiduelle.

        Args:
            x: Tensor d'entrée, shape (..., d_model)

        Returns:
            Tensor de sortie, shape (..., d_model) — même shape que l'entrée.
        """
        # Adapter path : down → relu → up
        h = self.down_proj(x)
        h = self.activation(h)
        h = self.up_proj(h)

        # Connexion résiduelle
        return x + h


def insert_adapters(model: nn.Module, bottleneck_dim: int = 64) -> nn.Module:
    """Insère des adaptateurs dans un modèle Transformer gelé.

    Stratégie :
    1. Geler tous les paramètres du modèle de base (requires_grad = False)
    2. Identifier les couches d'attention et FFN dans chaque bloc Transformer
    3. Envelopper ces couches avec un adaptateur bottleneck
    4. Seuls les paramètres des adaptateurs sont entraînables

    Args:
        model: Modèle Transformer (nn.Module) à adapter.
        bottleneck_dim: Dimension du bottleneck des adaptateurs.

    Returns:
        Le modèle modifié avec adaptateurs insérés. Les paramètres originaux
        sont gelés et seuls les adaptateurs sont entraînables.
    """
    # Étape 1 : Geler tous les paramètres du modèle de base
    for param in model.parameters():
        param.requires_grad = False

    # Étape 2 : Identifier et envelopper les sous-modules cibles
    # On cherche les modules nommés "attention" et "feed_forward"
    _insert_adapters_recursive(model, bottleneck_dim)

    return model


def _insert_adapters_recursive(module: nn.Module, bottleneck_dim: int) -> None:
    """Parcourt récursivement le modèle pour insérer des adaptateurs.

    Insère un AdaptedLayer après chaque sous-module dont le nom contient
    'attention' ou 'feed_forward'.
    """
    for name, child in list(module.named_children()):
        # Vérifier si ce sous-module est une cible pour l'adaptation
        if _is_adapter_target(name, child):
            d_model = _infer_d_model(child)
            if d_model is not None:
                adapted = AdaptedLayer(child, d_model=d_model, bottleneck_dim=bottleneck_dim)
                setattr(module, name, adapted)
        else:
            # Continuer la recherche récursive
            _insert_adapters_recursive(child, bottleneck_dim)


def _is_adapter_target(name: str, module: nn.Module) -> bool:
    """Détermine si un module est une cible pour l'insertion d'adaptateur.

    Cibles : modules dont le nom contient 'attention' ou 'feed_forward'.
    """
    target_names = ("attention", "feed_forward")
    return any(t in name.lower() for t in target_names)


def _infer_d_model(module: nn.Module) -> int:
    """Infère la dimension d_model d'un module en inspectant ses paramètres."""
    # Chercher un attribut d_model explicite
    if hasattr(module, "d_model"):
        return module.d_model

    # Sinon, inspecter la dernière couche linéaire
    last_linear = None
    for child in module.modules():
        if isinstance(child, nn.Linear):
            last_linear = child
    if last_linear is not None:
        return last_linear.out_features

    return None


class AdaptedLayer(nn.Module):
    """Enveloppe un module existant avec un adaptateur bottleneck.

    Le forward pass exécute d'abord le module original (gelé),
    puis applique l'adaptateur sur la sortie.

    Args:
        original_module: Le module Transformer original (gelé).
        d_model: Dimension du modèle.
        bottleneck_dim: Dimension du bottleneck de l'adaptateur.
    """

    def __init__(self, original_module: nn.Module, d_model: int, bottleneck_dim: int = 64):
        super().__init__()
        self.original = original_module
        self.adapter = BottleneckAdapter(d_model=d_model, bottleneck_dim=bottleneck_dim)

    def forward(self, *args, **kwargs):
        """Forward pass : module original → adaptateur."""
        out = self.original(*args, **kwargs)
        return self.adapter(out)
