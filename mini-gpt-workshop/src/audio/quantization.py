"""
Quantization Module — Quantification Vectorielle (wav2vec 2.0)

Ce module implémente le module de quantification de wav2vec 2.0, qui
transforme les représentations continues en pseudo-tokens acoustiques
discrets via Product Quantization et Gumbel-Softmax.

Concept clé :
    Les représentations latentes continues sont mappées vers des entrées
    discrètes d'un codebook, créant des "tokens audio" analogues aux
    tokens textuels de BPE.

Architecture wav2vec 2.0 :
    - Product Quantization : G groupes × V entrées par groupe
    - Gumbel-Softmax : Sélection discrète différentiable
    - Temperature annealing : Température décroissante pendant l'entraînement

Deux implémentations :
1. From Scratch (NumPy) — argmin sur distances euclidiennes (non-différentiable)
2. PyTorch (nn.Module) — Gumbel-Softmax différentiable avec Product Quantization
"""

import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================
# IMPLEMENTATION 1: From Scratch (NumPy)
# ============================================
# Objectif: Comprendre la quantification vectorielle


def quantize_from_scratch(
    x: np.ndarray,
    codebook: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Quantification vectorielle from scratch avec argmin.

    Pour chaque vecteur d'entrée, trouve le vecteur codebook le plus proche
    (distance euclidienne minimale). Non-différentiable.

    Étapes explicites :
    1. Pour chaque vecteur x_i, calculer la distance à chaque entrée du codebook
    2. Sélectionner l'entrée la plus proche (argmin)
    3. Remplacer x_i par l'entrée du codebook sélectionnée

    Args:
        x: Vecteurs d'entrée, shape (T, d)
            T = nombre de pas temporels, d = dimension
        codebook: Livre de codes, shape (V, d)
            V = nombre d'entrées, d = dimension

    Returns:
        quantized: Vecteurs quantifiés, shape (T, d)
            Chaque vecteur est une entrée du codebook.
        indices: Indices des entrées sélectionnées, shape (T,)
    """
    T, d = x.shape
    V, d_cb = codebook.shape
    assert d == d_cb, f"Dimension mismatch: x has {d}, codebook has {d_cb}"

    indices = np.zeros(T, dtype=np.int64)
    quantized = np.zeros_like(x)

    for t in range(T):
        # Calculer la distance euclidienne à chaque entrée du codebook
        # ||x_t - c_v||² = ||x_t||² - 2 * x_t · c_v + ||c_v||²
        distances = np.sum((x[t:t+1] - codebook) ** 2, axis=1)  # shape (V,)

        # Sélectionner l'entrée la plus proche
        idx = np.argmin(distances)
        indices[t] = idx
        quantized[t] = codebook[idx]

    return quantized, indices


# ============================================
# IMPLEMENTATION 2: PyTorch (nn.Module)
# ============================================
# Objectif: Code professionnel et efficace


class GumbelVectorQuantizer(nn.Module):
    """Module de quantification vectorielle avec Gumbel-Softmax.

    Implémente la Product Quantization avec G groupes et V entrées par groupe.
    Utilise Gumbel-Softmax pour rendre la sélection discrète différentiable.

    Product Quantization :
        L'espace d'entrée (d dimensions) est divisé en G groupes de d/G dimensions.
        Chaque groupe a son propre codebook de V entrées.
        Le vecteur quantifié final est la concaténation des entrées sélectionnées.

    Gumbel-Softmax :
        Permet de sélectionner une entrée discrète du codebook tout en
        maintenant la différentiabilité pour la backpropagation.
        La température contrôle la "dureté" de la sélection :
        - Haute température → sélection douce (mélange d'entrées)
        - Basse température → sélection dure (une seule entrée)

    Méthodes PyTorch expliquées :
    - F.gumbel_softmax() : Sélection discrète différentiable
    - nn.Parameter() : Codebook comme paramètre apprenable

    Args:
        input_dim: Dimension d'entrée (d). Default: 512.
        num_groups: Nombre de groupes (G). Default: 2.
        num_vars: Nombre d'entrées par groupe (V). Default: 320.
        temp: Tuple (temp_start, temp_end, temp_decay) pour le temperature annealing.
    """

    def __init__(
        self,
        input_dim: int = 512,
        num_groups: int = 2,
        num_vars: int = 320,
        temp: Tuple[float, float, float] = (2.0, 0.5, 0.999995)
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_groups = num_groups
        self.num_vars = num_vars

        # Temperature annealing
        self.temp_start, self.temp_end, self.temp_decay = temp
        self.current_temp = self.temp_start

        # Dimension par groupe
        assert input_dim % num_groups == 0, (
            f"input_dim ({input_dim}) must be divisible by num_groups ({num_groups})"
        )
        self.dim_per_group = input_dim // num_groups

        # Projection linéaire : d → G × V (logits pour chaque groupe)
        self.weight_proj = nn.Linear(input_dim, num_groups * num_vars)

        # Codebook : G groupes × V entrées × (d/G) dimensions par entrée
        # nn.Parameter : Enregistre le tensor comme paramètre apprenable
        self.codebook = nn.Parameter(
            torch.FloatTensor(1, num_groups * num_vars, self.dim_per_group)
        )
        nn.init.uniform_(self.codebook)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantifie les vecteurs d'entrée via Gumbel-Softmax.

        Args:
            x: Vecteurs d'entrée, shape (batch, T, input_dim)

        Returns:
            quantized: Vecteurs quantifiés, shape (batch, T, input_dim)
            perplexity: Perplexité du codebook (mesure d'utilisation),
                        shape (num_groups, num_vars)
        """
        batch, T, d = x.shape
        assert d == self.input_dim, (
            f"Expected input_dim={self.input_dim}, got {d}"
        )

        # Étape 1 : Projeter vers les logits du codebook
        # x shape: (batch, T, d) → logits shape: (batch, T, G × V)
        logits = self.weight_proj(x)

        # Reshape pour séparer les groupes : (batch × T, G, V)
        logits = logits.view(batch * T, self.num_groups, self.num_vars)

        # Étape 2 : Gumbel-Softmax pour sélection différentiable
        # F.gumbel_softmax : Ajoute du bruit Gumbel puis applique softmax
        # hard=True : Retourne un one-hot en forward, mais gradient doux en backward
        if self.training:
            soft_one_hot = F.gumbel_softmax(
                logits, tau=self.current_temp, hard=True, dim=-1
            )
        else:
            # En inférence, sélection argmax directe
            indices = logits.argmax(dim=-1)
            soft_one_hot = F.one_hot(indices, self.num_vars).float()

        # soft_one_hot shape: (batch × T, G, V)

        # Étape 3 : Calculer la perplexité (mesure d'utilisation du codebook)
        # Probabilités moyennes d'utilisation sur le batch
        avg_probs = torch.mean(
            soft_one_hot.view(batch * T, self.num_groups, self.num_vars),
            dim=0
        )  # shape: (G, V)

        # Étape 4 : Sélectionner les entrées du codebook
        # codebook shape: (1, G × V, dim_per_group)
        # Reshape codebook : (G, V, dim_per_group)
        codebook = self.codebook.view(self.num_groups, self.num_vars, self.dim_per_group)

        # Multiplier one-hot par codebook pour sélectionner les entrées
        # soft_one_hot shape: (batch × T, G, V)
        # codebook shape: (G, V, dim_per_group)
        # Résultat : (batch × T, G, dim_per_group)
        selected = torch.einsum('bgv,gvd->bgd', soft_one_hot, codebook)

        # Concaténer les groupes : (batch × T, G × dim_per_group) = (batch × T, d)
        quantized = selected.reshape(batch * T, self.input_dim)

        # Reshape final : (batch, T, d)
        quantized = quantized.reshape(batch, T, self.input_dim)

        # Étape 5 : Mettre à jour la température (annealing)
        if self.training:
            self.current_temp = max(
                self.temp_end,
                self.current_temp * self.temp_decay
            )

        return quantized, avg_probs
