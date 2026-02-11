"""
Contrastive Loss Module — Perte Contrastive wav2vec 2.0

Ce module implémente la perte contrastive utilisée pour l'entraînement
auto-supervisé de wav2vec 2.0. Le modèle apprend à distinguer la cible
quantifiée correcte parmi un ensemble de distracteurs négatifs.

Formules clés :
    Similarité cosinus : sim(a, b) = a·b / (||a|| × ||b||)
    Perte contrastive : L_m = -log( exp(sim(c_t, q_t)/κ) / Σ_ñ exp(sim(c_t, q̃)/κ) )
    Perte de diversité : L_d = (1/GV) × Σ_g Σ_v p̄_{g,v} × log(p̄_{g,v})
    Perte totale : L = L_m + α × L_d

Deux implémentations :
1. From Scratch (NumPy) — boucles explicites pour comprendre chaque terme
2. PyTorch — vectorisé pour l'entraînement efficace
"""

import numpy as np
from typing import Tuple

import torch
import torch.nn.functional as F


# ============================================
# IMPLEMENTATION 1: From Scratch (NumPy)
# ============================================
# Objectif: Comprendre le formalisme mathématique


def cosine_similarity_from_scratch(a: np.ndarray, b: np.ndarray) -> float:
    """Similarité cosinus manuelle : a·b / (||a|| × ||b||).

    Étapes explicites :
    1. Produit scalaire a·b
    2. Normes ||a|| et ||b||
    3. Division (avec epsilon pour éviter division par zéro)

    Args:
        a: Vecteur 1D, shape (d,)
        b: Vecteur 1D, shape (d,)

    Returns:
        Similarité cosinus, scalaire dans [-1, 1].
    """
    # Étape 1 : Produit scalaire
    dot_product = np.dot(a, b)

    # Étape 2 : Normes L2
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # Étape 3 : Division avec epsilon
    eps = 1e-8
    similarity = dot_product / (norm_a * norm_b + eps)

    return float(similarity)


def contrastive_loss_from_scratch(
    context: np.ndarray,
    target: np.ndarray,
    negatives: np.ndarray,
    temperature: float = 0.1
) -> float:
    """Perte contrastive L_m from scratch avec boucle sur les distracteurs.

    Formule :
        L_m = -log( exp(sim(c, q) / κ) / (exp(sim(c, q) / κ) + Σ_ñ exp(sim(c, q̃) / κ)) )

    Le modèle doit apprendre à maximiser la similarité entre le contexte c
    et la cible quantifiée q, tout en minimisant la similarité avec les
    distracteurs négatifs q̃.

    Args:
        context: Vecteur contexte c_t, shape (d,)
        target: Vecteur cible quantifié q_t, shape (d,)
        negatives: Distracteurs négatifs, shape (num_negatives, d)
        temperature: Température κ (contrôle la netteté de la distribution)

    Returns:
        Perte contrastive L_m, scalaire >= 0.
    """
    # Similarité entre contexte et cible positive
    sim_positive = cosine_similarity_from_scratch(context, target)
    # Score logit pour la cible positive
    logit_positive = sim_positive / temperature

    # Scores logits pour les distracteurs négatifs (boucle explicite)
    logit_negatives = []
    for i in range(negatives.shape[0]):
        sim_neg = cosine_similarity_from_scratch(context, negatives[i])
        logit_negatives.append(sim_neg / temperature)

    # Calcul du log-softmax (stabilité numérique)
    # L_m = -log( exp(logit_pos) / (exp(logit_pos) + Σ exp(logit_neg)) )
    all_logits = [logit_positive] + logit_negatives
    max_logit = max(all_logits)

    # log-sum-exp trick pour la stabilité numérique
    log_sum_exp = max_logit + np.log(
        sum(np.exp(l - max_logit) for l in all_logits)
    )

    loss = -(logit_positive - log_sum_exp)
    return float(loss)


# ============================================
# IMPLEMENTATION 2: PyTorch
# ============================================
# Objectif: Code professionnel et efficace


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Similarité cosinus vectorisée avec F.cosine_similarity.

    Args:
        a: Tensor, shape (..., d)
        b: Tensor, shape (..., d)
        dim: Dimension sur laquelle calculer la similarité.

    Returns:
        Similarité cosinus, shape (...)
    """
    return F.cosine_similarity(a, b, dim=dim)


def sample_negatives(
    quantized: torch.Tensor,
    num_negatives: int = 100
) -> torch.Tensor:
    """Échantillonne des distracteurs négatifs depuis la même séquence.

    Pour chaque position temporelle t, on échantillonne num_negatives
    vecteurs quantifiés depuis d'autres positions de la même séquence.

    Args:
        quantized: Vecteurs quantifiés, shape (batch, T, d)
        num_negatives: Nombre de distracteurs par position.

    Returns:
        Distracteurs négatifs, shape (batch, T, num_negatives, d)
    """
    batch, T, d = quantized.shape

    # Pour chaque position, échantillonner des indices différents
    # Créer des indices aléatoires dans [0, T-1] pour chaque position
    neg_indices = torch.randint(0, T, (batch, T, num_negatives))

    # Rassembler les vecteurs négatifs
    # quantized shape: (batch, T, d)
    # On utilise gather sur la dimension temporelle
    neg_indices_expanded = neg_indices.unsqueeze(-1).expand(-1, -1, -1, d)
    quantized_expanded = quantized.unsqueeze(2).expand(-1, -1, num_negatives, -1)

    # Indexation avancée pour récupérer les négatifs
    negatives = torch.zeros(batch, T, num_negatives, d, device=quantized.device)
    for b in range(batch):
        for t in range(T):
            negatives[b, t] = quantized[b, neg_indices[b, t]]

    return negatives


def contrastive_loss(
    context: torch.Tensor,
    quantized: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.1
) -> torch.Tensor:
    """Perte contrastive L_m vectorisée.

    Formule :
        L_m = -log( exp(sim(c_t, q_t)/κ) / (exp(sim(c_t, q_t)/κ) + Σ exp(sim(c_t, q̃)/κ)) )

    Args:
        context: Vecteurs contexte, shape (batch, T, d)
        quantized: Vecteurs cibles quantifiés, shape (batch, T, d)
        negatives: Distracteurs négatifs, shape (batch, T, num_negatives, d)
        temperature: Température κ.

    Returns:
        Perte contrastive moyenne, scalaire.
    """
    # Similarité positive : sim(c_t, q_t) pour chaque (batch, t)
    # context shape: (batch, T, d), quantized shape: (batch, T, d)
    pos_sim = F.cosine_similarity(context, quantized, dim=-1)  # (batch, T)
    pos_logits = pos_sim / temperature  # (batch, T)

    # Similarité négative : sim(c_t, q̃) pour chaque négatif
    # context shape: (batch, T, 1, d), negatives shape: (batch, T, num_neg, d)
    context_expanded = context.unsqueeze(2)  # (batch, T, 1, d)
    neg_sim = F.cosine_similarity(context_expanded, negatives, dim=-1)  # (batch, T, num_neg)
    neg_logits = neg_sim / temperature  # (batch, T, num_neg)

    # Log-softmax sur [positive, negatives]
    # Concaténer : (batch, T, 1 + num_neg)
    all_logits = torch.cat([pos_logits.unsqueeze(-1), neg_logits], dim=-1)

    # La cible est toujours l'indice 0 (le positif)
    targets = torch.zeros(all_logits.shape[0], all_logits.shape[1],
                          dtype=torch.long, device=all_logits.device)

    # Cross-entropy loss (inclut log-softmax)
    loss = F.cross_entropy(
        all_logits.view(-1, all_logits.shape[-1]),
        targets.view(-1),
        reduction='mean'
    )

    return loss


def diversity_loss(
    perplexity: torch.Tensor,
    num_groups: int = 2,
    num_vars: int = 320
) -> torch.Tensor:
    """Perte de diversité L_d pour éviter le mode collapse.

    Encourage une utilisation uniforme de toutes les entrées du codebook.

    Formule :
        L_d = (1 / GV) × Σ_g Σ_v p̄_{g,v} × log(p̄_{g,v})

    Où p̄_{g,v} est la probabilité moyenne de sélection de l'entrée v
    dans le groupe g sur le batch.

    La perte est maximale quand toutes les entrées sont utilisées
    uniformément (entropie maximale), et minimale quand une seule
    entrée est toujours sélectionnée (mode collapse).

    Args:
        perplexity: Probabilités moyennes d'utilisation du codebook,
                    shape (num_groups, num_vars)
        num_groups: Nombre de groupes de quantification (G).
        num_vars: Nombre d'entrées par groupe (V).

    Returns:
        Perte de diversité, scalaire.
    """
    # Entropie négative : Σ p * log(p)
    # Plus l'entropie est élevée, plus l'utilisation est uniforme
    # On veut maximiser l'entropie → minimiser -entropie
    eps = 1e-7
    entropy = -torch.sum(perplexity * torch.log(perplexity + eps))

    # Normaliser par le nombre total d'entrées
    max_entropy = num_groups * np.log(num_vars)

    # L_d = 1 - entropy / max_entropy
    # Quand l'utilisation est uniforme : entropy ≈ max_entropy → L_d ≈ 0
    # Quand mode collapse : entropy ≈ 0 → L_d ≈ 1
    loss = 1.0 - entropy / max_entropy

    return loss
