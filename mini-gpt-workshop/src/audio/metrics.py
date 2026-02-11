"""
Audio ASR Evaluation Metrics Module — WER & CER

Ce module fournit des fonctions pour calculer le Word Error Rate (WER)
et le Character Error Rate (CER), métriques standard d'évaluation ASR.

Deux implémentations sont fournies :
1. From scratch (Python pur) — pour comprendre l'algorithme de Levenshtein
2. Optimisée via jiwer — pour un usage professionnel

Edge cases gérés :
- Référence vide → raise ValueError (division par zéro)
- Hypothèse vide avec référence non vide → retourne valeur appropriée
"""


# ============================================
# IMPLEMENTATION 1: From Scratch (Python pur)
# ============================================
# Objectif: Comprendre l'algorithme de distance d'édition de Levenshtein


def levenshtein_distance_from_scratch(ref: list, hyp: list) -> int:
    """Distance d'édition de Levenshtein via programmation dynamique.

    Calcule le nombre minimal d'opérations (insertion, suppression,
    substitution) pour transformer hyp en ref.

    Args:
        ref: Liste de tokens de référence (mots ou caractères).
        hyp: Liste de tokens d'hypothèse (mots ou caractères).

    Returns:
        Distance d'édition (entier >= 0).
    """
    n = len(ref)
    m = len(hyp)

    # Matrice (n+1) x (m+1) — dp[i][j] = distance entre ref[:i] et hyp[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Cas de base : transformer une chaîne vide
    for i in range(n + 1):
        dp[i][0] = i  # i suppressions
    for j in range(m + 1):
        dp[0][j] = j  # j insertions

    # Remplissage de la matrice
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                # Pas d'opération nécessaire
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Suppression
                    dp[i][j - 1],      # Insertion
                    dp[i - 1][j - 1],  # Substitution
                )

    return dp[n][m]


def compute_wer_from_scratch(reference: str, hypothesis: str) -> float:
    """Calcule le Word Error Rate from scratch.

    WER = levenshtein(ref_words, hyp_words) / len(ref_words)

    Args:
        reference: Texte de référence.
        hypothesis: Texte transcrit par le modèle.

    Returns:
        WER (float >= 0).

    Raises:
        ValueError: Si la référence est vide.
    """
    ref_words = reference.strip().split()
    if len(ref_words) == 0:
        raise ValueError("Reference string is empty — cannot compute WER (division by zero).")

    hyp_words = hypothesis.strip().split()
    distance = levenshtein_distance_from_scratch(ref_words, hyp_words)
    return distance / len(ref_words)


def compute_cer_from_scratch(reference: str, hypothesis: str) -> float:
    """Calcule le Character Error Rate from scratch.

    CER = levenshtein(ref_chars, hyp_chars) / len(ref_chars)

    Args:
        reference: Texte de référence.
        hypothesis: Texte transcrit par le modèle.

    Returns:
        CER (float >= 0).

    Raises:
        ValueError: Si la référence est vide.
    """
    ref_chars = list(reference)
    if len(ref_chars) == 0:
        raise ValueError("Reference string is empty — cannot compute CER (division by zero).")

    hyp_chars = list(hypothesis)
    distance = levenshtein_distance_from_scratch(ref_chars, hyp_chars)
    return distance / len(ref_chars)


# ============================================
# IMPLEMENTATION 2: Optimisée (jiwer)
# ============================================
# Objectif: Code professionnel et efficace

import jiwer


def compute_wer(reference: str, hypothesis: str) -> float:
    """Calcule le Word Error Rate via jiwer.

    Args:
        reference: Texte de référence.
        hypothesis: Texte transcrit par le modèle.

    Returns:
        WER (float >= 0).

    Raises:
        ValueError: Si la référence est vide.
    """
    if len(reference.strip()) == 0:
        raise ValueError("Reference string is empty — cannot compute WER (division by zero).")

    return jiwer.wer(reference, hypothesis)


def compute_cer(reference: str, hypothesis: str) -> float:
    """Calcule le Character Error Rate via jiwer.

    Args:
        reference: Texte de référence.
        hypothesis: Texte transcrit par le modèle.

    Returns:
        CER (float >= 0).

    Raises:
        ValueError: Si la référence est vide.
    """
    if len(reference) == 0:
        raise ValueError("Reference string is empty — cannot compute CER (division by zero).")

    return jiwer.cer(reference, hypothesis)
