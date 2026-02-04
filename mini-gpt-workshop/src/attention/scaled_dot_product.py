"""
Scaled Dot-Product Attention Implementation

Ce module implémente le mécanisme d'attention fondamental des transformers.
L'attention permet à chaque token de "regarder" les autres tokens et de pondérer
leur importance pour la représentation finale.

Formule mathématique:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

Où:
    - Q (Query): "Ce que je cherche"
    - K (Key): "Ce que j'offre comme information"
    - V (Value): "L'information elle-même"
    - d_k: Dimension des clés (pour la normalisation)
"""

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ============================================
# IMPLEMENTATION 1: From Scratch (NumPy)
# ============================================
# Objectif: Comprendre le formalisme mathématique

def softmax_from_scratch(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Implémentation from-scratch du softmax pour la stabilité numérique.
    
    Formule:
        softmax(x_i) = exp(x_i) / sum(exp(x_j))
    
    Astuce de stabilité numérique:
        softmax(x) = softmax(x - max(x))
    
    Args:
        x: Tableau NumPy de scores
        axis: Axe sur lequel appliquer le softmax
    
    Returns:
        Probabilités normalisées (somme = 1 sur l'axe spécifié)
    
    Dimensions:
        Input: (..., n)
        Output: (..., n) avec sum(output, axis=axis) = 1
    """
    # Soustraire le max pour la stabilité numérique (évite overflow)
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    
    # Calculer les exponentielles
    exp_x = np.exp(x_shifted)
    
    # Normaliser pour obtenir des probabilités
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention_from_scratch(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implémentation from-scratch de l'attention avec NumPy.
    
    Cette fonction décompose l'attention en 4 étapes explicites pour
    comprendre chaque opération mathématique.
    
    Formule complète:
        Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
    
    Décomposition étape par étape:
        1. Scores de similarité: S = QK^T
        2. Normalisation (scaling): S' = S / sqrt(d_k)
        3. Application du masque (optionnel): S'[mask==0] = -inf
        4. Poids d'attention: A = softmax(S')
        5. Sortie pondérée: O = AV
    
    Args:
        Q: Matrice Query de shape (batch_size, seq_len, d_k)
           Représente "ce que chaque token cherche"
        K: Matrice Key de shape (batch_size, seq_len, d_k)
           Représente "ce que chaque token offre comme information"
        V: Matrice Value de shape (batch_size, seq_len, d_v)
           Représente "l'information elle-même"
        mask: Masque optionnel de shape (seq_len, seq_len)
              1 = autoriser l'attention, 0 = bloquer l'attention
    
    Returns:
        output: Sortie pondérée de shape (batch_size, seq_len, d_v)
        attention_weights: Poids d'attention de shape (batch_size, seq_len, seq_len)
    
    Exemple:
        >>> Q = np.random.randn(2, 4, 64)  # batch=2, seq_len=4, d_k=64
        >>> K = np.random.randn(2, 4, 64)
        >>> V = np.random.randn(2, 4, 64)
        >>> output, weights = scaled_dot_product_attention_from_scratch(Q, K, V)
        >>> print(output.shape)  # (2, 4, 64)
        >>> print(weights.shape)  # (2, 4, 4)
        >>> print(np.allclose(weights.sum(axis=-1), 1.0))  # True (normalisé)
    """
    # Récupérer la dimension des clés pour la normalisation
    # d_k est la dernière dimension de Q et K
    d_k = Q.shape[-1]
    
    # ========================================
    # ÉTAPE 1: Calcul des scores de similarité
    # ========================================
    # Formule: S = QK^T
    # Intuition: Mesure la similarité entre chaque paire de tokens
    # 
    # Q shape: (batch_size, seq_len, d_k)
    # K^T shape: (batch_size, d_k, seq_len)
    # S shape: (batch_size, seq_len, seq_len)
    #
    # S[i,j] = similarité entre le token i (query) et le token j (key)
    
    scores = np.matmul(Q, K.transpose(0, 2, 1))  # QK^T
    # Note: transpose(0, 2, 1) échange les deux dernières dimensions
    # (batch, seq_len, d_k) @ (batch, d_k, seq_len) -> (batch, seq_len, seq_len)
    
    print(f"[Attention] Scores shape après QK^T: {scores.shape}")
    
    # ========================================
    # ÉTAPE 2: Normalisation (Scaling)
    # ========================================
    # Formule: S' = S / sqrt(d_k)
    # Intuition: Évite que les scores deviennent trop grands
    # 
    # Pourquoi diviser par sqrt(d_k)?
    # - Les produits scalaires croissent avec la dimension
    # - Sans normalisation, softmax sature (gradients → 0)
    # - sqrt(d_k) est la déviation standard théorique du produit scalaire
    
    scores = scores / math.sqrt(d_k)
    
    print(f"[Attention] Scores après scaling par sqrt({d_k}) = {math.sqrt(d_k):.2f}")
    
    # ========================================
    # ÉTAPE 3: Application du masque (optionnel)
    # ========================================
    # Formule: S'[mask==0] = -inf
    # Intuition: Empêche certains tokens d'en voir d'autres
    # 
    # Cas d'usage:
    # - Masque causal (GPT): empêche de voir le futur
    # - Masque de padding: ignore les tokens de remplissage
    #
    # Pourquoi -inf?
    # - softmax(-inf) = 0, donc attention nulle
    
    if mask is not None:
        # Remplacer les positions masquées par -inf
        # mask == 0 → position bloquée → score = -inf → attention = 0
        scores = np.where(mask == 0, -1e9, scores)  # -1e9 ≈ -inf
        print(f"[Attention] Masque appliqué, shape: {mask.shape}")
    
    # ========================================
    # ÉTAPE 4: Calcul des poids d'attention
    # ========================================
    # Formule: A = softmax(S')
    # Intuition: Convertit les scores en probabilités
    # 
    # Propriétés du softmax:
    # - Toutes les valeurs sont entre 0 et 1
    # - La somme sur chaque ligne = 1
    # - Les scores élevés → probabilités élevées
    #
    # A[i,j] = "combien le token i doit prêter attention au token j"
    
    attention_weights = softmax_from_scratch(scores, axis=-1)
    # axis=-1 signifie: normaliser sur la dernière dimension (les keys)
    # Chaque query (ligne) a une distribution de probabilité sur toutes les keys
    
    print(f"[Attention] Poids d'attention shape: {attention_weights.shape}")
    print(f"[Attention] Vérification normalisation: sum = {attention_weights[0, 0, :].sum():.4f} (doit être ≈ 1.0)")
    
    # ========================================
    # ÉTAPE 5: Pondération des valeurs
    # ========================================
    # Formule: O = AV
    # Intuition: Combine les valeurs selon les poids d'attention
    # 
    # A shape: (batch_size, seq_len, seq_len)
    # V shape: (batch_size, seq_len, d_v)
    # O shape: (batch_size, seq_len, d_v)
    #
    # O[i] = somme pondérée des valeurs, où les poids sont A[i,:]
    # O[i] = sum_j A[i,j] * V[j]
    
    output = np.matmul(attention_weights, V)
    
    print(f"[Attention] Output shape: {output.shape}")
    
    return output, attention_weights


# ============================================
# EXEMPLE D'UTILISATION
# ============================================

def example_attention_from_scratch():
    """
    Exemple pédagogique avec des petites matrices pour comprendre l'attention.
    
    Scénario: 3 tokens avec d_k = 4
    """
    print("\n" + "="*60)
    print("EXEMPLE: Scaled Dot-Product Attention from Scratch")
    print("="*60 + "\n")
    
    # Paramètres
    batch_size = 1
    seq_len = 3  # 3 tokens
    d_k = 4      # dimension des clés/queries
    d_v = 4      # dimension des valeurs
    
    # Créer des matrices Q, K, V simples pour la démonstration
    np.random.seed(42)
    Q = np.random.randn(batch_size, seq_len, d_k)
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_v)
    
    print("Input shapes:")
    print(f"  Q (Query):  {Q.shape} - 'Ce que je cherche'")
    print(f"  K (Key):    {K.shape} - 'Ce que j'offre'")
    print(f"  V (Value):  {V.shape} - 'L'information'")
    print()
    
    # Calculer l'attention
    output, attention_weights = scaled_dot_product_attention_from_scratch(Q, K, V)
    
    print("\nOutput shapes:")
    print(f"  Output:            {output.shape}")
    print(f"  Attention weights: {attention_weights.shape}")
    
    print("\nAttention weights (batch 0):")
    print(attention_weights[0])
    print("\nInterprétation:")
    print("  - Ligne i = distribution d'attention du token i")
    print("  - Colonne j = combien les autres tokens regardent le token j")
    print("  - Chaque ligne somme à 1.0 (probabilités)")
    
    # Vérifier la normalisation
    row_sums = attention_weights.sum(axis=-1)
    print(f"\nVérification: somme de chaque ligne = {row_sums[0]}")
    print(f"Toutes les sommes ≈ 1.0? {np.allclose(row_sums, 1.0)}")
    
    return output, attention_weights


# ============================================
# IMPLEMENTATION 2: PyTorch (nn.Module)
# ============================================
# Objectif: Code professionnel et efficace


class ScaledDotProductAttention(nn.Module):
    """
    Implémentation PyTorch professionnelle de l'attention Scaled Dot-Product.
    
    Cette classe utilise les opérations PyTorch optimisées pour le GPU et
    fournit une implémentation stable numériquement et efficace.
    
    Formule:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    
    Méthodes PyTorch utilisées:
        - torch.matmul(): Multiplication matricielle optimisée GPU avec broadcasting
        - masked_fill(): Remplace les valeurs selon un masque (opération in-place)
        - F.softmax(): Softmax stable numériquement avec contrôle de dimension
    
    Args:
        d_k: Dimension des clés (utilisée pour la normalisation)
    
    Attributes:
        d_k: Dimension des clés stockée pour le scaling
        scale: Facteur de normalisation précalculé (1 / sqrt(d_k))
    
    Example:
        >>> attention = ScaledDotProductAttention(d_k=64)
        >>> Q = torch.randn(2, 4, 64)  # (batch_size, seq_len, d_k)
        >>> K = torch.randn(2, 4, 64)
        >>> V = torch.randn(2, 4, 64)
        >>> output, weights = attention(Q, K, V)
        >>> print(output.shape)  # torch.Size([2, 4, 64])
        >>> print(weights.shape)  # torch.Size([2, 4, 4])
    """
    
    def __init__(self, d_k: int):
        """
        Initialise le module d'attention.
        
        Args:
            d_k: Dimension des clés/queries (utilisée pour normaliser les scores)
        """
        super().__init__()
        self.d_k = d_k
        # Précalculer le facteur de scaling pour l'efficacité
        self.scale = 1.0 / math.sqrt(d_k)
    
    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcule l'attention Scaled Dot-Product.
        
        Cette méthode implémente la formule d'attention en 4 étapes:
        1. Calcul des scores: QK^T
        2. Normalisation: division par sqrt(d_k)
        3. Application du masque (optionnel)
        4. Softmax et pondération des valeurs
        
        Args:
            Q: Tensor Query de shape (batch_size, seq_len, d_k)
               Représente "ce que chaque token cherche"
            K: Tensor Key de shape (batch_size, seq_len, d_k)
               Représente "ce que chaque token offre comme information"
            V: Tensor Value de shape (batch_size, seq_len, d_v)
               Représente "l'information elle-même"
            mask: Masque optionnel de shape (seq_len, seq_len) ou (batch_size, seq_len, seq_len)
                  Valeurs: 1 = autoriser l'attention, 0 = bloquer l'attention
                  Utilisé pour le masque causal (GPT) ou le masque de padding
        
        Returns:
            output: Sortie pondérée de shape (batch_size, seq_len, d_v)
                   Représente l'information agrégée selon les poids d'attention
            attention_weights: Poids d'attention de shape (batch_size, seq_len, seq_len)
                              Matrice de probabilités (chaque ligne somme à 1.0)
                              weights[i,j] = "combien le token i prête attention au token j"
        
        Shape Flow:
            Q: (batch_size, seq_len, d_k)
            K: (batch_size, seq_len, d_k)
            K^T: (batch_size, d_k, seq_len)
            scores: (batch_size, seq_len, seq_len)
            attention_weights: (batch_size, seq_len, seq_len)
            V: (batch_size, seq_len, d_v)
            output: (batch_size, seq_len, d_v)
        
        Notes:
            - torch.matmul() gère automatiquement le batching
            - masked_fill() remplace les positions masquées par -inf
            - F.softmax() avec dim=-1 normalise sur la dernière dimension (les keys)
            - Les poids d'attention sont retournés pour la visualisation
        """
        
        # ========================================
        # ÉTAPE 1: Calcul des scores d'attention
        # ========================================
        # torch.matmul(): Multiplication matricielle batch-aware
        # 
        # Avantages de torch.matmul():
        # - Optimisé GPU: Utilise cuBLAS pour accélération matérielle
        # - Broadcasting automatique: Gère les dimensions batch
        # - Précision mixte: Support FP16/BF16 pour économiser mémoire
        # - Gradient efficace: Autograd optimisé pour backprop
        #
        # Q @ K^T calcule la similarité entre chaque paire de tokens
        # transpose(-2, -1) échange les deux dernières dimensions
        # (batch, seq_len, d_k) @ (batch, d_k, seq_len) -> (batch, seq_len, seq_len)
        
        scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # ========================================
        # ÉTAPE 2: Normalisation (Scaling)
        # ========================================
        # Division par sqrt(d_k) pour stabiliser les gradients
        # 
        # Pourquoi normaliser?
        # - Les produits scalaires croissent avec la dimension d_k
        # - Sans normalisation, softmax sature (gradients → 0)
        # - sqrt(d_k) est la déviation standard théorique du produit scalaire
        #   si Q et K sont des vecteurs aléatoires normalisés
        #
        # Utilisation de self.scale précalculé pour l'efficacité
        
        scores = scores * self.scale
        
        # ========================================
        # ÉTAPE 3: Application du masque (optionnel)
        # ========================================
        # masked_fill(): Remplace les valeurs selon un masque booléen
        #
        # Fonctionnement de masked_fill():
        # - Opération in-place possible (mais ici on crée un nouveau tensor)
        # - mask == 0 crée un masque booléen
        # - Les positions True sont remplacées par float('-inf')
        # - float('-inf') → softmax → 0 (attention nulle)
        #
        # Cas d'usage du masque:
        # - Masque causal (GPT): Empêche de voir le futur
        #   mask[i,j] = 0 si j > i (triangulaire supérieur)
        # - Masque de padding: Ignore les tokens de remplissage
        #   mask[i,j] = 0 si j est un token de padding
        #
        # Pourquoi -inf et pas un grand nombre négatif?
        # - -inf garantit que softmax(-inf) = 0 exactement
        # - Un grand nombre négatif (-1e9) peut causer des problèmes numériques
        # - PyTorch gère -inf correctement dans softmax
        
        if mask is not None:
            # Créer un nouveau tensor avec les valeurs masquées
            # mask == 0 identifie les positions à bloquer
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # ========================================
        # ÉTAPE 4: Calcul des poids d'attention
        # ========================================
        # F.softmax(): Fonction softmax stable numériquement
        #
        # Fonctionnement de F.softmax():
        # - dim=-1: Applique softmax sur la dernière dimension (les keys)
        # - Stabilité numérique: Soustrait automatiquement le max avant exp()
        # - Gère -inf correctement: exp(-inf) = 0
        # - Gradient efficace: Dérivée du softmax optimisée
        #
        # Résultat:
        # - Chaque ligne est une distribution de probabilité
        # - sum(attention_weights[i, :]) = 1.0 pour tout i
        # - attention_weights[i,j] ∈ [0, 1]
        #
        # Interprétation:
        # - attention_weights[i,j] = "combien le token i prête attention au token j"
        # - Valeurs élevées → forte attention
        # - Valeurs proches de 0 → attention faible/nulle
        
        attention_weights = F.softmax(scores, dim=-1)
        
        # ========================================
        # ÉTAPE 5: Pondération des valeurs
        # ========================================
        # Multiplication matricielle finale: attention_weights @ V
        #
        # Calcul:
        # - Pour chaque token i, on calcule une somme pondérée des valeurs
        # - output[i] = sum_j attention_weights[i,j] * V[j]
        # - C'est une moyenne pondérée des valeurs selon l'attention
        #
        # Dimensions:
        # - attention_weights: (batch, seq_len, seq_len)
        # - V: (batch, seq_len, d_v)
        # - output: (batch, seq_len, d_v)
        #
        # Interprétation:
        # - Chaque token reçoit une combinaison des valeurs des autres tokens
        # - Les poids déterminent l'importance de chaque valeur
        # - C'est le mécanisme qui permet au modèle de "lire" le contexte
        
        output = torch.matmul(attention_weights, V)
        
        # Retourner à la fois la sortie et les poids d'attention
        # Les poids sont utiles pour la visualisation et l'interprétation
        return output, attention_weights


# ============================================
# EXEMPLE D'UTILISATION PYTORCH
# ============================================

def example_attention_pytorch():
    """
    Exemple pédagogique avec PyTorch pour comprendre l'attention.
    
    Compare l'implémentation NumPy et PyTorch sur les mêmes données.
    """
    print("\n" + "="*60)
    print("EXEMPLE: Scaled Dot-Product Attention avec PyTorch")
    print("="*60 + "\n")
    
    # Paramètres
    batch_size = 2
    seq_len = 4
    d_k = 8
    d_v = 8
    
    # Créer des tensors PyTorch
    torch.manual_seed(42)
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_v)
    
    print("Input shapes (PyTorch tensors):")
    print(f"  Q (Query):  {Q.shape}")
    print(f"  K (Key):    {K.shape}")
    print(f"  V (Value):  {V.shape}")
    print(f"  Device: {Q.device}")
    print()
    
    # Créer le module d'attention
    attention = ScaledDotProductAttention(d_k=d_k)
    
    # Calculer l'attention
    output, attention_weights = attention(Q, K, V)
    
    print("Output shapes:")
    print(f"  Output:            {output.shape}")
    print(f"  Attention weights: {attention_weights.shape}")
    
    print("\nAttention weights (batch 0):")
    print(attention_weights[0])
    
    # Vérifier la normalisation
    row_sums = attention_weights.sum(dim=-1)
    print("\nVérification: somme de chaque ligne (batch 0):")
    print(row_sums[0])
    print(f"Toutes les sommes ≈ 1.0? {torch.allclose(row_sums, torch.ones_like(row_sums))}")
    
    # Tester avec GPU si disponible
    if torch.cuda.is_available():
        print("\n" + "-"*60)
        print("Test avec GPU")
        print("-"*60)
        
        Q_gpu = Q.cuda()
        K_gpu = K.cuda()
        V_gpu = V.cuda()
        attention_gpu = attention.cuda()
        
        output_gpu, weights_gpu = attention_gpu(Q_gpu, K_gpu, V_gpu)
        
        print(f"Device: {output_gpu.device}")
        print(f"Output shape: {output_gpu.shape}")
        
        # Vérifier que les résultats CPU et GPU sont identiques
        output_cpu = output_gpu.cpu()
        print(f"CPU vs GPU match? {torch.allclose(output, output_cpu, atol=1e-5)}")
    
    return output, attention_weights


def example_attention_with_causal_mask_pytorch():
    """
    Exemple d'attention avec masque causal (pour GPT).
    
    Démontre comment le masque empêche les tokens de voir le futur.
    """
    print("\n" + "="*60)
    print("EXEMPLE: Attention avec Masque Causal (PyTorch)")
    print("="*60 + "\n")
    
    # Paramètres
    batch_size = 1
    seq_len = 5
    d_k = 8
    
    # Créer des tensors
    torch.manual_seed(42)
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_k)
    
    # Créer un masque causal avec torch.tril()
    # torch.tril(): Retourne la partie triangulaire inférieure d'une matrice
    # 
    # Fonctionnement:
    # - torch.ones(seq_len, seq_len) crée une matrice de 1
    # - torch.tril() garde seulement la partie inférieure (diagonal incluse)
    # - Résultat: matrice triangulaire inférieure
    #
    # Exemple pour seq_len=5:
    # [[1, 0, 0, 0, 0],
    #  [1, 1, 0, 0, 0],
    #  [1, 1, 1, 0, 0],
    #  [1, 1, 1, 1, 0],
    #  [1, 1, 1, 1, 1]]
    #
    # Interprétation:
    # - Ligne i peut voir les tokens 0 à i (passé + présent)
    # - Ligne i ne peut pas voir les tokens i+1 à seq_len-1 (futur)
    
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    
    print("Masque causal (1=autorisé, 0=bloqué):")
    print(causal_mask)
    print("\nInterprétation:")
    print("  - Chaque ligne représente un token")
    print("  - 1 = peut voir ce token, 0 = ne peut pas voir")
    print("  - Token i peut voir tokens 0 à i (autorégressif)")
    
    # Créer le module d'attention
    attention = ScaledDotProductAttention(d_k=d_k)
    
    # Calculer l'attention SANS masque
    print("\n" + "-"*60)
    print("Attention SANS masque (bidirectionnelle - BERT)")
    print("-"*60)
    output_no_mask, weights_no_mask = attention(Q, K, V, mask=None)
    print("\nPoids d'attention (tous les tokens se voient):")
    print(weights_no_mask[0])
    
    # Calculer l'attention AVEC masque causal
    print("\n" + "-"*60)
    print("Attention AVEC masque causal (autorégressif - GPT)")
    print("-"*60)
    output_masked, weights_masked = attention(Q, K, V, mask=causal_mask)
    print("\nPoids d'attention (masque causal appliqué):")
    print(weights_masked[0])
    print("\nObservation:")
    print("  - La partie supérieure droite est nulle (pas d'attention au futur)")
    print("  - Chaque token ne voit que lui-même et les tokens précédents")
    print("  - C'est le mécanisme clé de GPT pour la génération autoregressive")
    
    # Visualiser la différence
    print("\n" + "-"*60)
    print("Comparaison: Nombre de tokens visibles par position")
    print("-"*60)
    for i in range(seq_len):
        visible_no_mask = (weights_no_mask[0, i] > 0).sum().item()
        visible_masked = (weights_masked[0, i] > 0).sum().item()
        print(f"  Position {i}: Sans masque={visible_no_mask}, Avec masque={visible_masked}")
    
    return output_masked, weights_masked


if __name__ == "__main__":
    # Exécuter l'exemple from-scratch
    example_attention_from_scratch()
    
    print("\n" + "="*60)
    print("EXEMPLE AVEC MASQUE CAUSAL (NumPy)")
    print("="*60 + "\n")
    
    # Exemple avec masque causal (pour GPT) - NumPy
    batch_size = 1
    seq_len = 4
    d_k = 8
    
    Q = np.random.randn(batch_size, seq_len, d_k)
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_k)
    
    # Créer un masque causal (triangulaire inférieur)
    # 1 = autoriser, 0 = bloquer
    causal_mask = np.tril(np.ones((seq_len, seq_len)))
    print("Masque causal (1=autorisé, 0=bloqué):")
    print(causal_mask)
    print("\nInterprétation: Chaque token ne peut voir que lui-même et les tokens précédents")
    
    output, attention_weights = scaled_dot_product_attention_from_scratch(
        Q, K, V, mask=causal_mask
    )
    
    print("\nPoids d'attention avec masque causal:")
    print(attention_weights[0])
    print("\nObservation: La partie supérieure droite est nulle (pas d'attention au futur)")
    
    # Exécuter les exemples PyTorch
    example_attention_pytorch()
    example_attention_with_causal_mask_pytorch()
