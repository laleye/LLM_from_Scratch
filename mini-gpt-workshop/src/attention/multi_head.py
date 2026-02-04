"""
Multi-Head Attention Implementation

Ce module implémente le mécanisme de Multi-Head Attention, qui est au cœur
des transformers. L'idée est d'appliquer l'attention plusieurs fois en parallèle
avec différentes projections, permettant au modèle de capturer différents types
de relations entre les tokens.

Concept clé:
    Au lieu d'une seule attention, on utilise plusieurs "têtes" (heads) qui
    regardent différents aspects des relations entre tokens.

Formule mathématique:
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    où head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)

Architecture:
    1. Projections linéaires: Q, K, V → d_model dimensions
    2. Division en têtes: d_model → num_heads × d_k
    3. Attention par tête: Chaque tête calcule son attention indépendamment
    4. Concaténation: Combiner toutes les têtes
    5. Projection finale: Retour à d_model dimensions

Pourquoi multi-head?
    - Chaque tête peut se spécialiser dans différents patterns
    - Tête 1: Relations syntaxiques (sujet-verbe)
    - Tête 2: Relations sémantiques (coréférences)
    - Tête 3: Relations positionnelles (proximité)
    - etc.
"""

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# Importer l'attention scaled dot-product
try:
    from .scaled_dot_product import (
        scaled_dot_product_attention_from_scratch,
        ScaledDotProductAttention
    )
except ImportError:
    # Pour l'exécution standalone
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from attention.scaled_dot_product import (
        scaled_dot_product_attention_from_scratch,
        ScaledDotProductAttention
    )


# ============================================
# IMPLEMENTATION 1: From Scratch (NumPy)
# ============================================
# Objectif: Comprendre le formalisme mathématique


def multi_head_attention_from_scratch(
    x: np.ndarray,
    W_q: np.ndarray,
    W_k: np.ndarray,
    W_v: np.ndarray,
    W_o: np.ndarray,
    num_heads: int,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implémentation from-scratch de Multi-Head Attention avec NumPy.
    
    Cette fonction décompose le multi-head attention en étapes explicites:
    1. Projections linéaires (Q, K, V)
    2. Division en têtes (reshape)
    3. Attention par tête (boucle)
    4. Concaténation des têtes
    5. Projection finale
    
    Formule complète:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
        où head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
    
    Dimensions:
        - d_model: Dimension du modèle (ex: 512)
        - num_heads: Nombre de têtes (ex: 8)
        - d_k = d_model / num_heads: Dimension par tête (ex: 64)
    
    Args:
        x: Input de shape (batch_size, seq_len, d_model)
           Représente les embeddings d'entrée
        W_q: Matrice de projection Query de shape (d_model, d_model)
        W_k: Matrice de projection Key de shape (d_model, d_model)
        W_v: Matrice de projection Value de shape (d_model, d_model)
        W_o: Matrice de projection finale de shape (d_model, d_model)
        num_heads: Nombre de têtes d'attention
        mask: Masque optionnel de shape (seq_len, seq_len)
    
    Returns:
        output: Sortie de shape (batch_size, seq_len, d_model)
        attention_weights: Liste de poids d'attention pour chaque tête
                          Chaque élément a shape (batch_size, seq_len, seq_len)
    
    Example:
        >>> batch_size, seq_len, d_model = 2, 4, 8
        >>> num_heads = 2
        >>> x = np.random.randn(batch_size, seq_len, d_model)
        >>> W_q = np.random.randn(d_model, d_model)
        >>> W_k = np.random.randn(d_model, d_model)
        >>> W_v = np.random.randn(d_model, d_model)
        >>> W_o = np.random.randn(d_model, d_model)
        >>> output, weights = multi_head_attention_from_scratch(
        ...     x, W_q, W_k, W_v, W_o, num_heads
        ... )
        >>> print(output.shape)  # (2, 4, 8)
    """
    
    batch_size, seq_len, d_model = x.shape
    
    # Vérifier que d_model est divisible par num_heads
    assert d_model % num_heads == 0, \
        f"d_model ({d_model}) doit être divisible par num_heads ({num_heads})"
    
    # Calculer la dimension par tête
    d_k = d_model // num_heads
    
    print(f"\n[Multi-Head Attention] Configuration:")
    print(f"  - Input shape: {x.shape}")
    print(f"  - d_model: {d_model}")
    print(f"  - num_heads: {num_heads}")
    print(f"  - d_k (dimension par tête): {d_k}")
    
    # ========================================
    # ÉTAPE 1: Projections linéaires
    # ========================================
    # Formule: Q = XW^Q, K = XW^K, V = XW^V
    # 
    # Intuition: Transformer l'input en queries, keys, et values
    # 
    # x shape: (batch_size, seq_len, d_model)
    # W_q shape: (d_model, d_model)
    # Q shape: (batch_size, seq_len, d_model)
    #
    # Note: On projette vers d_model dimensions, puis on divisera en têtes
    
    Q = np.matmul(x, W_q)  # (batch, seq_len, d_model)
    K = np.matmul(x, W_k)  # (batch, seq_len, d_model)
    V = np.matmul(x, W_v)  # (batch, seq_len, d_model)
    
    print(f"\n[Étape 1] Projections linéaires:")
    print(f"  - Q shape: {Q.shape}")
    print(f"  - K shape: {K.shape}")
    print(f"  - V shape: {V.shape}")
    
    # ========================================
    # ÉTAPE 2: Division en têtes (reshape)
    # ========================================
    # Formule: Reshape de (batch, seq_len, d_model) 
    #          vers (batch, seq_len, num_heads, d_k)
    #          puis transpose vers (batch, num_heads, seq_len, d_k)
    # 
    # Intuition: Diviser les d_model dimensions en num_heads groupes de d_k
    # 
    # Exemple avec d_model=8, num_heads=2, d_k=4:
    #   [a1, a2, a3, a4, a5, a6, a7, a8]
    #   → [[a1, a2, a3, a4], [a5, a6, a7, a8]]  (2 têtes de 4 dimensions)
    #
    # Pourquoi reshape puis transpose?
    # - reshape: Organise les dimensions en groupes
    # - transpose: Met les têtes en dimension 1 pour traiter en parallèle
    
    # Reshape: (batch, seq_len, d_model) → (batch, seq_len, num_heads, d_k)
    Q_heads = Q.reshape(batch_size, seq_len, num_heads, d_k)
    K_heads = K.reshape(batch_size, seq_len, num_heads, d_k)
    V_heads = V.reshape(batch_size, seq_len, num_heads, d_k)
    
    # Transpose: (batch, seq_len, num_heads, d_k) → (batch, num_heads, seq_len, d_k)
    # Cela met les têtes en dimension 1, ce qui facilite le traitement par tête
    Q_heads = Q_heads.transpose(0, 2, 1, 3)  # axes: (0, 1, 2, 3) → (0, 2, 1, 3)
    K_heads = K_heads.transpose(0, 2, 1, 3)
    V_heads = V_heads.transpose(0, 2, 1, 3)
    
    print(f"\n[Étape 2] Division en têtes:")
    print(f"  - Après reshape: (batch, seq_len, num_heads, d_k)")
    print(f"  - Après transpose: {Q_heads.shape} (batch, num_heads, seq_len, d_k)")
    
    # ========================================
    # ÉTAPE 3: Attention par tête (boucle)
    # ========================================
    # Formule: head_i = Attention(Q_i, K_i, V_i)
    # 
    # Intuition: Chaque tête calcule son attention indépendamment
    # 
    # Pour chaque tête i:
    # - Extraire Q_i, K_i, V_i de shape (batch, seq_len, d_k)
    # - Calculer l'attention scaled dot-product
    # - Stocker le résultat
    #
    # Pourquoi une boucle?
    # - Version pédagogique: montre explicitement le traitement par tête
    # - Version PyTorch: vectorisé (pas de boucle)
    
    print(f"\n[Étape 3] Calcul de l'attention pour chaque tête:")
    
    # Initialiser les listes pour stocker les résultats
    head_outputs = []
    all_attention_weights = []
    
    # Boucle sur chaque tête
    for i in range(num_heads):
        print(f"\n  Tête {i+1}/{num_heads}:")
        
        # Extraire les matrices Q, K, V pour cette tête
        # Q_heads shape: (batch, num_heads, seq_len, d_k)
        # Q_i shape: (batch, seq_len, d_k)
        Q_i = Q_heads[:, i, :, :]  # Sélectionner la tête i
        K_i = K_heads[:, i, :, :]
        V_i = V_heads[:, i, :, :]
        
        print(f"    - Q_i shape: {Q_i.shape}")
        
        # Calculer l'attention pour cette tête
        # Utilise la fonction scaled_dot_product_attention_from_scratch
        head_output, attention_weights = scaled_dot_product_attention_from_scratch(
            Q_i, K_i, V_i, mask=mask
        )
        
        print(f"    - Output shape: {head_output.shape}")
        
        # Stocker les résultats
        head_outputs.append(head_output)
        all_attention_weights.append(attention_weights)
    
    # ========================================
    # ÉTAPE 4: Concaténation des têtes
    # ========================================
    # Formule: Concat(head_1, ..., head_h)
    # 
    # Intuition: Combiner les sorties de toutes les têtes
    # 
    # Chaque head_output a shape: (batch, seq_len, d_k)
    # Après concaténation: (batch, seq_len, num_heads * d_k) = (batch, seq_len, d_model)
    #
    # Méthode:
    # 1. Stack: Empiler les têtes → (num_heads, batch, seq_len, d_k)
    # 2. Transpose: Réorganiser → (batch, seq_len, num_heads, d_k)
    # 3. Reshape: Aplatir les têtes → (batch, seq_len, d_model)
    
    # Stack les sorties des têtes
    # head_outputs est une liste de num_heads arrays de shape (batch, seq_len, d_k)
    # np.stack crée un array de shape (num_heads, batch, seq_len, d_k)
    stacked_heads = np.stack(head_outputs, axis=0)  # (num_heads, batch, seq_len, d_k)
    
    # Transpose pour mettre batch en premier
    # (num_heads, batch, seq_len, d_k) → (batch, num_heads, seq_len, d_k)
    stacked_heads = stacked_heads.transpose(1, 0, 2, 3)
    
    # Transpose pour mettre seq_len avant num_heads
    # (batch, num_heads, seq_len, d_k) → (batch, seq_len, num_heads, d_k)
    stacked_heads = stacked_heads.transpose(0, 2, 1, 3)
    
    # Reshape pour concaténer les têtes
    # (batch, seq_len, num_heads, d_k) → (batch, seq_len, num_heads * d_k)
    # num_heads * d_k = d_model
    concatenated = stacked_heads.reshape(batch_size, seq_len, d_model)
    
    print(f"\n[Étape 4] Concaténation des têtes:")
    print(f"  - Avant concat: {num_heads} têtes de shape (batch, seq_len, {d_k})")
    print(f"  - Après concat: {concatenated.shape} (batch, seq_len, d_model)")
    
    # ========================================
    # ÉTAPE 5: Projection finale
    # ========================================
    # Formule: Output = Concat(heads)W^O
    # 
    # Intuition: Mélanger les informations de toutes les têtes
    # 
    # concatenated shape: (batch, seq_len, d_model)
    # W_o shape: (d_model, d_model)
    # output shape: (batch, seq_len, d_model)
    #
    # Pourquoi une projection finale?
    # - Permet aux têtes d'interagir
    # - Ajoute de la capacité d'apprentissage
    # - Transforme la concaténation en une représentation unifiée
    
    output = np.matmul(concatenated, W_o)
    
    print(f"\n[Étape 5] Projection finale:")
    print(f"  - Output shape: {output.shape}")
    
    print(f"\n[Multi-Head Attention] Terminé!")
    print(f"  - Input shape: {x.shape}")
    print(f"  - Output shape: {output.shape}")
    print(f"  - Dimension préservée: {x.shape == output.shape}")
    
    return output, all_attention_weights


# ============================================
# EXEMPLE D'UTILISATION
# ============================================


def example_multi_head_attention_from_scratch():
    """
    Exemple pédagogique avec des petites matrices pour comprendre multi-head attention.
    
    Scénario: 3 tokens, d_model=8, 2 têtes
    """
    print("\n" + "="*70)
    print("EXEMPLE: Multi-Head Attention from Scratch")
    print("="*70)
    
    # Paramètres
    batch_size = 1
    seq_len = 3      # 3 tokens
    d_model = 8      # dimension du modèle
    num_heads = 2    # 2 têtes d'attention
    
    # Créer l'input
    np.random.seed(42)
    x = np.random.randn(batch_size, seq_len, d_model)
    
    # Créer les matrices de projection
    # Dans un vrai modèle, ces matrices sont apprises pendant l'entraînement
    W_q = np.random.randn(d_model, d_model) * 0.1
    W_k = np.random.randn(d_model, d_model) * 0.1
    W_v = np.random.randn(d_model, d_model) * 0.1
    W_o = np.random.randn(d_model, d_model) * 0.1
    
    print(f"\nConfiguration:")
    print(f"  - batch_size: {batch_size}")
    print(f"  - seq_len: {seq_len}")
    print(f"  - d_model: {d_model}")
    print(f"  - num_heads: {num_heads}")
    print(f"  - d_k (par tête): {d_model // num_heads}")
    
    # Calculer multi-head attention
    output, attention_weights = multi_head_attention_from_scratch(
        x, W_q, W_k, W_v, W_o, num_heads
    )
    
    print("\n" + "="*70)
    print("RÉSULTATS")
    print("="*70)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Nombre de têtes: {len(attention_weights)}")
    
    print(f"\nPoids d'attention par tête:")
    for i, weights in enumerate(attention_weights):
        print(f"\n  Tête {i+1}:")
        print(f"    Shape: {weights.shape}")
        print(f"    Poids (batch 0):")
        print(f"{weights[0]}")
        
        # Vérifier la normalisation
        row_sums = weights[0].sum(axis=-1)
        print(f"    Somme par ligne: {row_sums}")
        print(f"    Normalisé? {np.allclose(row_sums, 1.0)}")
    
    print(f"\n" + "="*70)
    print("INTERPRÉTATION")
    print("="*70)
    print(f"""
    Multi-Head Attention permet au modèle de capturer différents types de relations:
    
    - Tête 1 pourrait se spécialiser dans les relations syntaxiques
      (ex: sujet-verbe, déterminant-nom)
    
    - Tête 2 pourrait se spécialiser dans les relations sémantiques
      (ex: coréférences, relations thématiques)
    
    Chaque tête apprend à regarder différents aspects du contexte!
    """)
    
    return output, attention_weights


def example_multi_head_with_causal_mask():
    """
    Exemple de multi-head attention avec masque causal (pour GPT).
    """
    print("\n" + "="*70)
    print("EXEMPLE: Multi-Head Attention avec Masque Causal")
    print("="*70)
    
    # Paramètres
    batch_size = 1
    seq_len = 4
    d_model = 8
    num_heads = 2
    
    # Créer l'input
    np.random.seed(42)
    x = np.random.randn(batch_size, seq_len, d_model)
    
    # Créer les matrices de projection
    W_q = np.random.randn(d_model, d_model) * 0.1
    W_k = np.random.randn(d_model, d_model) * 0.1
    W_v = np.random.randn(d_model, d_model) * 0.1
    W_o = np.random.randn(d_model, d_model) * 0.1
    
    # Créer un masque causal
    causal_mask = np.tril(np.ones((seq_len, seq_len)))
    
    print(f"\nMasque causal:")
    print(causal_mask)
    
    # Calculer multi-head attention avec masque
    output, attention_weights = multi_head_attention_from_scratch(
        x, W_q, W_k, W_v, W_o, num_heads, mask=causal_mask
    )
    
    print("\n" + "="*70)
    print("RÉSULTATS AVEC MASQUE CAUSAL")
    print("="*70)
    
    print(f"\nPoids d'attention par tête (avec masque):")
    for i, weights in enumerate(attention_weights):
        print(f"\n  Tête {i+1}:")
        print(f"{weights[0]}")
        print(f"    Observation: Partie supérieure droite = 0 (futur bloqué)")
    
    return output, attention_weights


# ============================================
# IMPLEMENTATION 2: PyTorch (nn.Module)
# ============================================
# Objectif: Code professionnel et efficace


class MultiHeadAttention(nn.Module):
    """
    Implémentation PyTorch professionnelle de Multi-Head Attention.
    
    Cette classe implémente le mécanisme de multi-head attention utilisé dans
    les transformers. Au lieu d'une seule attention, on utilise plusieurs "têtes"
    qui regardent différents aspects des relations entre tokens.
    
    Formule mathématique:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
        où head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
    
    Architecture:
        1. Projections linéaires: Q, K, V → d_model dimensions
        2. Division en têtes: d_model → num_heads × d_k
        3. Attention par tête: Chaque tête calcule son attention en parallèle
        4. Concaténation: Combiner toutes les têtes
        5. Projection finale: Retour à d_model dimensions
    
    Méthodes PyTorch utilisées:
        - nn.Linear(): Couche linéaire (fully connected) pour les projections
        - view(): Reshape un tensor (version de reshape qui partage la mémoire)
        - transpose(): Échange deux dimensions d'un tensor
        - contiguous(): Assure que le tensor est contigu en mémoire
    
    Args:
        d_model: Dimension du modèle (ex: 512)
        num_heads: Nombre de têtes d'attention (ex: 8)
        dropout: Taux de dropout (optionnel, par défaut 0.0)
    
    Attributes:
        d_model: Dimension du modèle
        num_heads: Nombre de têtes
        d_k: Dimension par tête (d_model // num_heads)
        W_q: Projection linéaire pour Query
        W_k: Projection linéaire pour Key
        W_v: Projection linéaire pour Value
        W_o: Projection linéaire finale
        attention: Module ScaledDotProductAttention
        dropout: Couche de dropout (optionnelle)
    
    Example:
        >>> mha = MultiHeadAttention(d_model=512, num_heads=8)
        >>> x = torch.randn(2, 10, 512)  # (batch_size, seq_len, d_model)
        >>> output = mha(x)
        >>> print(output.shape)  # torch.Size([2, 10, 512])
    
    Notes:
        - d_model doit être divisible par num_heads
        - Chaque tête a une dimension d_k = d_model / num_heads
        - Les têtes sont calculées en parallèle (pas de boucle)
        - L'input et l'output ont la même shape (dimension préservée)
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        """
        Initialise le module Multi-Head Attention.
        
        Args:
            d_model: Dimension du modèle (doit être divisible par num_heads)
            num_heads: Nombre de têtes d'attention
            dropout: Taux de dropout appliqué aux poids d'attention (0.0 = pas de dropout)
        
        Raises:
            AssertionError: Si d_model n'est pas divisible par num_heads
        """
        super().__init__()
        
        # Vérifier que d_model est divisible par num_heads
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) doit être divisible par num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # ========================================
        # Projections linéaires avec nn.Linear()
        # ========================================
        # nn.Linear(in_features, out_features, bias=True)
        #
        # Fonctionnement:
        # - Crée une couche fully connected: y = xW^T + b
        # - W est une matrice de poids de shape (out_features, in_features)
        # - b est un vecteur de biais de shape (out_features,)
        # - Les poids sont initialisés automatiquement (Xavier/Kaiming)
        #
        # Avantages:
        # - Gestion automatique des gradients (autograd)
        # - Initialisation optimale des poids
        # - Support GPU natif
        # - Optimisations CUDA pour les grandes matrices
        #
        # Dans notre cas:
        # - in_features = d_model (dimension d'entrée)
        # - out_features = d_model (on projette vers la même dimension)
        # - Chaque tête aura ensuite d_k dimensions après le split
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Module d'attention scaled dot-product
        self.attention = ScaledDotProductAttention(self.d_k)
        
        # Dropout optionnel sur les poids d'attention
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calcule le Multi-Head Attention.
        
        Cette méthode implémente l'attention multi-têtes en 5 étapes:
        1. Projections linéaires (Q, K, V)
        2. Division en têtes (reshape + transpose)
        3. Attention par tête (parallèle, pas de boucle)
        4. Concaténation des têtes
        5. Projection finale
        
        Args:
            x: Input tensor de shape (batch_size, seq_len, d_model)
               Représente les embeddings d'entrée
            mask: Masque optionnel de shape (seq_len, seq_len)
                  1 = autoriser l'attention, 0 = bloquer l'attention
                  Utilisé pour le masque causal (GPT) ou le masque de padding
        
        Returns:
            output: Tensor de shape (batch_size, seq_len, d_model)
                   Représente l'information agrégée par toutes les têtes
        
        Shape Flow:
            Input x: (batch_size, seq_len, d_model)
            
            Après projections:
            Q, K, V: (batch_size, seq_len, d_model)
            
            Après split en têtes:
            Q, K, V: (batch_size, num_heads, seq_len, d_k)
            
            Après attention:
            output: (batch_size, num_heads, seq_len, d_k)
            
            Après concat:
            output: (batch_size, seq_len, d_model)
            
            Après projection finale:
            output: (batch_size, seq_len, d_model)
        
        Notes:
            - Toutes les têtes sont calculées en parallèle (vectorisé)
            - view() et transpose() ne copient pas les données (efficace)
            - contiguous() est nécessaire avant view() après transpose()
        """
        
        batch_size, seq_len, _ = x.shape
        
        # ========================================
        # ÉTAPE 1: Projections linéaires
        # ========================================
        # Formule: Q = XW^Q, K = XW^K, V = XW^V
        #
        # self.W_q(x) applique la transformation linéaire:
        # - x shape: (batch_size, seq_len, d_model)
        # - W_q poids: (d_model, d_model)
        # - Q shape: (batch_size, seq_len, d_model)
        #
        # PyTorch gère automatiquement le batching:
        # - Applique la même transformation à chaque élément du batch
        # - Optimisé pour le GPU avec cuBLAS
        #
        # Pourquoi projeter vers d_model puis diviser?
        # - Flexibilité: Chaque tête peut apprendre des projections différentes
        # - Efficacité: Une seule multiplication matricielle au lieu de num_heads
        
        Q = self.W_q(x)  # (batch_size, seq_len, d_model)
        K = self.W_k(x)  # (batch_size, seq_len, d_model)
        V = self.W_v(x)  # (batch_size, seq_len, d_model)
        
        # ========================================
        # ÉTAPE 2: Division en têtes
        # ========================================
        # Objectif: Transformer (batch, seq_len, d_model)
        #           en (batch, num_heads, seq_len, d_k)
        #
        # Méthode en 2 sous-étapes:
        # 1. view(): Reshape pour créer la dimension des têtes
        # 2. transpose(): Réorganiser les dimensions
        
        # Sous-étape 2.1: view() pour créer la dimension des têtes
        # --------------------------------------------------------
        # view(batch_size, seq_len, num_heads, d_k)
        #
        # Fonctionnement de view():
        # - Reshape un tensor sans copier les données (partage la mémoire)
        # - Nécessite que le tensor soit contigu en mémoire
        # - Plus efficace que reshape() car pas de copie
        # - Équivalent à reshape() de NumPy mais avec vérification de contiguïté
        #
        # Transformation:
        # (batch, seq_len, d_model) → (batch, seq_len, num_heads, d_k)
        # où d_model = num_heads * d_k
        #
        # Exemple avec d_model=8, num_heads=2, d_k=4:
        # [a1, a2, a3, a4, a5, a6, a7, a8]
        # → [[a1, a2, a3, a4], [a5, a6, a7, a8]]  (2 têtes de 4 dimensions)
        #
        # Pourquoi cette shape?
        # - Organise les d_model dimensions en num_heads groupes de d_k
        # - Prépare pour le transpose qui mettra les têtes en dimension 1
        
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # Sous-étape 2.2: transpose() pour réorganiser les dimensions
        # -----------------------------------------------------------
        # transpose(1, 2) échange les dimensions 1 et 2
        #
        # Fonctionnement de transpose():
        # - Échange deux dimensions d'un tensor
        # - Ne copie pas les données (vue sur les mêmes données)
        # - Rend le tensor non-contigu en mémoire
        # - Nécessite contiguous() avant certaines opérations (comme view)
        #
        # Transformation:
        # (batch, seq_len, num_heads, d_k) → (batch, num_heads, seq_len, d_k)
        # dimensions: (0, 1, 2, 3) → (0, 2, 1, 3)
        #
        # Pourquoi transpose?
        # - Met les têtes en dimension 1 (après batch)
        # - Permet de traiter toutes les têtes en parallèle
        # - L'attention sera appliquée sur les dimensions (seq_len, d_k)
        # - Chaque tête est indépendante dans la dimension 1
        #
        # Exemple de layout mémoire:
        # Avant: [batch][seq][head][d_k]
        # Après: [batch][head][seq][d_k]
        # → Toutes les têtes sont maintenant "à plat" pour le traitement parallèle
        
        Q = Q.transpose(1, 2)  # (batch, num_heads, seq_len, d_k)
        K = K.transpose(1, 2)  # (batch, num_heads, seq_len, d_k)
        V = V.transpose(1, 2)  # (batch, num_heads, seq_len, d_k)
        
        # ========================================
        # ÉTAPE 3: Attention par tête (parallèle)
        # ========================================
        # Calcul de l'attention pour toutes les têtes en parallèle
        #
        # self.attention(Q, K, V, mask) applique l'attention:
        # - Q, K, V shape: (batch, num_heads, seq_len, d_k)
        # - PyTorch traite automatiquement les dimensions batch et num_heads
        # - Chaque tête calcule son attention indépendamment
        # - Pas de boucle explicite (vectorisé)
        #
        # Avantages du calcul parallèle:
        # - Beaucoup plus rapide que la boucle (from-scratch)
        # - Utilise pleinement le GPU (parallélisme massif)
        # - Moins de code, plus lisible
        # - Gradients calculés efficacement pour toutes les têtes
        #
        # Dimensions:
        # - Input: (batch, num_heads, seq_len, d_k)
        # - Output: (batch, num_heads, seq_len, d_k)
        # - Attention weights: (batch, num_heads, seq_len, seq_len)
        #
        # Note sur le masque:
        # - Le masque est broadcasté automatiquement sur la dimension num_heads
        # - Toutes les têtes utilisent le même masque
        
        output, attention_weights = self.attention(Q, K, V, mask)
        # output shape: (batch, num_heads, seq_len, d_k)
        
        # Dropout optionnel sur les poids d'attention
        if self.dropout is not None:
            attention_weights = self.dropout(attention_weights)
        
        # ========================================
        # ÉTAPE 4: Concaténation des têtes
        # ========================================
        # Objectif: Transformer (batch, num_heads, seq_len, d_k)
        #           en (batch, seq_len, d_model)
        #
        # Méthode en 2 sous-étapes:
        # 1. transpose(): Remettre seq_len avant num_heads
        # 2. contiguous() + view(): Aplatir les têtes
        
        # Sous-étape 4.1: transpose() pour réorganiser
        # --------------------------------------------
        # transpose(1, 2) échange les dimensions 1 et 2
        #
        # Transformation:
        # (batch, num_heads, seq_len, d_k) → (batch, seq_len, num_heads, d_k)
        # dimensions: (0, 1, 2, 3) → (0, 2, 1, 3)
        #
        # Pourquoi transpose?
        # - Remet seq_len en dimension 1 (comme l'input)
        # - Prépare pour la concaténation des têtes
        # - Les têtes sont maintenant adjacentes en mémoire pour chaque position
        
        output = output.transpose(1, 2)  # (batch, seq_len, num_heads, d_k)
        
        # Sous-étape 4.2: contiguous() + view() pour concaténer
        # ------------------------------------------------------
        # contiguous(): Rend le tensor contigu en mémoire
        #
        # Pourquoi contiguous()?
        # - Après transpose(), le tensor n'est plus contigu en mémoire
        # - view() nécessite un tensor contigu
        # - contiguous() crée une copie avec un layout mémoire contigu
        # - Coût: Une copie mémoire, mais nécessaire pour view()
        #
        # Fonctionnement:
        # - Vérifie si le tensor est déjà contigu
        # - Si oui: retourne le même tensor (pas de copie)
        # - Si non: crée une copie avec layout contigu
        #
        # view(batch_size, seq_len, d_model): Aplatit les têtes
        #
        # Transformation:
        # (batch, seq_len, num_heads, d_k) → (batch, seq_len, num_heads * d_k)
        # où num_heads * d_k = d_model
        #
        # Exemple avec num_heads=2, d_k=4:
        # [[a1, a2, a3, a4], [a5, a6, a7, a8]]
        # → [a1, a2, a3, a4, a5, a6, a7, a8]  (concaténation)
        #
        # Résultat:
        # - Les sorties de toutes les têtes sont concaténées
        # - Retour à la dimension d_model
        # - Prêt pour la projection finale
        
        output = output.contiguous().view(batch_size, seq_len, self.d_model)
        # output shape: (batch, seq_len, d_model)
        
        # ========================================
        # ÉTAPE 5: Projection finale
        # ========================================
        # Formule: Output = Concat(heads)W^O
        #
        # self.W_o(output) applique la transformation linéaire finale:
        # - output shape: (batch, seq_len, d_model)
        # - W_o poids: (d_model, d_model)
        # - résultat shape: (batch, seq_len, d_model)
        #
        # Pourquoi une projection finale?
        # - Permet aux têtes d'interagir et de se combiner
        # - Ajoute de la capacité d'apprentissage
        # - Transforme la concaténation en une représentation unifiée
        # - Les poids W_o sont appris pendant l'entraînement
        #
        # Résultat final:
        # - Même shape que l'input: (batch, seq_len, d_model)
        # - Dimension préservée (propriété importante des transformers)
        # - Contient l'information agrégée de toutes les têtes
        
        output = self.W_o(output)
        
        return output
    
    def get_attention_weights(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extrait les poids d'attention pour chaque tête (pour visualisation).
        
        Cette méthode est utile pour visualiser et interpréter ce que chaque
        tête d'attention regarde. Elle effectue le même calcul que forward()
        mais retourne les poids d'attention au lieu de la sortie.
        
        Args:
            x: Input tensor de shape (batch_size, seq_len, d_model)
            mask: Masque optionnel de shape (seq_len, seq_len)
        
        Returns:
            attention_weights: Tensor de shape (batch_size, num_heads, seq_len, seq_len)
                              Poids d'attention pour chaque tête
                              weights[b, h, i, j] = "combien la tête h du batch b,
                                                     token i prête attention au token j"
        
        Example:
            >>> mha = MultiHeadAttention(d_model=512, num_heads=8)
            >>> x = torch.randn(1, 10, 512)
            >>> weights = mha.get_attention_weights(x)
            >>> print(weights.shape)  # torch.Size([1, 8, 10, 10])
            >>> # Visualiser la tête 0
            >>> import matplotlib.pyplot as plt
            >>> plt.imshow(weights[0, 0].detach().numpy())
            >>> plt.colorbar()
            >>> plt.show()
        """
        
        batch_size, seq_len, _ = x.shape
        
        # Projections linéaires
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Division en têtes
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Calculer l'attention et extraire les poids
        _, attention_weights = self.attention(Q, K, V, mask)
        
        return attention_weights


# ============================================
# EXEMPLES D'UTILISATION PYTORCH
# ============================================


def example_multi_head_attention_pytorch():
    """
    Exemple pédagogique avec PyTorch pour comprendre multi-head attention.
    
    Compare l'implémentation NumPy et PyTorch sur les mêmes données.
    """
    print("\n" + "="*70)
    print("EXEMPLE: Multi-Head Attention avec PyTorch")
    print("="*70)
    
    # Paramètres
    batch_size = 2
    seq_len = 4
    d_model = 8
    num_heads = 2
    
    # Créer des tensors PyTorch
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\nConfiguration:")
    print(f"  - batch_size: {batch_size}")
    print(f"  - seq_len: {seq_len}")
    print(f"  - d_model: {d_model}")
    print(f"  - num_heads: {num_heads}")
    print(f"  - d_k (par tête): {d_model // num_heads}")
    
    print(f"\nInput shape: {x.shape}")
    print(f"Device: {x.device}")
    
    # Créer le module Multi-Head Attention
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    
    print(f"\nModèle créé:")
    print(f"  - Paramètres: {sum(p.numel() for p in mha.parameters())}")
    print(f"  - W_q shape: {mha.W_q.weight.shape}")
    print(f"  - W_k shape: {mha.W_k.weight.shape}")
    print(f"  - W_v shape: {mha.W_v.weight.shape}")
    print(f"  - W_o shape: {mha.W_o.weight.shape}")
    
    # Calculer multi-head attention
    output = mha(x)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Dimension préservée? {x.shape == output.shape}")
    
    # Extraire les poids d'attention
    attention_weights = mha.get_attention_weights(x)
    
    print(f"\nPoids d'attention shape: {attention_weights.shape}")
    print(f"  - batch_size: {attention_weights.shape[0]}")
    print(f"  - num_heads: {attention_weights.shape[1]}")
    print(f"  - seq_len × seq_len: {attention_weights.shape[2]} × {attention_weights.shape[3]}")
    
    print(f"\nPoids d'attention par tête (batch 0):")
    for i in range(num_heads):
        print(f"\n  Tête {i+1}:")
        print(attention_weights[0, i])
        
        # Vérifier la normalisation
        row_sums = attention_weights[0, i].sum(dim=-1)
        print(f"    Somme par ligne: {row_sums}")
        print(f"    Normalisé? {torch.allclose(row_sums, torch.ones_like(row_sums))}")
    
    # Tester avec GPU si disponible
    if torch.cuda.is_available():
        print("\n" + "-"*70)
        print("Test avec GPU")
        print("-"*70)
        
        x_gpu = x.cuda()
        mha_gpu = mha.cuda()
        
        output_gpu = mha_gpu(x_gpu)
        
        print(f"Device: {output_gpu.device}")
        print(f"Output shape: {output_gpu.shape}")
        
        # Vérifier que les résultats CPU et GPU sont similaires
        # (pas identiques car initialisation différente)
        print(f"Calcul GPU réussi!")
    
    print("\n" + "="*70)
    print("INTERPRÉTATION")
    print("="*70)
    print("""
    Multi-Head Attention avec PyTorch:
    
    Avantages par rapport à l'implémentation from-scratch:
    - Calcul parallèle: Toutes les têtes en une seule opération
    - Optimisé GPU: Utilise cuBLAS pour accélération matérielle
    - Gradients automatiques: Autograd gère la backpropagation
    - Code plus court: Pas de boucles explicites
    
    Chaque tête peut se spécialiser dans différents patterns:
    - Tête 1: Relations syntaxiques (sujet-verbe)
    - Tête 2: Relations sémantiques (coréférences)
    - etc.
    
    Les poids W_q, W_k, W_v, W_o sont appris pendant l'entraînement!
    """)
    
    return output, attention_weights


def example_multi_head_with_causal_mask_pytorch():
    """
    Exemple de multi-head attention avec masque causal (pour GPT).
    """
    print("\n" + "="*70)
    print("EXEMPLE: Multi-Head Attention avec Masque Causal (PyTorch)")
    print("="*70)
    
    # Paramètres
    batch_size = 1
    seq_len = 5
    d_model = 8
    num_heads = 2
    
    # Créer des tensors
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Créer un masque causal avec torch.tril()
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    
    print(f"\nMasque causal (1=autorisé, 0=bloqué):")
    print(causal_mask)
    
    # Créer le module Multi-Head Attention
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    
    # Calculer l'attention SANS masque
    print("\n" + "-"*70)
    print("Attention SANS masque (bidirectionnelle - BERT)")
    print("-"*70)
    output_no_mask = mha(x, mask=None)
    weights_no_mask = mha.get_attention_weights(x, mask=None)
    
    print(f"\nPoids d'attention (toutes les têtes peuvent voir tous les tokens):")
    for i in range(num_heads):
        print(f"\n  Tête {i+1}:")
        print(weights_no_mask[0, i])
    
    # Calculer l'attention AVEC masque causal
    print("\n" + "-"*70)
    print("Attention AVEC masque causal (autorégressif - GPT)")
    print("-"*70)
    output_masked = mha(x, mask=causal_mask)
    weights_masked = mha.get_attention_weights(x, mask=causal_mask)
    
    print(f"\nPoids d'attention (masque causal appliqué):")
    for i in range(num_heads):
        print(f"\n  Tête {i+1}:")
        print(weights_masked[0, i])
        print(f"    Observation: Partie supérieure droite = 0 (futur bloqué)")
    
    # Visualiser la différence
    print("\n" + "-"*70)
    print("Comparaison: Nombre de tokens visibles par position")
    print("-"*70)
    for i in range(seq_len):
        visible_no_mask = (weights_no_mask[0, 0, i] > 0).sum().item()
        visible_masked = (weights_masked[0, 0, i] > 0).sum().item()
        print(f"  Position {i}: Sans masque={visible_no_mask}, Avec masque={visible_masked}")
    
    print("\n" + "="*70)
    print("INTERPRÉTATION")
    print("="*70)
    print("""
    Masque Causal dans Multi-Head Attention:
    
    - BERT (sans masque): Chaque token voit tous les autres tokens
      → Utile pour la compréhension (classification, QA)
    
    - GPT (avec masque): Chaque token ne voit que le passé
      → Nécessaire pour la génération autoregressive
    
    Le masque est appliqué à TOUTES les têtes:
    - Toutes les têtes respectent la contrainte causale
    - Mais chaque tête peut se spécialiser différemment dans le passé
    
    C'est le mécanisme clé qui permet à GPT de générer du texte!
    """)
    
    return output_masked, weights_masked


if __name__ == "__main__":
    # Exécuter l'exemple from-scratch
    example_multi_head_attention_from_scratch()
    
    # Exécuter l'exemple avec masque causal (from-scratch)
    example_multi_head_with_causal_mask()
    
    # Exécuter les exemples PyTorch
    example_multi_head_attention_pytorch()
    example_multi_head_with_causal_mask_pytorch()
