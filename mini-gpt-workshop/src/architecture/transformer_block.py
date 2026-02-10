"""
Transformer Block Implementation

Ce module implémente le TransformerBlock complet, qui est l'unité de base
des architectures transformer (BERT, GPT, etc.). Un TransformerBlock combine:
1. Multi-Head Attention
2. Layer Normalization
3. Feed-Forward Network
4. Residual Connections (skip connections)
5. Dropout pour la régularisation

Architecture d'un TransformerBlock:
    Input
      ↓
    Multi-Head Attention
      ↓
    Add & Norm (residual + LayerNorm)
      ↓
    Feed-Forward Network
      ↓
    Add & Norm (residual + LayerNorm)
      ↓
    Output

Formule mathématique:
    x' = LayerNorm(x + MultiHeadAttention(x))
    output = LayerNorm(x' + FeedForward(x'))

Concepts clés:
    - Residual Connections: Permettent au gradient de circuler facilement
    - Layer Normalization: Stabilise l'entraînement
    - Dropout: Régularisation pour éviter le surapprentissage
"""

import torch
import torch.nn as nn
from typing import Optional

# Importer les composants nécessaires
try:
    from ..attention.multi_head import MultiHeadAttention
    from .feed_forward import FeedForward
except ImportError:
    # Pour l'exécution standalone
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from attention.multi_head import MultiHeadAttention
    from architecture.feed_forward import FeedForward



# ============================================
# IMPLEMENTATION: PyTorch TransformerBlock
# ============================================
# Objectif: Assembler tous les composants en un bloc transformer complet


class TransformerBlock(nn.Module):
    """
    Bloc Transformer complet combinant attention, normalisation et feed-forward.
    
    Un TransformerBlock est l'unité de base des architectures transformer.
    Il combine plusieurs composants essentiels:
    
    1. Multi-Head Attention: Permet aux tokens de s'attendre mutuellement
    2. Layer Normalization: Normalise les activations pour stabiliser l'entraînement
    3. Feed-Forward Network: Transformations non-linéaires position-wise
    4. Residual Connections: Connexions résiduelles pour faciliter l'apprentissage
    5. Dropout: Régularisation pour éviter le surapprentissage
    
    Architecture détaillée:
        Input (x)
          ↓
        ┌─────────────────────┐
        │ Multi-Head Attention│
        └─────────────────────┘
          ↓
        Add (x + attention_output)  ← Residual Connection
          ↓
        Dropout
          ↓
        LayerNorm
          ↓ (x')
        ┌─────────────────────┐
        │ Feed-Forward Network│
        └─────────────────────┘
          ↓
        Add (x' + ff_output)  ← Residual Connection
          ↓
        Dropout
          ↓
        LayerNorm
          ↓
        Output
    
    Méthodes PyTorch utilisées:
        - nn.LayerNorm(): Normalisation par couche
          * Normalise les activations sur la dimension des features
          * Formule: y = (x - mean) / sqrt(var + eps) * gamma + beta
          * gamma et beta sont des paramètres apprenables
          * Différent de BatchNorm: normalise sur features, pas sur batch
          * Avantage: Fonctionne bien avec des petits batches
          * Utilisé dans tous les transformers modernes
        
        - nn.Dropout(): Régularisation par dropout
          * Désactive aléatoirement des neurones pendant l'entraînement
          * Probabilité p: chaque neurone a p% de chance d'être désactivé
          * Pendant l'entraînement: neurones × (1-p) actifs
          * Pendant l'évaluation: tous les neurones actifs (scaling automatique)
          * Réduit le surapprentissage en forçant la redondance
          * Automatiquement désactivé avec model.eval()
    
    Args:
        d_model: Dimension du modèle (ex: 512, 768)
        num_heads: Nombre de têtes d'attention (ex: 8, 12)
        d_ff: Dimension du feed-forward (typiquement 4 * d_model)
        dropout: Taux de dropout (ex: 0.1 = 10% des neurones désactivés)
    
    Shape:
        - Input: (batch_size, seq_len, d_model)
        - Output: (batch_size, seq_len, d_model)
        - La dimension est préservée (propriété importante!)
    
    Example:
        >>> block = TransformerBlock(d_model=512, num_heads=8, d_ff=2048)
        >>> x = torch.randn(32, 128, 512)  # (batch, seq_len, d_model)
        >>> output = block(x)
        >>> print(output.shape)  # torch.Size([32, 128, 512])
    
    Notes:
        - Les residual connections sont essentielles pour l'entraînement profond
        - LayerNorm stabilise l'entraînement et accélère la convergence
        - Dropout prévient le surapprentissage
        - L'ordre des opérations (Add & Norm) peut varier selon les implémentations
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        """
        Initialise le TransformerBlock.
        
        Args:
            d_model: Dimension du modèle (doit être divisible par num_heads)
            num_heads: Nombre de têtes d'attention
            d_ff: Dimension de la couche cachée du feed-forward
            dropout: Taux de dropout pour la régularisation
        
        Raises:
            AssertionError: Si d_model n'est pas divisible par num_heads
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout

        
        # ========================================
        # Composant 1: Multi-Head Attention
        # ========================================
        # Permet aux tokens de s'attendre mutuellement
        # Chaque tête peut se spécialiser dans différents types de relations
        self.attention = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        
        # ========================================
        # Composant 2: Layer Normalization (après attention)
        # ========================================
        # nn.LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True)
        #
        # Fonctionnement de LayerNorm:
        # - Normalise les activations sur la dimension des features (d_model)
        # - Pour chaque exemple du batch, calcule mean et variance sur d_model
        # - Formule: y = (x - mean) / sqrt(var + eps) * gamma + beta
        #   * mean: moyenne sur la dimension d_model
        #   * var: variance sur la dimension d_model
        #   * eps: petit nombre pour éviter division par zéro (1e-5)
        #   * gamma: paramètre d'échelle apprenable (initialisé à 1)
        #   * beta: paramètre de décalage apprenable (initialisé à 0)
        #
        # Différence avec BatchNorm:
        # - BatchNorm: normalise sur le batch (dimension 0)
        # - LayerNorm: normalise sur les features (dimension -1)
        # - LayerNorm fonctionne mieux pour les séquences et petits batches
        #
        # Pourquoi LayerNorm?
        # - Stabilise l'entraînement en réduisant le "covariate shift"
        # - Permet d'utiliser des learning rates plus élevés
        # - Réduit la sensibilité à l'initialisation
        # - Accélère la convergence
        #
        # Dans les transformers:
        # - Appliqué après chaque sous-couche (attention et feed-forward)
        # - Combiné avec residual connections (Add & Norm)
        #
        # Exemple de normalisation:
        # Input:  [1.0, 2.0, 3.0, 4.0]  (mean=2.5, std≈1.29)
        # Output: [-1.34, -0.45, 0.45, 1.34]  (mean≈0, std≈1)
        # Après gamma et beta: peut être ajusté pendant l'entraînement
        
        self.norm1 = nn.LayerNorm(d_model)
        
        # ========================================
        # Composant 3: Feed-Forward Network
        # ========================================
        # Transformations non-linéaires appliquées indépendamment à chaque position
        # Architecture: Linear(d_model → d_ff) → GELU → Linear(d_ff → d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout=dropout)
        
        # ========================================
        # Composant 4: Layer Normalization (après feed-forward)
        # ========================================
        # Deuxième LayerNorm appliqué après le feed-forward network
        self.norm2 = nn.LayerNorm(d_model)
        
        # ========================================
        # Composant 5: Dropout
        # ========================================
        # nn.Dropout(p=dropout, inplace=False)
        #
        # Fonctionnement du Dropout:
        # - Pendant l'entraînement (model.train()):
        #   * Chaque neurone a une probabilité p d'être désactivé (mis à 0)
        #   * Les neurones actifs sont multipliés par 1/(1-p) pour compenser
        #   * Exemple avec p=0.1: 10% des neurones sont désactivés
        #   * Les neurones désactivés changent à chaque forward pass
        #
        # - Pendant l'évaluation (model.eval()):
        #   * Tous les neurones sont actifs
        #   * Pas de scaling nécessaire (déjà fait pendant l'entraînement)
        #   * Comportement déterministe
        #
        # Pourquoi le Dropout?
        # - Régularisation: Prévient le surapprentissage (overfitting)
        # - Force la redondance: Le réseau ne peut pas dépendre d'un seul neurone
        # - Effet d'ensemble: Comme entraîner plusieurs réseaux en parallèle
        # - Améliore la généralisation sur les données de test
        #
        # Où appliquer le Dropout dans un TransformerBlock?
        # 1. Après l'attention (avant Add & Norm)
        # 2. Après le feed-forward (avant Add & Norm)
        # 3. Optionnel: Dans les sous-composants (attention, feed-forward)
        #
        # Taux de dropout typiques:
        # - 0.1 (10%): Standard pour les transformers
        # - 0.0: Pas de dropout (pour debugging ou petits datasets)
        # - 0.3-0.5: Pour des modèles très larges ou datasets petits
        #
        # Exemple de dropout:
        # Input:  [1.0, 2.0, 3.0, 4.0]  (p=0.5)
        # Mask:   [1,   0,   1,   0]    (aléatoire)
        # Output: [2.0, 0.0, 6.0, 0.0]  (scaling par 1/(1-0.5)=2)
        
        self.dropout = nn.Dropout(dropout)

    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass à travers le TransformerBlock.
        
        Ce forward pass implémente l'architecture complète du transformer block:
        1. Multi-Head Attention avec residual connection et LayerNorm
        2. Feed-Forward Network avec residual connection et LayerNorm
        
        Architecture détaillée:
            x → MultiHeadAttention → Dropout → Add(x, ·) → LayerNorm → x'
            x' → FeedForward → Dropout → Add(x', ·) → LayerNorm → output
        
        Args:
            x: Input tensor de shape (batch_size, seq_len, d_model)
               Représente les embeddings d'entrée ou la sortie du bloc précédent
            mask: Masque optionnel de shape (seq_len, seq_len)
                  1 = autoriser l'attention, 0 = bloquer l'attention
                  Utilisé pour le masque causal (GPT) ou le masque de padding
        
        Returns:
            output: Tensor de shape (batch_size, seq_len, d_model)
                   Représente la sortie du bloc transformer
        
        Shape Flow:
            Input x:                    (batch_size, seq_len, d_model)
            
            Après attention:            (batch_size, seq_len, d_model)
            Après dropout:              (batch_size, seq_len, d_model)
            Après residual (x + attn):  (batch_size, seq_len, d_model)
            Après norm1:                (batch_size, seq_len, d_model)
            
            Après feed_forward:         (batch_size, seq_len, d_model)
            Après dropout:              (batch_size, seq_len, d_model)
            Après residual (x' + ff):   (batch_size, seq_len, d_model)
            Après norm2:                (batch_size, seq_len, d_model)
        
        Notes sur les Residual Connections:
            - Formule: output = LayerNorm(x + SubLayer(x))
            - Permet au gradient de circuler directement (gradient highway)
            - Essentiel pour entraîner des réseaux profonds (>10 couches)
            - Introduit dans ResNet, adopté par tous les transformers
            - Sans residual: gradient vanishing dans les réseaux profonds
        
        Notes sur l'ordre Add & Norm:
            - Post-Norm (utilisé ici): x + LayerNorm(SubLayer(x))
              * Plus stable pour l'entraînement
              * Utilisé dans le papier original "Attention is All You Need"
            - Pre-Norm (alternative): LayerNorm(x + SubLayer(x))
              * Converge plus vite
              * Utilisé dans GPT-2, GPT-3
              * Permet d'entraîner des modèles plus profonds
        """
        
        # Vérification de la dimension d'entrée
        assert x.size(-1) == self.d_model, \
            f"Expected input dimension {self.d_model}, got {x.size(-1)}"
        
        # ========================================
        # SOUS-COUCHE 1: Multi-Head Attention
        # ========================================
        # Formule: x' = LayerNorm(x + Dropout(MultiHeadAttention(x)))
        #
        # Étapes:
        # 1. Calculer l'attention multi-têtes
        # 2. Appliquer le dropout pour régularisation
        # 3. Ajouter la residual connection (x + attention_output)
        # 4. Normaliser avec LayerNorm
        #
        # Pourquoi dans cet ordre?
        # - Attention: Capture les relations entre tokens
        # - Dropout: Régularisation (désactive aléatoirement des connexions)
        # - Residual: Permet au gradient de circuler facilement
        # - LayerNorm: Stabilise les activations pour la couche suivante
        #
        # Residual Connection (Skip Connection):
        # - Permet au réseau d'apprendre des transformations résiduelles
        # - Si l'attention n'apporte rien, le réseau peut l'ignorer (poids → 0)
        # - Gradient peut circuler directement: ∂L/∂x = ∂L/∂output + ∂L/∂attention
        # - Essentiel pour entraîner des transformers profonds (12-96 couches)
        #
        # Exemple de residual:
        # x = [1, 2, 3]
        # attention_output = [0.1, 0.2, 0.3]
        # x + attention_output = [1.1, 2.2, 3.3]  ← Residual connection
        
        # Étape 1: Calculer l'attention multi-têtes
        # self.attention(x, mask) applique multi-head attention
        # Input: (batch_size, seq_len, d_model)
        # Output: (batch_size, seq_len, d_model)
        attn_output = self.attention(x, mask)
        
        # Étape 2: Appliquer le dropout
        # Désactive aléatoirement des connexions pendant l'entraînement
        # Pas d'effet pendant l'évaluation (model.eval())
        attn_output = self.dropout(attn_output)
        
        # Étape 3: Residual connection + LayerNorm
        # x + attn_output: Ajoute l'input original (residual connection)
        # self.norm1(): Normalise le résultat
        # Résultat: x' qui sera l'input du feed-forward
        x = self.norm1(x + attn_output)

        
        # ========================================
        # SOUS-COUCHE 2: Feed-Forward Network
        # ========================================
        # Formule: output = LayerNorm(x' + Dropout(FeedForward(x')))
        #
        # Étapes:
        # 1. Appliquer le feed-forward network
        # 2. Appliquer le dropout pour régularisation
        # 3. Ajouter la residual connection (x' + ff_output)
        # 4. Normaliser avec LayerNorm
        #
        # Feed-Forward Network:
        # - Architecture: Linear(d_model → d_ff) → GELU → Linear(d_ff → d_model)
        # - Appliqué indépendamment à chaque position (position-wise)
        # - Ajoute de la capacité de transformation non-linéaire
        # - d_ff typiquement 4× plus grand que d_model (ex: 2048 vs 512)
        #
        # Pourquoi Feed-Forward après Attention?
        # - Attention: Agrège l'information entre tokens
        # - Feed-Forward: Transforme l'information de chaque token individuellement
        # - Complémentaires: Attention = communication, FFN = computation
        #
        # Exemple de transformation:
        # x' = [1.0, 2.0, 3.0]  (d_model=3)
        # Après Linear1: [0.5, 1.5, 2.5, 3.5]  (d_ff=4, expansion)
        # Après GELU: [0.35, 1.48, 2.49, 3.50]  (non-linéarité)
        # Après Linear2: [1.2, 2.1, 3.3]  (d_model=3, projection)
        
        # Étape 1: Appliquer le feed-forward network
        # self.feed_forward(x) applique FFN position-wise
        # Input: (batch_size, seq_len, d_model)
        # Output: (batch_size, seq_len, d_model)
        ff_output = self.feed_forward(x)
        
        # Étape 2: Appliquer le dropout
        # Régularisation pour éviter le surapprentissage
        ff_output = self.dropout(ff_output)
        
        # Étape 3: Residual connection + LayerNorm
        # x + ff_output: Ajoute l'input de cette sous-couche (residual)
        # self.norm2(): Normalise le résultat final
        # Résultat: output du TransformerBlock
        x = self.norm2(x + ff_output)
        
        # Vérification de la dimension de sortie
        assert x.size(-1) == self.d_model, \
            f"Expected output dimension {self.d_model}, got {x.size(-1)}"
        
        return x


# ============================================
# VALIDATION ET EXEMPLES
# ============================================


def validate_transformer_block(
    d_model: int = 256,
    num_heads: int = 8,
    d_ff: int = 1024,
    batch_size: int = 2,
    seq_len: int = 10,
    dropout: float = 0.1
) -> bool:
    """
    Valide que le TransformerBlock fonctionne correctement.
    
    Cette fonction vérifie:
    1. La dimension d'entrée est préservée en sortie
    2. Les dimensions batch et sequence sont inchangées
    3. Le bloc fonctionne avec et sans masque
    4. Le dropout est bien appliqué pendant l'entraînement
    
    Args:
        d_model: Dimension du modèle
        num_heads: Nombre de têtes d'attention
        d_ff: Dimension du feed-forward
        batch_size: Taille du batch pour le test
        seq_len: Longueur de séquence pour le test
        dropout: Taux de dropout
    
    Returns:
        True si toutes les validations passent
    
    Raises:
        AssertionError si une validation échoue
    """
    print(f"\n{'='*70}")
    print(f"Validation: TransformerBlock")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  - d_model: {d_model}")
    print(f"  - num_heads: {num_heads}")
    print(f"  - d_ff: {d_ff}")
    print(f"  - batch_size: {batch_size}")
    print(f"  - seq_len: {seq_len}")
    print(f"  - dropout: {dropout}")
    print(f"{'='*70}\n")
    
    # Créer le TransformerBlock
    print("Création du TransformerBlock...")
    block = TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
    
    # Compter les paramètres
    num_params = sum(p.numel() for p in block.parameters())
    print(f"  - Nombre de paramètres: {num_params:,}")
    
    # Créer un input de test
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"\nInput shape: {x.shape}")
    
    # Test 1: Forward pass sans masque
    print("\n[Test 1] Forward pass sans masque...")
    block.eval()  # Mode évaluation (désactive dropout)
    output = block(x, mask=None)
    print(f"  Output shape: {output.shape}")
    
    # Vérifications
    assert output.shape == x.shape, \
        f"Shape mismatch: expected {x.shape}, got {output.shape}"
    assert output.size(0) == batch_size, \
        f"Batch size mismatch: expected {batch_size}, got {output.size(0)}"
    assert output.size(1) == seq_len, \
        f"Sequence length mismatch: expected {seq_len}, got {output.size(1)}"
    assert output.size(2) == d_model, \
        f"Model dimension mismatch: expected {d_model}, got {output.size(2)}"
    
    print("  ✓ Dimension préservée!")
    
    # Test 2: Forward pass avec masque causal
    print("\n[Test 2] Forward pass avec masque causal...")
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    output_masked = block(x, mask=causal_mask)
    print(f"  Output shape: {output_masked.shape}")
    
    assert output_masked.shape == x.shape, \
        f"Shape mismatch with mask: expected {x.shape}, got {output_masked.shape}"
    
    print("  ✓ Fonctionne avec masque causal!")
    
    # Test 3: Vérifier que le dropout a un effet
    print("\n[Test 3] Vérification du dropout...")
    block.train()  # Mode entraînement (active dropout)
    
    output_train_1 = block(x, mask=None)
    output_train_2 = block(x, mask=None)
    
    # Avec dropout, les sorties devraient être différentes
    diff = torch.abs(output_train_1 - output_train_2).max().item()
    print(f"  Différence max entre deux forward passes: {diff:.6f}")
    
    if dropout > 0:
        assert diff > 1e-6, "Dropout devrait produire des sorties différentes"
        print("  ✓ Dropout fonctionne correctement!")
    else:
        print("  ℹ Dropout désactivé (dropout=0)")
    
    # Test 4: Vérifier que eval() désactive le dropout
    print("\n[Test 4] Vérification du mode eval()...")
    block.eval()
    
    output_eval_1 = block(x, mask=None)
    output_eval_2 = block(x, mask=None)
    
    diff_eval = torch.abs(output_eval_1 - output_eval_2).max().item()
    print(f"  Différence max en mode eval: {diff_eval:.10f}")
    
    assert diff_eval < 1e-6, "En mode eval, les sorties devraient être identiques"
    print("  ✓ Mode eval désactive le dropout!")
    
    print(f"\n{'='*70}")
    print("✓ Toutes les validations sont passées avec succès!")
    print(f"{'='*70}\n")
    
    return True



def example_transformer_block():
    """
    Exemple pédagogique montrant l'utilisation du TransformerBlock.
    """
    print("\n" + "="*70)
    print("EXEMPLE: TransformerBlock")
    print("="*70)
    
    # Configuration
    d_model = 128
    num_heads = 4
    d_ff = 512
    batch_size = 2
    seq_len = 8
    
    print(f"\nConfiguration:")
    print(f"  - d_model: {d_model}")
    print(f"  - num_heads: {num_heads}")
    print(f"  - d_ff: {d_ff}")
    print(f"  - batch_size: {batch_size}")
    print(f"  - seq_len: {seq_len}")
    
    # Créer le bloc
    block = TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=0.1)
    
    # Compter les paramètres
    num_params = sum(p.numel() for p in block.parameters())
    print(f"\nNombre de paramètres: {num_params:,}")
    
    # Détail des paramètres par composant
    print(f"\nDétail des paramètres:")
    for name, param in block.named_parameters():
        print(f"  {name}: {param.shape} ({param.numel():,} params)")
    
    # Créer un input
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass sans masque (BERT-style)
    print("\n" + "-"*70)
    print("Forward pass SANS masque (bidirectionnel - BERT)")
    print("-"*70)
    
    block.eval()
    output_no_mask = block(x, mask=None)
    
    print(f"Output shape: {output_no_mask.shape}")
    print(f"Dimension préservée: {x.shape == output_no_mask.shape}")
    
    # Forward pass avec masque causal (GPT-style)
    print("\n" + "-"*70)
    print("Forward pass AVEC masque causal (autorégressif - GPT)")
    print("-"*70)
    
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    print(f"\nMasque causal:")
    print(causal_mask)
    
    output_masked = block(x, mask=causal_mask)
    
    print(f"\nOutput shape: {output_masked.shape}")
    print(f"Dimension préservée: {x.shape == output_masked.shape}")
    
    # Comparer les sorties
    print("\n" + "-"*70)
    print("Comparaison des sorties")
    print("-"*70)
    
    diff = torch.abs(output_no_mask - output_masked).mean().item()
    print(f"Différence moyenne entre BERT et GPT: {diff:.6f}")
    print(f"Les sorties sont différentes car le masque change l'attention!")
    
    # Visualiser l'effet du masque sur une position
    print("\n" + "-"*70)
    print("Effet du masque sur la position 3")
    print("-"*70)
    
    print(f"\nSans masque (BERT): Le token 3 peut voir tous les tokens (0-7)")
    print(f"Avec masque (GPT): Le token 3 ne peut voir que les tokens 0-3")
    
    print(f"\nValeurs du token 3:")
    print(f"  Sans masque: {output_no_mask[0, 3, :5]}")  # Premiers 5 dims
    print(f"  Avec masque: {output_masked[0, 3, :5]}")
    
    print("\n" + "="*70)
    print("INTERPRÉTATION")
    print("="*70)
    print("""
    Le TransformerBlock est l'unité de base des transformers:
    
    1. Multi-Head Attention:
       - Permet aux tokens de s'attendre mutuellement
       - Chaque tête se spécialise dans différents patterns
    
    2. Residual Connections:
       - Permettent au gradient de circuler facilement
       - Essentielles pour entraîner des réseaux profonds
    
    3. Layer Normalization:
       - Stabilise l'entraînement
       - Accélère la convergence
    
    4. Feed-Forward Network:
       - Transformations non-linéaires position-wise
       - Ajoute de la capacité de computation
    
    5. Dropout:
       - Régularisation pour éviter le surapprentissage
       - Désactivé automatiquement en mode eval()
    
    Différence BERT vs GPT:
    - BERT: Pas de masque → attention bidirectionnelle
    - GPT: Masque causal → attention autoregressive
    
    Un transformer complet = Stack de plusieurs TransformerBlocks!
    """)
    
    return output_masked


def example_stacked_transformer_blocks():
    """
    Exemple montrant comment empiler plusieurs TransformerBlocks.
    """
    print("\n" + "="*70)
    print("EXEMPLE: Stack de TransformerBlocks")
    print("="*70)
    
    # Configuration
    d_model = 128
    num_heads = 4
    d_ff = 512
    num_layers = 3  # 3 blocs empilés
    batch_size = 2
    seq_len = 8
    
    print(f"\nConfiguration:")
    print(f"  - d_model: {d_model}")
    print(f"  - num_heads: {num_heads}")
    print(f"  - d_ff: {d_ff}")
    print(f"  - num_layers: {num_layers}")
    
    # Créer un stack de blocs avec nn.ModuleList
    blocks = nn.ModuleList([
        TransformerBlock(d_model, num_heads, d_ff, dropout=0.1)
        for _ in range(num_layers)
    ])
    
    # Compter les paramètres totaux
    total_params = sum(p.numel() for p in blocks.parameters())
    print(f"\nNombre total de paramètres: {total_params:,}")
    print(f"Paramètres par bloc: {total_params // num_layers:,}")
    
    # Créer un input
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\nInput shape: {x.shape}")
    
    # Créer un masque causal
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    
    # Passer à travers tous les blocs
    print("\n" + "-"*70)
    print("Forward pass à travers le stack")
    print("-"*70)
    
    for i, block in enumerate(blocks):
        block.eval()
        x = block(x, mask=causal_mask)
        print(f"Après bloc {i+1}: shape = {x.shape}")
    
    print(f"\nOutput final shape: {x.shape}")
    print(f"Dimension préservée à travers tous les blocs!")
    
    print("\n" + "="*70)
    print("INTERPRÉTATION")
    print("="*70)
    print(f"""
    Stack de {num_layers} TransformerBlocks:
    
    - Chaque bloc transforme l'input de manière progressive
    - La dimension est préservée à chaque étape (d_model={d_model})
    - Les blocs profonds capturent des patterns plus abstraits
    
    Architecture complète:
      Input → Block 1 → Block 2 → Block 3 → Output
    
    Exemples de modèles:
    - GPT-2 Small: 12 blocs, d_model=768
    - GPT-2 Medium: 24 blocs, d_model=1024
    - GPT-3: 96 blocs, d_model=12288
    
    Plus de blocs = Plus de capacité = Meilleure performance
    (mais aussi plus lent et plus de mémoire)
    """)
    
    return x


if __name__ == "__main__":
    """
    Script de test et démonstration.
    
    Ce script peut être exécuté directement pour:
    1. Valider que le TransformerBlock fonctionne correctement
    2. Voir des exemples d'utilisation
    3. Comprendre comment empiler plusieurs blocs
    """
    print("TransformerBlock - Tests et Démonstration")
    print("=" * 70)
    
    # Test 1: Validation
    print("\n[Test 1] Validation du TransformerBlock")
    validate_transformer_block(d_model=256, num_heads=8, d_ff=1024)
    
    # Test 2: Exemple simple
    print("\n[Test 2] Exemple d'utilisation")
    example_transformer_block()
    
    # Test 3: Stack de blocs
    print("\n[Test 3] Stack de TransformerBlocks")
    example_stacked_transformer_blocks()
    
    print("\n" + "=" * 70)
    print("✓ Tous les tests sont passés avec succès!")
    print("=" * 70)
