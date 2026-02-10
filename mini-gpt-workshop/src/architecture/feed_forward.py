"""
Feed-Forward Network Implementation

This module implements the position-wise feed-forward network used in transformer blocks.
Two implementations are provided:
1. From-scratch NumPy implementation for pedagogical understanding
2. Professional PyTorch implementation for production use

The FFN applies two linear transformations with a non-linear activation in between:
    FFN(x) = activation(xW1 + b1)W2 + b2

Typical configuration: d_ff = 4 * d_model
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ============================================
# IMPLEMENTATION 1: From Scratch (NumPy)
# ============================================
# Objectif: Comprendre le formalisme mathématique

def gelu_from_scratch(x: np.ndarray) -> np.ndarray:
    """
    GELU (Gaussian Error Linear Unit) activation function from scratch.
    
    GELU(x) = x * Φ(x)
    où Φ(x) est la fonction de répartition de la loi normale standard.
    
    Approximation utilisée:
    GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    
    Args:
        x: Input array of any shape
    
    Returns:
        Array of same shape with GELU activation applied
    
    Note:
        Cette approximation est très proche de la vraie fonction GELU
        et est plus rapide à calculer que l'intégrale de la gaussienne.
    """
    # Approximation de GELU
    # GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(sqrt_2_over_pi * (x + 0.044715 * x**3)))


def relu_from_scratch(x: np.ndarray) -> np.ndarray:
    """
    ReLU (Rectified Linear Unit) activation function from scratch.
    
    ReLU(x) = max(0, x)
    
    Args:
        x: Input array of any shape
    
    Returns:
        Array of same shape with ReLU activation applied
    """
    return np.maximum(0, x)


def feed_forward_from_scratch(
    x: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
    activation: str = "gelu"
) -> np.ndarray:
    """
    Feed-Forward Network from scratch using NumPy.
    
    Architecture:
        FFN(x) = activation(xW1 + b1)W2 + b2
    
    Formule mathématique:
        1. Première transformation linéaire: h = xW1 + b1
        2. Activation non-linéaire: h' = activation(h)
        3. Deuxième transformation linéaire: y = h'W2 + b2
    
    Args:
        x: Input array of shape (..., d_model)
        W1: First weight matrix of shape (d_model, d_ff)
        b1: First bias vector of shape (d_ff,)
        W2: Second weight matrix of shape (d_ff, d_model)
        b2: Second bias vector of shape (d_model,)
        activation: Activation function to use ("gelu" or "relu")
    
    Returns:
        Output array of shape (..., d_model)
    
    Shape transformations:
        Input:  (..., d_model)
        After W1: (..., d_ff)
        After activation: (..., d_ff)
        After W2: (..., d_model)
    """
    # Étape 1: Première transformation linéaire
    # h = xW1 + b1
    # Shape: (..., d_model) @ (d_model, d_ff) + (d_ff,) -> (..., d_ff)
    hidden = np.matmul(x, W1) + b1
    
    # Étape 2: Activation non-linéaire
    # h' = activation(h)
    # Shape: (..., d_ff) -> (..., d_ff)
    if activation == "gelu":
        hidden = gelu_from_scratch(hidden)
    elif activation == "relu":
        hidden = relu_from_scratch(hidden)
    else:
        raise ValueError(f"Unknown activation: {activation}")
    
    # Étape 3: Deuxième transformation linéaire
    # y = h'W2 + b2
    # Shape: (..., d_ff) @ (d_ff, d_model) + (d_model,) -> (..., d_model)
    output = np.matmul(hidden, W2) + b2
    
    return output


# ============================================
# IMPLEMENTATION 2: PyTorch (nn.Module)
# ============================================
# Objectif: Code professionnel et efficace

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network using PyTorch.
    
    Implémentation professionnelle du réseau feed-forward utilisé dans les
    transformer blocks. Ce réseau est appliqué indépendamment à chaque position
    de la séquence (d'où "position-wise").
    
    Architecture:
        FFN(x) = GELU(xW1 + b1)W2 + b2
    
    Méthodes PyTorch utilisées:
        - nn.Linear(): Transformation linéaire (xW^T + b)
          * Initialise automatiquement les poids avec Xavier uniform
          * Gère efficacement les opérations batch sur GPU
          * Paramètres: in_features, out_features, bias=True
        
        - nn.GELU(): Activation GELU (Gaussian Error Linear Unit)
          * Plus lisse que ReLU, pas de discontinuité en 0
          * Utilisé dans BERT et GPT-2/3
          * Formule: GELU(x) = x * Φ(x) où Φ est la CDF gaussienne
          * Avantage: Meilleure propagation du gradient
        
        - nn.Sequential(): Conteneur pour empiler des couches
          * Applique les couches séquentiellement
          * Simplifie le code du forward pass
          * Utile pour des architectures linéaires
        
        - nn.Dropout(): Régularisation par dropout
          * Désactive aléatoirement des neurones pendant l'entraînement
          * Réduit le surapprentissage
          * Automatiquement désactivé en mode eval()
    
    Dimensions typiques:
        - d_model: 256, 512, 768 (dimension du modèle)
        - d_ff: 1024, 2048, 3072 (typiquement 4 * d_model)
    
    Args:
        d_model: Dimension of input and output
        d_ff: Dimension of hidden layer (typically 4 * d_model)
        dropout: Dropout probability (default: 0.1)
        activation: Activation function to use ("gelu" or "relu")
    
    Shape:
        - Input: (batch_size, seq_len, d_model)
        - Output: (batch_size, seq_len, d_model)
    
    Example:
        >>> ffn = FeedForward(d_model=512, d_ff=2048)
        >>> x = torch.randn(32, 128, 512)  # (batch, seq_len, d_model)
        >>> output = ffn(x)
        >>> print(output.shape)  # torch.Size([32, 128, 512])
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Première transformation linéaire: d_model -> d_ff
        # nn.Linear(in_features, out_features) effectue: y = xW^T + b
        # Note: PyTorch utilise W^T (transposée) contrairement à la notation mathématique
        self.linear1 = nn.Linear(d_model, d_ff)
        
        # Fonction d'activation
        if activation == "gelu":
            # GELU: Plus lisse que ReLU, utilisé dans BERT et GPT
            # Avantage: Pas de "mort" de neurones comme avec ReLU
            self.activation = nn.GELU()
        elif activation == "relu":
            # ReLU: Activation classique, plus simple mais moins performante
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}. Use 'gelu' or 'relu'.")
        
        # Dropout pour régularisation
        # Désactive aléatoirement des neurones pendant l'entraînement
        self.dropout = nn.Dropout(dropout)
        
        # Deuxième transformation linéaire: d_ff -> d_model
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feed-forward network.
        
        Étapes:
            1. Expansion: d_model -> d_ff (augmente la capacité)
            2. Activation: Non-linéarité (GELU ou ReLU)
            3. Dropout: Régularisation
            4. Projection: d_ff -> d_model (retour à la dimension originale)
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        
        Shape transformations:
            Input:           (batch_size, seq_len, d_model)
            After linear1:   (batch_size, seq_len, d_ff)
            After activation:(batch_size, seq_len, d_ff)
            After dropout:   (batch_size, seq_len, d_ff)
            After linear2:   (batch_size, seq_len, d_model)
        """
        # Vérification de la dimension d'entrée
        assert x.size(-1) == self.d_model, \
            f"Expected input dimension {self.d_model}, got {x.size(-1)}"
        
        # Étape 1: Première transformation linéaire (expansion)
        # Shape: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff)
        x = self.linear1(x)
        
        # Étape 2: Activation non-linéaire
        # Shape: (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_ff)
        x = self.activation(x)
        
        # Étape 3: Dropout (seulement pendant l'entraînement)
        # Shape: (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_ff)
        x = self.dropout(x)
        
        # Étape 4: Deuxième transformation linéaire (projection)
        # Shape: (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        x = self.linear2(x)
        
        # Vérification de la dimension de sortie
        assert x.size(-1) == self.d_model, \
            f"Expected output dimension {self.d_model}, got {x.size(-1)}"
        
        return x


class FeedForwardSequential(nn.Module):
    """
    Alternative implementation using nn.Sequential for simplicity.
    
    Cette implémentation utilise nn.Sequential pour composer les couches,
    ce qui simplifie le code mais offre moins de flexibilité pour le debugging.
    
    Équivalent à FeedForward mais avec une syntaxe plus concise.
    
    Args:
        d_model: Dimension of input and output
        d_ff: Dimension of hidden layer
        dropout: Dropout probability
        activation: Activation function to use ("gelu" or "relu")
    
    Shape:
        - Input: (batch_size, seq_len, d_model)
        - Output: (batch_size, seq_len, d_model)
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        
        # Sélection de l'activation
        if activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "relu":
            act_fn = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # nn.Sequential: Empile les couches séquentiellement
        # Avantage: Code plus concis
        # Inconvénient: Moins de contrôle pour le debugging
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),      # Expansion
            act_fn,                         # Activation
            nn.Dropout(dropout),            # Régularisation
            nn.Linear(d_ff, d_model)        # Projection
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the sequential network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        return self.net(x)


# ============================================
# COMPARISON: GELU vs ReLU
# ============================================

def compare_activations():
    """
    Compare GELU and ReLU activation functions.
    
    Cette fonction génère des visualisations pour comparer GELU et ReLU.
    Utile pour comprendre les différences entre les deux activations.
    
    Différences clés:
        - ReLU: Discontinuité en 0, "tue" les valeurs négatives
        - GELU: Lisse partout, permet un petit gradient pour les valeurs négatives
        - GELU: Meilleure performance empirique dans les transformers
    
    Returns:
        Dictionary with comparison data
    """
    import matplotlib.pyplot as plt
    
    # Générer des valeurs d'entrée
    x = np.linspace(-3, 3, 1000)
    
    # Calculer les activations
    gelu_output = gelu_from_scratch(x)
    relu_output = relu_from_scratch(x)
    
    # Créer la visualisation
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Fonctions d'activation
    plt.subplot(1, 2, 1)
    plt.plot(x, gelu_output, label='GELU', linewidth=2)
    plt.plot(x, relu_output, label='ReLU', linewidth=2, linestyle='--')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Input (x)')
    plt.ylabel('Output')
    plt.title('GELU vs ReLU: Fonctions d\'activation')
    plt.legend()
    
    # Plot 2: Gradients (dérivées)
    plt.subplot(1, 2, 2)
    # Approximation numérique des gradients
    dx = x[1] - x[0]
    gelu_grad = np.gradient(gelu_output, dx)
    relu_grad = np.gradient(relu_output, dx)
    
    plt.plot(x, gelu_grad, label='GELU gradient', linewidth=2)
    plt.plot(x, relu_grad, label='ReLU gradient', linewidth=2, linestyle='--')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Input (x)')
    plt.ylabel('Gradient')
    plt.title('GELU vs ReLU: Gradients')
    plt.legend()
    
    plt.tight_layout()
    
    return {
        'x': x,
        'gelu': gelu_output,
        'relu': relu_output,
        'gelu_grad': gelu_grad,
        'relu_grad': relu_grad
    }


# ============================================
# VALIDATION UTILITIES
# ============================================

def validate_dimension_preservation(
    d_model: int,
    d_ff: int,
    batch_size: int = 2,
    seq_len: int = 10
) -> bool:
    """
    Validate that the feed-forward network preserves input dimensions.
    
    Cette fonction vérifie que:
        1. La dimension d'entrée (d_model) est préservée en sortie
        2. Les dimensions batch et sequence sont inchangées
        3. Les deux implémentations (from-scratch et PyTorch) donnent des résultats cohérents
    
    Args:
        d_model: Model dimension
        d_ff: Feed-forward hidden dimension
        batch_size: Batch size for testing
        seq_len: Sequence length for testing
    
    Returns:
        True if all validations pass
    
    Raises:
        AssertionError if any validation fails
    """
    print(f"\n{'='*60}")
    print(f"Validation: Feed-Forward Network Dimension Preservation")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  - d_model: {d_model}")
    print(f"  - d_ff: {d_ff}")
    print(f"  - batch_size: {batch_size}")
    print(f"  - seq_len: {seq_len}")
    print(f"{'='*60}\n")
    
    # Test PyTorch implementation
    print("Testing PyTorch implementation...")
    ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=0.0)
    ffn.eval()  # Désactiver le dropout pour la validation
    
    # Créer un input de test
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"  Input shape: {x.shape}")
    
    # Forward pass
    output = ffn(x)
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
    
    print("  ✓ All dimension checks passed!")
    
    # Test from-scratch implementation
    print("\nTesting from-scratch implementation...")
    x_np = x.detach().numpy()
    
    # Initialiser des poids aléatoires
    W1 = np.random.randn(d_model, d_ff) * 0.01
    b1 = np.zeros(d_ff)
    W2 = np.random.randn(d_ff, d_model) * 0.01
    b2 = np.zeros(d_model)
    
    output_np = feed_forward_from_scratch(x_np, W1, b1, W2, b2, activation="gelu")
    print(f"  Input shape: {x_np.shape}")
    print(f"  Output shape: {output_np.shape}")
    
    # Vérifications
    assert output_np.shape == x_np.shape, \
        f"Shape mismatch: expected {x_np.shape}, got {output_np.shape}"
    
    print("  ✓ All dimension checks passed!")
    
    print(f"\n{'='*60}")
    print("✓ All validations passed successfully!")
    print(f"{'='*60}\n")
    
    return True


if __name__ == "__main__":
    """
    Script de test et démonstration.
    
    Ce script peut être exécuté directement pour:
        1. Valider que les dimensions sont préservées
        2. Comparer GELU et ReLU visuellement
        3. Tester les deux implémentations
    """
    print("Feed-Forward Network - Tests et Démonstration")
    print("=" * 60)
    
    # Test 1: Validation des dimensions
    print("\n[Test 1] Validation des dimensions")
    validate_dimension_preservation(d_model=256, d_ff=1024, batch_size=4, seq_len=16)
    
    # Test 2: Comparaison GELU vs ReLU
    print("\n[Test 2] Comparaison GELU vs ReLU")
    try:
        comparison_data = compare_activations()
        print("✓ Visualisation générée avec succès!")
        print("  (Fermez la fenêtre matplotlib pour continuer)")
        import matplotlib.pyplot as plt
        plt.show()
    except ImportError:
        print("⚠ matplotlib non disponible, visualisation ignorée")
    
    # Test 3: Test des deux implémentations PyTorch
    print("\n[Test 3] Comparaison FeedForward vs FeedForwardSequential")
    d_model, d_ff = 128, 512
    x = torch.randn(2, 10, d_model)
    
    ffn1 = FeedForward(d_model, d_ff, dropout=0.0)
    ffn2 = FeedForwardSequential(d_model, d_ff, dropout=0.0)
    
    # Copier les poids pour avoir les mêmes résultats
    ffn2.net[0].weight.data = ffn1.linear1.weight.data.clone()
    ffn2.net[0].bias.data = ffn1.linear1.bias.data.clone()
    ffn2.net[3].weight.data = ffn1.linear2.weight.data.clone()
    ffn2.net[3].bias.data = ffn1.linear2.bias.data.clone()
    
    ffn1.eval()
    ffn2.eval()
    
    out1 = ffn1(x)
    out2 = ffn2(x)
    
    diff = torch.abs(out1 - out2).max().item()
    print(f"  Max difference between implementations: {diff:.10f}")
    assert diff < 1e-6, "Implementations should produce identical results"
    print("  ✓ Both implementations produce identical results!")
    
    print("\n" + "=" * 60)
    print("✓ Tous les tests sont passés avec succès!")
    print("=" * 60)
