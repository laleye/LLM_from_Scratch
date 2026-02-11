"""
Feature Encoder Module — CNN 1D Multi-Couches (wav2vec 2.0)

Ce module implémente le Feature Encoder de wav2vec 2.0, qui transforme
l'audio brut (waveform) en représentations latentes via 7 couches de
convolution 1D avec réduction temporelle progressive.

Réduction temporelle totale : ~320x
    16 000 échantillons/seconde → ~50 vecteurs latents/seconde

Deux implémentations sont fournies :
1. From Scratch (NumPy) — pour comprendre la convolution 1D et le stride
2. PyTorch (nn.Module) — code professionnel avec 7 blocs Conv1D + LayerNorm + GELU

Architecture wav2vec 2.0 Feature Encoder :
    Couche 1 : Conv1D(1, 512, kernel=10, stride=5)
    Couche 2 : Conv1D(512, 512, kernel=3, stride=2)
    Couche 3 : Conv1D(512, 512, kernel=3, stride=2)
    Couche 4 : Conv1D(512, 512, kernel=3, stride=2)
    Couche 5 : Conv1D(512, 512, kernel=3, stride=2)
    Couche 6 : Conv1D(512, 512, kernel=2, stride=2)
    Couche 7 : Conv1D(512, 512, kernel=2, stride=2)
    Stride total : 5 × 2 × 2 × 2 × 2 × 2 × 2 = 320
"""

import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn


# ============================================
# IMPLEMENTATION 1: From Scratch (NumPy)
# ============================================
# Objectif: Comprendre la convolution 1D et le stride


def conv1d_from_scratch(
    x: np.ndarray,
    kernel: np.ndarray,
    stride: int = 1
) -> np.ndarray:
    """Convolution 1D manuelle avec boucle explicite.

    Montre comment le stride réduit la résolution temporelle.

    Args:
        x: Signal d'entrée, shape (channels_in, length)
        kernel: Noyau de convolution, shape (channels_out, channels_in, kernel_size)
        stride: Pas de la convolution (contrôle la réduction temporelle)

    Returns:
        Signal de sortie, shape (channels_out, output_length)
        où output_length = (length - kernel_size) // stride + 1
    """
    channels_in, length = x.shape
    channels_out, k_cin, kernel_size = kernel.shape

    assert k_cin == channels_in, (
        f"Kernel channels_in ({k_cin}) != input channels_in ({channels_in})"
    )

    # --- Calcul de la longueur de sortie ---
    output_length = (length - kernel_size) // stride + 1
    # print(f"  [Conv1D] input=({channels_in}, {length}), kernel=({channels_out}, {k_cin}, {kernel_size}), "
    #       f"stride={stride} → output=({channels_out}, {output_length})")

    output = np.zeros((channels_out, output_length))

    # --- Boucle explicite sur chaque position de sortie ---
    for t in range(output_length):
        # Position de départ dans le signal d'entrée
        start = t * stride
        # Extraire le segment d'entrée : shape (channels_in, kernel_size)
        segment = x[:, start:start + kernel_size]

        # Pour chaque canal de sortie, calculer le produit scalaire
        for c_out in range(channels_out):
            # kernel[c_out] shape: (channels_in, kernel_size)
            # segment shape: (channels_in, kernel_size)
            # Produit élément par élément puis somme = produit scalaire
            output[c_out, t] = np.sum(kernel[c_out] * segment)

    return output


def _layer_norm_from_scratch(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Layer normalization from scratch sur la dimension des canaux.

    Args:
        x: shape (channels, length)
        eps: Epsilon pour la stabilité numérique

    Returns:
        Normalized x, même shape
    """
    # Normaliser sur la dimension des canaux (axis=0)
    mean = x.mean(axis=0, keepdims=True)
    var = x.var(axis=0, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def _gelu_from_scratch(x: np.ndarray) -> np.ndarray:
    """Activation GELU from scratch.

    GELU(x) = x * Φ(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    """
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def feature_encoder_from_scratch(
    waveform: np.ndarray,
    kernels: List[np.ndarray] = None,
    strides: List[int] = None
) -> np.ndarray:
    """Pipeline complet d'extraction de features from scratch.

    Applique séquentiellement : Conv1D → LayerNorm → GELU pour chaque couche.
    Affiche les shapes intermédiaires.

    Args:
        waveform: Signal audio 1D, shape (num_samples,)
        kernels: Liste de noyaux de convolution. Si None, utilise des noyaux aléatoires
                 avec l'architecture wav2vec 2.0 (7 couches).
        strides: Liste des strides. Si None, utilise [5, 2, 2, 2, 2, 2, 2].

    Returns:
        Représentations latentes, shape (d_model, T)
        où T ≈ num_samples / 320
    """
    if strides is None:
        strides = [5, 2, 2, 2, 2, 2, 2]

    # Architecture wav2vec 2.0 : kernel sizes et channels
    kernel_sizes = [10, 3, 3, 3, 3, 2, 2]
    channels = [1, 512, 512, 512, 512, 512, 512]  # input channels per layer
    d_model = 512

    if kernels is None:
        # Initialiser des noyaux aléatoires (pour la démonstration)
        rng = np.random.default_rng(42)
        kernels = []
        for i in range(7):
            c_in = channels[i]
            c_out = d_model
            k_size = kernel_sizes[i]
            # Initialisation Xavier
            scale = np.sqrt(2.0 / (c_in * k_size + c_out))
            kernels.append(rng.normal(0, scale, (c_out, c_in, k_size)))

    # Reshape waveform : (num_samples,) → (1, num_samples) pour 1 canal d'entrée
    x = waveform.reshape(1, -1)
    print(f"  [Feature Encoder] Input shape: {x.shape}")

    for i, (kernel, stride) in enumerate(zip(kernels, strides)):
        # Conv1D
        x = conv1d_from_scratch(x, kernel, stride=stride)
        # LayerNorm
        x = _layer_norm_from_scratch(x)
        # GELU
        x = _gelu_from_scratch(x)
        print(f"  [Feature Encoder] Couche {i+1}: shape={x.shape}, stride={stride}")

    print(f"  [Feature Encoder] Output shape: {x.shape}")
    return x


# ============================================
# IMPLEMENTATION 2: PyTorch (nn.Module)
# ============================================
# Objectif: Code professionnel et efficace


class FeatureEncoder(nn.Module):
    """CNN 1D multi-couches inspiré de wav2vec 2.0.

    Architecture : 7 blocs Conv1D + LayerNorm + GELU
    Entrée : waveform brute (batch, 1, samples)
    Sortie : représentations latentes (batch, d_model, T)

    La réduction temporelle totale est le produit de tous les strides :
        5 × 2 × 2 × 2 × 2 × 2 × 2 = 320

    Méthodes PyTorch expliquées :
    - nn.Conv1d() : Convolution 1D avec kernel_size et stride
    - nn.LayerNorm() : Normalisation par couche
    - nn.GELU() : Activation Gaussian Error Linear Unit
    - nn.ModuleList() : Liste de modules enregistrés pour autograd

    Args:
        d_model: Dimension du modèle (nombre de canaux de sortie). Default: 512.
    """

    # Architecture wav2vec 2.0 : (kernel_size, stride)
    LAYER_CONFIGS = [
        (10, 5),   # Couche 1 : grande fenêtre initiale
        (3, 2),    # Couche 2
        (3, 2),    # Couche 3
        (3, 2),    # Couche 4
        (3, 2),    # Couche 5
        (2, 2),    # Couche 6
        (2, 2),    # Couche 7
    ]

    def __init__(self, d_model: int = 512):
        super().__init__()
        self.d_model = d_model

        # Calculer le stride total pour la documentation
        self.total_stride = 1
        for _, s in self.LAYER_CONFIGS:
            self.total_stride *= s
        # total_stride = 5 * 2^6 = 320

        # Construire les 7 couches
        layers = []
        in_channels = 1  # waveform mono
        for i, (kernel_size, stride) in enumerate(self.LAYER_CONFIGS):
            out_channels = d_model
            layers.append(
                nn.Sequential(
                    # nn.Conv1d : Convolution 1D
                    #   in_channels → out_channels, kernel_size, stride
                    #   Réduit la dimension temporelle par un facteur stride
                    nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, bias=False),
                    # nn.GELU : Activation douce (utilisée dans wav2vec 2.0)
                    nn.GELU(),
                )
            )
            in_channels = out_channels

        self.layers = nn.ModuleList(layers)

        # Layer norm appliquée sur la dimension des canaux après toutes les convolutions
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass du Feature Encoder.

        Args:
            x: Waveform brute, shape (batch, 1, num_samples)

        Returns:
            Représentations latentes, shape (batch, d_model, T)
            où T ≈ num_samples / total_stride

        Raises:
            ValueError: Si la shape d'entrée est incorrecte.
        """
        if x.dim() != 3 or x.shape[1] != 1:
            raise ValueError(
                f"Expected input shape (batch, 1, num_samples), got {x.shape}"
            )

        # Appliquer les 7 couches Conv1D + GELU
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # print(f"  [FeatureEncoder] Couche {i+1}: {x.shape}")

        # Layer norm sur la dimension des canaux
        # x shape: (batch, d_model, T) → transpose → (batch, T, d_model) → norm → transpose back
        x = x.transpose(1, 2)          # (batch, T, d_model)
        x = self.layer_norm(x)         # normalise sur d_model
        x = x.transpose(1, 2)          # (batch, d_model, T)

        return x

    def compute_output_length(self, input_length: int) -> int:
        """Calcule la longueur de sortie pour une longueur d'entrée donnée.

        Applique la formule de réduction pour chaque couche :
            output_length = (input_length - kernel_size) // stride + 1

        Args:
            input_length: Nombre d'échantillons en entrée.

        Returns:
            Nombre de vecteurs latents en sortie (T).
        """
        length = input_length
        for kernel_size, stride in self.LAYER_CONFIGS:
            length = (length - kernel_size) // stride + 1
        return length
