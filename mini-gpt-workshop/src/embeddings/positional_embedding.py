"""
Positional Embedding Implementation

This module provides both from-scratch (NumPy) and PyTorch implementations
of positional embeddings for the Mini-GPT pedagogical notebook.

Positional embeddings encode the position of each token in the sequence,
allowing the model to understand word order since self-attention is
position-agnostic.
"""

import numpy as np
import torch
import torch.nn as nn
import math


# ============================================
# IMPLEMENTATION 1: From Scratch (NumPy)
# ============================================
# Objectif: Comprendre le formalisme mathématique de l'encodage positionnel

def create_sinusoidal_positional_encoding(max_seq_len: int, d_model: int) -> np.ndarray:
    """
    Crée une matrice d'encodage positionnel sinusoïdal.
    
    Formule mathématique (Vaswani et al., 2017 "Attention is All You Need"):
    
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    où:
    - pos est la position dans la séquence (0, 1, 2, ..., max_seq_len-1)
    - i est l'indice de la dimension (0, 1, 2, ..., d_model/2-1)
    - 2i correspond aux dimensions paires (0, 2, 4, ...)
    - 2i+1 correspond aux dimensions impaires (1, 3, 5, ...)
    
    Intuition:
    - Les fréquences varient de 1 (dimension 0) à 1/10000 (dernière dimension)
    - Permet au modèle de distinguer facilement les positions proches
    - Les positions éloignées ont des patterns similaires mais décalés
    
    Args:
        max_seq_len: Longueur maximale de séquence supportée
        d_model: Dimension de l'embedding (doit être pair)
    
    Returns:
        positional_encoding: Matrice d'encodage, shape (max_seq_len, d_model)
    
    Validation des dimensions:
        - Input: max_seq_len (int), d_model (int)
        - Output: (max_seq_len, d_model)
    """
    # Validation des dimensions
    assert max_seq_len > 0, f"max_seq_len doit être positif, reçu: {max_seq_len}"
    assert d_model > 0, f"d_model doit être positif, reçu: {d_model}"
    assert d_model % 2 == 0, f"d_model doit être pair pour l'encodage sinusoïdal, reçu: {d_model}"
    
    # Initialisation de la matrice d'encodage positionnel
    positional_encoding = np.zeros((max_seq_len, d_model))
    
    # Création du vecteur de positions: [0, 1, 2, ..., max_seq_len-1]
    # Shape: (max_seq_len, 1) pour le broadcasting
    position = np.arange(max_seq_len).reshape(-1, 1)
    
    # Calcul des diviseurs pour les différentes dimensions
    # div_term = 10000^(2i/d_model) pour i = 0, 1, 2, ..., d_model/2-1
    # 
    # Formule équivalente (plus stable numériquement):
    # div_term = exp(2i * -log(10000) / d_model)
    #          = exp(-2i * log(10000) / d_model)
    #
    # Shape: (d_model/2,)
    i = np.arange(0, d_model, 2)  # [0, 2, 4, ..., d_model-2]
    div_term = np.exp(i * -(np.log(10000.0) / d_model))
    
    # Application des fonctions sinusoïdales
    # Dimensions paires (0, 2, 4, ...): sin
    positional_encoding[:, 0::2] = np.sin(position * div_term)
    
    # Dimensions impaires (1, 3, 5, ...): cos
    positional_encoding[:, 1::2] = np.cos(position * div_term)
    
    # Vérification de la dimension de sortie
    assert positional_encoding.shape == (max_seq_len, d_model), \
        f"Shape incorrecte: attendu ({max_seq_len}, {d_model}), obtenu {positional_encoding.shape}"
    
    return positional_encoding


def get_positional_encoding(positions: np.ndarray, 
                           positional_encoding_matrix: np.ndarray) -> np.ndarray:
    """
    Récupère les encodages positionnels pour des positions données.
    
    Formule mathématique:
    - Pour une position p, on récupère la ligne PE[p] de la matrice d'encodage
    - Pour une séquence de positions [p₁, p₂, ..., pₙ], on obtient [PE[p₁], PE[p₂], ..., PE[pₙ]]
    
    Args:
        positions: Indices de positions, shape (batch_size, seq_len) ou (seq_len,)
        positional_encoding_matrix: Matrice d'encodage, shape (max_seq_len, d_model)
    
    Returns:
        encodings: Encodages positionnels, shape (*positions.shape, d_model)
    
    Validation des dimensions:
        - Input positions: (batch_size, seq_len) ou (seq_len,)
        - Input positional_encoding_matrix: (max_seq_len, d_model)
        - Output: (*positions.shape, d_model)
    
    Exemple:
        >>> pe_matrix = create_sinusoidal_positional_encoding(max_seq_len=100, d_model=64)
        >>> positions = np.array([[0, 1, 2], [3, 4, 5]])  # (2, 3)
        >>> encodings = get_positional_encoding(positions, pe_matrix)
        >>> encodings.shape
        (2, 3, 64)
    """
    max_seq_len, d_model = positional_encoding_matrix.shape
    
    # Validation: vérifier que toutes les positions sont valides
    assert np.all(positions >= 0), "Les positions doivent être non-négatives"
    assert np.all(positions < max_seq_len), \
        f"Toutes les positions doivent être < max_seq_len ({max_seq_len}), max trouvé: {np.max(positions)}"
    
    # Lookup: récupération des encodages positionnels
    encodings = positional_encoding_matrix[positions]
    
    # Vérification de la dimension de sortie
    expected_shape = (*positions.shape, d_model)
    assert encodings.shape == expected_shape, \
        f"Shape incorrecte: attendu {expected_shape}, obtenu {encodings.shape}"
    
    return encodings


def create_position_ids(batch_size: int, seq_len: int) -> np.ndarray:
    """
    Crée des IDs de positions pour un batch de séquences.
    
    Pour chaque séquence dans le batch, crée [0, 1, 2, ..., seq_len-1].
    
    Args:
        batch_size: Nombre de séquences dans le batch
        seq_len: Longueur de chaque séquence
    
    Returns:
        position_ids: Matrice de positions, shape (batch_size, seq_len)
    
    Exemple:
        >>> create_position_ids(batch_size=2, seq_len=4)
        array([[0, 1, 2, 3],
               [0, 1, 2, 3]])
    """
    # Création d'une séquence de positions [0, 1, 2, ..., seq_len-1]
    positions = np.arange(seq_len)
    
    # Répétition pour chaque élément du batch
    # Shape: (batch_size, seq_len)
    position_ids = np.tile(positions, (batch_size, 1))
    
    assert position_ids.shape == (batch_size, seq_len), \
        f"Shape incorrecte: attendu ({batch_size}, {seq_len}), obtenu {position_ids.shape}"
    
    return position_ids


# ============================================
# Fonctions utilitaires pour la pédagogie
# ============================================

def print_positional_encoding_info(positional_encoding: np.ndarray, 
                                   positions: np.ndarray = None):
    """
    Affiche des informations sur l'encodage positionnel pour la validation pédagogique.
    
    Args:
        positional_encoding: Matrice d'encodage positionnel
        positions: (Optionnel) Positions pour afficher leurs encodages
    """
    max_seq_len, d_model = positional_encoding.shape
    
    print("=" * 60)
    print("INFORMATIONS SUR L'ENCODAGE POSITIONNEL")
    print("=" * 60)
    print(f"Longueur maximale de séquence (max_seq_len): {max_seq_len}")
    print(f"Dimension de l'encodage (d_model): {d_model}")
    print(f"Shape de la matrice d'encodage: {positional_encoding.shape}")
    print(f"Type d'encodage: Sinusoïdal (sin/cos)")
    print(f"\nStatistiques de la matrice:")
    print(f"  - Moyenne: {np.mean(positional_encoding):.6f}")
    print(f"  - Écart-type: {np.std(positional_encoding):.6f}")
    print(f"  - Min: {np.min(positional_encoding):.6f}")
    print(f"  - Max: {np.max(positional_encoding):.6f}")
    
    # Vérification des propriétés sinusoïdales
    print(f"\nPropriétés sinusoïdales:")
    print(f"  - Dimensions paires (0, 2, 4, ...): sin")
    print(f"  - Dimensions impaires (1, 3, 5, ...): cos")
    print(f"  - Valeurs dans [-1, 1]: {np.all(np.abs(positional_encoding) <= 1.0)}")
    
    if positions is not None:
        print(f"\nExemple d'encodage pour positions: {positions}")
        encodings = get_positional_encoding(positions, positional_encoding)
        print(f"Shape des encodages résultants: {encodings.shape}")
        if encodings.ndim == 2:
            print(f"Premier encodage (position {positions[0]}):")
            print(f"  {encodings[0][:5]}... (premiers 5 éléments)")
    
    print("=" * 60)


def visualize_positional_encoding_pattern(positional_encoding: np.ndarray, 
                                         num_positions: int = 50):
    """
    Affiche un aperçu textuel des patterns d'encodage positionnel.
    (Pour une vraie visualisation, utiliser matplotlib dans un notebook)
    
    Args:
        positional_encoding: Matrice d'encodage positionnel
        num_positions: Nombre de positions à afficher
    """
    max_seq_len, d_model = positional_encoding.shape
    num_positions = min(num_positions, max_seq_len)
    
    print("\n" + "=" * 60)
    print("PATTERN D'ENCODAGE POSITIONNEL")
    print("=" * 60)
    print(f"Affichage des {num_positions} premières positions")
    print(f"Dimensions affichées: 0-7 (sur {d_model} total)")
    print()
    
    # Affichage des premières dimensions pour quelques positions
    print("Position | Dim 0 (sin) | Dim 1 (cos) | Dim 2 (sin) | Dim 3 (cos) | ...")
    print("-" * 60)
    
    for pos in range(0, num_positions, max(1, num_positions // 10)):
        values = positional_encoding[pos, :8]
        print(f"{pos:8d} | ", end="")
        for val in values[:4]:
            print(f"{val:11.6f} | ", end="")
        print("...")
    
    print("=" * 60)
    
    # Explication des fréquences
    print("\nExplication des fréquences:")
    print("  - Dimensions basses (0, 1): Haute fréquence → distingue positions proches")
    print("  - Dimensions hautes: Basse fréquence → patterns sur longues distances")
    print("  - Combinaison: Le modèle peut apprendre à utiliser différentes échelles")
    print("=" * 60 + "\n")


def validate_positional_encoding_dimensions(max_seq_len: int, d_model: int,
                                           batch_size: int, seq_len: int):
    """
    Valide les dimensions à travers le processus d'encodage positionnel complet.
    Fonction pédagogique pour vérifier la compréhension des shapes.
    
    Args:
        max_seq_len: Longueur maximale de séquence
        d_model: Dimension de l'encodage
        batch_size: Taille du batch
        seq_len: Longueur de la séquence (doit être <= max_seq_len)
    """
    assert seq_len <= max_seq_len, \
        f"seq_len ({seq_len}) doit être <= max_seq_len ({max_seq_len})"
    
    print("\n" + "=" * 60)
    print("VALIDATION DES DIMENSIONS - POSITIONAL ENCODING")
    print("=" * 60)
    
    # Étape 1: Création de la matrice d'encodage positionnel
    print(f"\n1. Création de la matrice d'encodage positionnel:")
    print(f"   max_seq_len={max_seq_len}, d_model={d_model}")
    pe_matrix = create_sinusoidal_positional_encoding(max_seq_len, d_model)
    print(f"   ✓ Shape de la matrice: {pe_matrix.shape}")
    
    # Étape 2: Création des IDs de positions
    print(f"\n2. Création des IDs de positions:")
    print(f"   batch_size={batch_size}, seq_len={seq_len}")
    position_ids = create_position_ids(batch_size, seq_len)
    print(f"   ✓ Shape des position_ids: {position_ids.shape}")
    print(f"   ✓ Exemple de positions: {position_ids[0]}")
    
    # Étape 3: Récupération des encodages positionnels
    print(f"\n3. Récupération des encodages positionnels:")
    encodings = get_positional_encoding(position_ids, pe_matrix)
    print(f"   ✓ Shape des encodages: {encodings.shape}")
    
    # Étape 4: Vérification finale
    expected_shape = (batch_size, seq_len, d_model)
    print(f"\n4. Vérification finale:")
    print(f"   Shape attendue: {expected_shape}")
    print(f"   Shape obtenue: {encodings.shape}")
    
    if encodings.shape == expected_shape:
        print(f"   ✓ SUCCÈS: Les dimensions sont correctes!")
    else:
        print(f"   ✗ ERREUR: Les dimensions ne correspondent pas!")
    
    print("=" * 60 + "\n")
    
    return encodings


def combine_token_and_positional_embeddings(token_embeddings: np.ndarray,
                                           positional_encodings: np.ndarray) -> np.ndarray:
    """
    Combine les embeddings de tokens et les encodages positionnels.
    
    Formule mathématique:
    E_input = E_token + E_position
    
    où:
    - E_token: Embeddings des tokens (contenu sémantique)
    - E_position: Encodages positionnels (information de position)
    - E_input: Embeddings finaux combinés
    
    Args:
        token_embeddings: Embeddings de tokens, shape (batch_size, seq_len, d_model)
        positional_encodings: Encodages positionnels, shape (batch_size, seq_len, d_model)
    
    Returns:
        combined_embeddings: Embeddings combinés, shape (batch_size, seq_len, d_model)
    
    Validation des dimensions:
        - Les deux inputs doivent avoir exactement la même shape
        - Output a la même shape que les inputs
    """
    # Validation: les shapes doivent correspondre
    assert token_embeddings.shape == positional_encodings.shape, \
        f"Les shapes doivent correspondre: token_embeddings {token_embeddings.shape} " \
        f"vs positional_encodings {positional_encodings.shape}"
    
    # Combinaison par addition élément par élément
    combined_embeddings = token_embeddings + positional_encodings
    
    # Vérification de la dimension de sortie
    assert combined_embeddings.shape == token_embeddings.shape, \
        f"Shape incorrecte après combinaison: {combined_embeddings.shape}"
    
    return combined_embeddings


# ============================================
# Exemple d'utilisation pédagogique
# ============================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DÉMONSTRATION: POSITIONAL ENCODING FROM SCRATCH")
    print("=" * 60)
    
    # Configuration
    max_seq_len = 100  # Séquences jusqu'à 100 tokens
    d_model = 128      # Vecteurs de dimension 128 (doit être pair)
    batch_size = 4     # Batch de 4 séquences
    seq_len = 10       # Séquences de longueur 10
    
    print(f"\nConfiguration:")
    print(f"  - Longueur max de séquence: {max_seq_len}")
    print(f"  - Dimension d'encodage: {d_model}")
    print(f"  - Taille du batch: {batch_size}")
    print(f"  - Longueur de séquence: {seq_len}")
    
    # Validation complète des dimensions
    encodings = validate_positional_encoding_dimensions(
        max_seq_len, d_model, batch_size, seq_len
    )
    
    # Affichage d'informations détaillées
    pe_matrix = create_sinusoidal_positional_encoding(max_seq_len, d_model)
    sample_positions = np.array([0, 1, 2, 5, 10, 20, 50])
    print_positional_encoding_info(pe_matrix, sample_positions)
    
    # Visualisation du pattern
    visualize_positional_encoding_pattern(pe_matrix, num_positions=50)
    
    # Démonstration de la combinaison avec des token embeddings
    print("\n" + "=" * 60)
    print("DÉMONSTRATION: COMBINAISON TOKEN + POSITIONAL")
    print("=" * 60)
    
    # Création de faux token embeddings pour la démonstration
    token_embeddings = np.random.randn(batch_size, seq_len, d_model) * 0.01
    print(f"\nToken embeddings shape: {token_embeddings.shape}")
    print(f"Positional encodings shape: {encodings.shape}")
    
    # Combinaison
    combined = combine_token_and_positional_embeddings(token_embeddings, encodings)
    print(f"Combined embeddings shape: {combined.shape}")
    print(f"\n✓ Les embeddings ont été combinés avec succès!")
    
    print("\n" + "=" * 60)
    print("✓ Démonstration terminée avec succès!")
    print("=" * 60 + "\n")


# ============================================
# IMPLEMENTATION 2: PyTorch (nn.Module)
# ============================================
# Objectif: Code professionnel et efficace avec PyTorch

class PositionalEmbedding(nn.Module):
    """
    Implémentation PyTorch professionnelle de l'encodage positionnel.
    
    Cette classe utilise nn.Embedding pour stocker les encodages positionnels
    pré-calculés. Contrairement aux token embeddings, les poids ne sont PAS
    entraînés - ils sont fixés selon la formule sinusoïdale.
    
    Méthodes PyTorch utilisées:
    - nn.Embedding: Couche d'embedding utilisée comme lookup table
      * Paramètres:
        - num_embeddings (max_seq_len): Nombre de positions possibles
        - embedding_dim (d_model): Dimension des vecteurs de position
      * Attributs:
        - weight: Matrice d'encodage de shape (max_seq_len, d_model)
      * Comportement:
        - Input: LongTensor de shape (seq_len,) ou (batch_size, seq_len)
        - Output: FloatTensor de shape (seq_len, d_model) ou (batch_size, seq_len, d_model)
    
    - register_buffer: Enregistre un tensor comme partie du module mais non-entraînable
      * Utilisé pour stocker les encodages positionnels fixes
      * Le buffer est automatiquement déplacé sur GPU avec le modèle
      * Sauvegardé/chargé avec le modèle mais sans gradients
    
    - torch.arange: Crée une séquence de nombres [0, 1, 2, ..., n-1]
      * Équivalent à np.arange mais pour PyTorch
      * Supporte les opérations GPU
    
    - torch.sin / torch.cos: Fonctions trigonométriques
      * Appliquées élément par élément
      * Support GPU et différentiation automatique
    
    Différence clé avec TokenEmbedding:
    - TokenEmbedding: Poids apprenables (mis à jour pendant l'entraînement)
    - PositionalEmbedding: Poids fixes (calculés une fois, jamais modifiés)
    
    Args:
        max_seq_len: Longueur maximale de séquence supportée
        d_model: Dimension de l'encodage (doit être pair)
    
    Attributes:
        max_seq_len: Longueur maximale de séquence
        d_model: Dimension de l'encodage
        pe: Buffer contenant les encodages positionnels pré-calculés
    
    Shape:
        - Input: (batch_size, seq_len) - LongTensor ou juste (seq_len,)
        - Output: (batch_size, seq_len, d_model) - FloatTensor
    
    Exemple:
        >>> pos_emb = PositionalEmbedding(max_seq_len=100, d_model=128)
        >>> # Méthode 1: Passer les IDs de positions explicitement
        >>> positions = torch.arange(10).unsqueeze(0)  # (1, 10)
        >>> encodings = pos_emb(positions)
        >>> encodings.shape
        torch.Size([1, 10, 128])
        >>> 
        >>> # Méthode 2: Passer les token_ids, les positions sont créées automatiquement
        >>> token_ids = torch.randint(0, 1000, (2, 10))  # (2, 10)
        >>> encodings = pos_emb(token_ids)
        >>> encodings.shape
        torch.Size([2, 10, 128])
    """
    
    def __init__(self, max_seq_len: int, d_model: int):
        """
        Initialise la couche d'encodage positionnel.
        
        Calcule les encodages sinusoïdaux une seule fois et les stocke
        dans un buffer non-entraînable.
        
        Args:
            max_seq_len: Longueur maximale de séquence
            d_model: Dimension de l'encodage (doit être pair)
        """
        super().__init__()
        
        # Validation des dimensions
        assert max_seq_len > 0, f"max_seq_len doit être positif, reçu: {max_seq_len}"
        assert d_model > 0, f"d_model doit être positif, reçu: {d_model}"
        assert d_model % 2 == 0, f"d_model doit être pair pour l'encodage sinusoïdal, reçu: {d_model}"
        
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        # Création des encodages positionnels sinusoïdaux
        # Formule: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        #          PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        
        # Initialisation de la matrice d'encodage
        pe = torch.zeros(max_seq_len, d_model)
        
        # Création du vecteur de positions: [0, 1, 2, ..., max_seq_len-1]
        # Shape: (max_seq_len, 1) pour le broadcasting
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Calcul des diviseurs pour les différentes dimensions
        # div_term = 10000^(2i/d_model) pour i = 0, 1, 2, ..., d_model/2-1
        # Formule équivalente (plus stable): exp(-2i * log(10000) / d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * 
            -(math.log(10000.0) / d_model)
        )
        
        # Application des fonctions sinusoïdales
        # Dimensions paires (0, 2, 4, ...): sin
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Dimensions impaires (1, 3, 5, ...): cos
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Enregistrement comme buffer (non-entraînable)
        # register_buffer permet de:
        # - Sauvegarder le tensor avec le modèle
        # - Le déplacer automatiquement sur GPU avec le modèle
        # - Ne PAS calculer de gradients pour ce tensor
        self.register_buffer('pe', pe)
        
        # Note pédagogique: self.pe est maintenant accessible comme attribut
        # mais ne sera pas mis à jour pendant l'entraînement
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Récupère les encodages positionnels pour une séquence.
        
        Cette méthode accepte deux types d'input:
        1. Token IDs: (batch_size, seq_len) - crée automatiquement les positions
        2. Position IDs: (batch_size, seq_len) - utilise les positions fournies
        
        Dans les deux cas, retourne les encodages positionnels correspondants.
        
        Args:
            x: Tensor d'IDs (tokens ou positions), shape (batch_size, seq_len)
               Peut aussi être juste (seq_len,) pour une seule séquence
        
        Returns:
            encodings: Encodages positionnels, shape (batch_size, seq_len, d_model)
                      ou (seq_len, d_model) si input est (seq_len,)
        
        Validation des dimensions:
            - Input: (batch_size, seq_len) ou (seq_len,)
            - Output: (batch_size, seq_len, d_model) ou (seq_len, d_model)
        
        Exemple:
            >>> pos_emb = PositionalEmbedding(max_seq_len=100, d_model=64)
            >>> x = torch.randint(0, 1000, (2, 10))  # (2, 10)
            >>> out = pos_emb(x)
            >>> out.shape
            torch.Size([2, 10, 64])
        """
        # Détermination de la longueur de séquence
        if x.dim() == 1:
            seq_len = x.size(0)
            batch_size = None
        else:
            batch_size, seq_len = x.shape
        
        # Validation: la séquence ne doit pas dépasser max_seq_len
        assert seq_len <= self.max_seq_len, \
            f"La longueur de séquence ({seq_len}) dépasse max_seq_len ({self.max_seq_len})"
        
        # Récupération des encodages positionnels
        # On prend les seq_len premières positions: [0, 1, 2, ..., seq_len-1]
        # self.pe a shape (max_seq_len, d_model)
        # self.pe[:seq_len] a shape (seq_len, d_model)
        encodings = self.pe[:seq_len]
        
        # Si input est un batch, on ajoute la dimension batch
        if batch_size is not None:
            # unsqueeze(0) ajoute une dimension batch: (seq_len, d_model) -> (1, seq_len, d_model)
            # expand répète pour chaque élément du batch: (1, seq_len, d_model) -> (batch_size, seq_len, d_model)
            encodings = encodings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Vérification de la dimension de sortie
        if batch_size is not None:
            expected_shape = (batch_size, seq_len, self.d_model)
        else:
            expected_shape = (seq_len, self.d_model)
        
        assert encodings.shape == expected_shape, \
            f"Shape incorrecte: attendu {expected_shape}, obtenu {encodings.shape}"
        
        return encodings
    
    def get_positional_encoding_matrix(self) -> torch.Tensor:
        """
        Retourne la matrice complète d'encodage positionnel.
        
        Utile pour:
        - Inspection pédagogique des encodages
        - Visualisation des patterns sinusoïdaux
        - Analyse des fréquences
        
        Returns:
            pe: Matrice d'encodage, shape (max_seq_len, d_model)
        """
        return self.pe
    
    def extra_repr(self) -> str:
        """
        Représentation textuelle pour print(model).
        
        Returns:
            Description de la couche avec ses paramètres
        """
        return f'max_seq_len={self.max_seq_len}, d_model={self.d_model}'


# ============================================
# Fonctions utilitaires PyTorch
# ============================================

def check_positional_encoding_shape(input_tensor: torch.Tensor,
                                    encodings: torch.Tensor,
                                    expected_d_model: int) -> bool:
    """
    Vérifie que les dimensions des encodages positionnels sont correctes.
    Fonction pédagogique pour valider la compréhension des shapes.
    
    Args:
        input_tensor: Tensor d'input, shape (batch_size, seq_len) ou (seq_len,)
        encodings: Tensor d'encodages, shape (batch_size, seq_len, d_model) ou (seq_len, d_model)
        expected_d_model: Dimension d'encodage attendue
    
    Returns:
        True si les dimensions sont correctes, False sinon
    """
    if input_tensor.dim() == 1:
        seq_len = input_tensor.size(0)
        expected_shape = (seq_len, expected_d_model)
    else:
        batch_size, seq_len = input_tensor.shape
        expected_shape = (batch_size, seq_len, expected_d_model)
    
    print("\n" + "=" * 60)
    print("VÉRIFICATION DES DIMENSIONS - POSITIONAL ENCODING (PyTorch)")
    print("=" * 60)
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Output encodings shape: {encodings.shape}")
    print(f"Expected shape: {expected_shape}")
    
    if encodings.shape == expected_shape:
        print("✓ SUCCÈS: Les dimensions sont correctes!")
        print("=" * 60 + "\n")
        return True
    else:
        print("✗ ERREUR: Les dimensions ne correspondent pas!")
        print("=" * 60 + "\n")
        return False


def demonstrate_pytorch_positional_embedding():
    """
    Démonstration complète de l'utilisation de PositionalEmbedding PyTorch.
    Fonction pédagogique pour montrer toutes les fonctionnalités.
    """
    print("\n" + "=" * 60)
    print("DÉMONSTRATION: POSITIONAL EMBEDDING PYTORCH")
    print("=" * 60)
    
    # Configuration
    max_seq_len = 100
    d_model = 128
    batch_size = 4
    seq_len = 10
    
    print(f"\nConfiguration:")
    print(f"  - Longueur max de séquence: {max_seq_len}")
    print(f"  - Dimension d'encodage: {d_model}")
    print(f"  - Taille du batch: {batch_size}")
    print(f"  - Longueur de séquence: {seq_len}")
    
    # Création de la couche d'encodage positionnel
    print(f"\n1. Création de la couche PositionalEmbedding:")
    pos_emb = PositionalEmbedding(max_seq_len=max_seq_len, d_model=d_model)
    print(f"   ✓ Couche créée: {pos_emb}")
    
    # Vérification que les poids ne sont pas entraînables
    num_params = sum(p.numel() for p in pos_emb.parameters())
    print(f"   ✓ Nombre de paramètres entraînables: {num_params}")
    print(f"   ✓ Note: Les encodages positionnels sont FIXES (non-entraînables)")
    
    # Création de token IDs (ou position IDs)
    print(f"\n2. Création de token IDs:")
    token_ids = torch.randint(0, 1000, (batch_size, seq_len))
    print(f"   ✓ Shape des token_ids: {token_ids.shape}")
    print(f"   ✓ Type: {token_ids.dtype}")
    
    # Forward pass
    print(f"\n3. Forward pass (récupération des encodages):")
    encodings = pos_emb(token_ids)
    print(f"   ✓ Shape des encodings: {encodings.shape}")
    print(f"   ✓ Type: {encodings.dtype}")
    print(f"   ✓ Device: {encodings.device}")
    
    # Vérification des dimensions
    print(f"\n4. Vérification des dimensions:")
    check_positional_encoding_shape(token_ids, encodings, d_model)
    
    # Statistiques des encodages
    print(f"5. Statistiques des encodages:")
    print(f"   - Moyenne: {encodings.mean().item():.6f}")
    print(f"   - Écart-type: {encodings.std().item():.6f}")
    print(f"   - Min: {encodings.min().item():.6f}")
    print(f"   - Max: {encodings.max().item():.6f}")
    print(f"   - Valeurs dans [-1, 1]: {torch.all(torch.abs(encodings) <= 1.0).item()}")
    
    # Accès à la matrice d'encodage
    print(f"\n6. Accès à la matrice d'encodage:")
    pe_matrix = pos_emb.get_positional_encoding_matrix()
    print(f"   ✓ Shape de la matrice: {pe_matrix.shape}")
    print(f"   ✓ Accessible via: pos_emb.get_positional_encoding_matrix()")
    
    # Propriétés sinusoïdales
    print(f"\n7. Propriétés sinusoïdales:")
    print(f"   - Dimensions paires (0, 2, 4, ...): sin")
    print(f"   - Dimensions impaires (1, 3, 5, ...): cos")
    print(f"   - Fréquences: De haute (dim 0) à basse (dernière dim)")
    
    # Comparaison avec l'implémentation NumPy
    print(f"\n8. Comparaison avec NumPy:")
    print(f"   - NumPy: Implémentation manuelle, CPU uniquement")
    print(f"   - PyTorch: Optimisé, support GPU, intégration avec le modèle")
    print(f"   - Résultat: Même formule mathématique, mais PyTorch est plus efficace")
    
    print("\n" + "=" * 60)
    print("✓ Démonstration terminée avec succès!")
    print("=" * 60 + "\n")
    
    return pos_emb, token_ids, encodings


def demonstrate_combined_embeddings():
    """
    Démonstration de la combinaison des embeddings de tokens et positionnels.
    Montre comment les deux types d'embeddings sont utilisés ensemble.
    """
    print("\n" + "=" * 60)
    print("DÉMONSTRATION: COMBINAISON TOKEN + POSITIONAL (PyTorch)")
    print("=" * 60)
    
    # Configuration
    vocab_size = 1000
    max_seq_len = 100
    d_model = 128
    batch_size = 4
    seq_len = 10
    
    print(f"\nConfiguration:")
    print(f"  - Vocabulaire: {vocab_size}")
    print(f"  - Longueur max: {max_seq_len}")
    print(f"  - Dimension: {d_model}")
    print(f"  - Batch: {batch_size}")
    print(f"  - Séquence: {seq_len}")
    
    # Import de TokenEmbedding (défini dans token_embedding.py)
    from token_embedding import TokenEmbedding
    
    # Création des deux couches d'embedding
    print(f"\n1. Création des couches d'embedding:")
    token_emb = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)
    pos_emb = PositionalEmbedding(max_seq_len=max_seq_len, d_model=d_model)
    print(f"   ✓ TokenEmbedding créé")
    print(f"   ✓ PositionalEmbedding créé")
    
    # Création de token IDs
    print(f"\n2. Création de token IDs:")
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"   ✓ Shape: {token_ids.shape}")
    
    # Calcul des embeddings de tokens
    print(f"\n3. Calcul des embeddings de tokens:")
    token_embeddings = token_emb(token_ids)
    print(f"   ✓ Shape: {token_embeddings.shape}")
    
    # Calcul des encodages positionnels
    print(f"\n4. Calcul des encodages positionnels:")
    positional_encodings = pos_emb(token_ids)
    print(f"   ✓ Shape: {positional_encodings.shape}")
    
    # Combinaison par addition
    print(f"\n5. Combinaison par addition:")
    combined_embeddings = token_embeddings + positional_encodings
    print(f"   ✓ Shape: {combined_embeddings.shape}")
    print(f"   ✓ Formule: E_input = E_token + E_position")
    
    # Vérification
    print(f"\n6. Vérification:")
    expected_shape = (batch_size, seq_len, d_model)
    print(f"   - Shape attendue: {expected_shape}")
    print(f"   - Shape obtenue: {combined_embeddings.shape}")
    
    if combined_embeddings.shape == expected_shape:
        print(f"   ✓ SUCCÈS: Les embeddings sont correctement combinés!")
    
    print("\n" + "=" * 60)
    print("✓ Démonstration terminée avec succès!")
    print("=" * 60 + "\n")
    
    return token_emb, pos_emb, combined_embeddings


# ============================================
# Point d'entrée pour les tests
# ============================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DÉMONSTRATION: POSITIONAL ENCODING FROM SCRATCH")
    print("=" * 60)
    
    # Configuration
    max_seq_len = 100  # Séquences jusqu'à 100 tokens
    d_model = 128      # Vecteurs de dimension 128 (doit être pair)
    batch_size = 4     # Batch de 4 séquences
    seq_len = 10       # Séquences de longueur 10
    
    print(f"\nConfiguration:")
    print(f"  - Longueur max de séquence: {max_seq_len}")
    print(f"  - Dimension d'encodage: {d_model}")
    print(f"  - Taille du batch: {batch_size}")
    print(f"  - Longueur de séquence: {seq_len}")
    
    # Validation complète des dimensions
    encodings = validate_positional_encoding_dimensions(
        max_seq_len, d_model, batch_size, seq_len
    )
    
    # Affichage d'informations détaillées
    pe_matrix = create_sinusoidal_positional_encoding(max_seq_len, d_model)
    sample_positions = np.array([0, 1, 2, 5, 10, 20, 50])
    print_positional_encoding_info(pe_matrix, sample_positions)
    
    # Visualisation du pattern
    visualize_positional_encoding_pattern(pe_matrix, num_positions=50)
    
    # Démonstration de la combinaison avec des token embeddings
    print("\n" + "=" * 60)
    print("DÉMONSTRATION: COMBINAISON TOKEN + POSITIONAL")
    print("=" * 60)
    
    # Création de faux token embeddings pour la démonstration
    token_embeddings = np.random.randn(batch_size, seq_len, d_model) * 0.01
    print(f"\nToken embeddings shape: {token_embeddings.shape}")
    print(f"Positional encodings shape: {encodings.shape}")
    
    # Combinaison
    combined = combine_token_and_positional_embeddings(token_embeddings, encodings)
    print(f"Combined embeddings shape: {combined.shape}")
    print(f"\n✓ Les embeddings ont été combinés avec succès!")
    
    print("\n" + "=" * 60)
    print("✓ Démonstration NumPy terminée avec succès!")
    print("=" * 60 + "\n")
    
    # Démonstration PyTorch
    demonstrate_pytorch_positional_embedding()
    
    # Démonstration de la combinaison PyTorch
    try:
        demonstrate_combined_embeddings()
    except ImportError:
        print("\nNote: Pour la démonstration de combinaison, assurez-vous que")
        print("token_embedding.py est dans le même répertoire.")
