"""
Token Embedding Implementation

This module provides both from-scratch (NumPy) and PyTorch implementations
of token embeddings for the Mini-GPT pedagogical notebook.

Token embeddings convert discrete token IDs into dense continuous vector
representations that the neural network can process.
"""

import numpy as np
import torch
import torch.nn as nn


# ============================================
# IMPLEMENTATION 1: From Scratch (NumPy)
# ============================================
# Objectif: Comprendre le formalisme mathématique de l'embedding

def create_token_embedding_matrix(vocab_size: int, d_model: int, seed: int = 42) -> np.ndarray:
    """
    Crée une matrice d'embedding de tokens initialisée aléatoirement.
    
    Formule mathématique:
    - E ∈ ℝ^(vocab_size × d_model)
    - Chaque ligne E[i] représente le vecteur d'embedding pour le token i
    
    Args:
        vocab_size: Taille du vocabulaire (nombre de tokens uniques)
        d_model: Dimension de l'embedding (taille du vecteur pour chaque token)
        seed: Graine aléatoire pour la reproductibilité
    
    Returns:
        embedding_matrix: Matrice d'embedding de shape (vocab_size, d_model)
    
    Validation des dimensions:
        - Input: vocab_size (int), d_model (int)
        - Output: (vocab_size, d_model)
    """
    # Validation des dimensions
    assert vocab_size > 0, f"vocab_size doit être positif, reçu: {vocab_size}"
    assert d_model > 0, f"d_model doit être positif, reçu: {d_model}"
    
    # Initialisation aléatoire avec distribution normale
    # Utilise une petite variance pour éviter des valeurs trop grandes
    np.random.seed(seed)
    embedding_matrix = np.random.randn(vocab_size, d_model) * 0.01
    
    # Vérification de la dimension de sortie
    assert embedding_matrix.shape == (vocab_size, d_model), \
        f"Shape incorrecte: attendu ({vocab_size}, {d_model}), obtenu {embedding_matrix.shape}"
    
    return embedding_matrix


def token_embedding_lookup(token_ids: np.ndarray, embedding_matrix: np.ndarray) -> np.ndarray:
    """
    Effectue une recherche (lookup) dans la matrice d'embedding pour convertir
    les IDs de tokens en vecteurs d'embedding.
    
    Formule mathématique:
    - Pour un token ID i, on récupère la ligne E[i] de la matrice d'embedding
    - Pour une séquence [i₁, i₂, ..., iₙ], on obtient [E[i₁], E[i₂], ..., E[iₙ]]
    
    Args:
        token_ids: IDs des tokens, shape (batch_size, seq_len) ou (seq_len,)
        embedding_matrix: Matrice d'embedding, shape (vocab_size, d_model)
    
    Returns:
        embeddings: Vecteurs d'embedding, shape (*token_ids.shape, d_model)
    
    Validation des dimensions:
        - Input token_ids: (batch_size, seq_len) ou (seq_len,)
        - Input embedding_matrix: (vocab_size, d_model)
        - Output: (*token_ids.shape, d_model)
    
    Exemple:
        >>> embedding_matrix = create_token_embedding_matrix(vocab_size=100, d_model=64)
        >>> token_ids = np.array([[5, 10, 15], [20, 25, 30]])  # (2, 3)
        >>> embeddings = token_embedding_lookup(token_ids, embedding_matrix)
        >>> embeddings.shape
        (2, 3, 64)
    """
    vocab_size, d_model = embedding_matrix.shape
    
    # Validation: vérifier que tous les token IDs sont dans le vocabulaire
    assert np.all(token_ids >= 0), "Les token IDs doivent être non-négatifs"
    assert np.all(token_ids < vocab_size), \
        f"Tous les token IDs doivent être < vocab_size ({vocab_size}), max trouvé: {np.max(token_ids)}"
    
    # Lookup: récupération des vecteurs d'embedding
    # NumPy permet l'indexation avancée pour récupérer plusieurs lignes
    embeddings = embedding_matrix[token_ids]
    
    # Vérification de la dimension de sortie
    expected_shape = (*token_ids.shape, d_model)
    assert embeddings.shape == expected_shape, \
        f"Shape incorrecte: attendu {expected_shape}, obtenu {embeddings.shape}"
    
    return embeddings


# ============================================
# Fonctions utilitaires pour la pédagogie
# ============================================

def print_embedding_info(embedding_matrix: np.ndarray, token_ids: np.ndarray = None):
    """
    Affiche des informations sur la matrice d'embedding pour la validation pédagogique.
    
    Args:
        embedding_matrix: Matrice d'embedding
        token_ids: (Optionnel) IDs de tokens pour afficher leurs embeddings
    """
    vocab_size, d_model = embedding_matrix.shape
    
    print("=" * 60)
    print("INFORMATIONS SUR L'EMBEDDING DE TOKENS")
    print("=" * 60)
    print(f"Taille du vocabulaire (vocab_size): {vocab_size}")
    print(f"Dimension de l'embedding (d_model): {d_model}")
    print(f"Shape de la matrice d'embedding: {embedding_matrix.shape}")
    print(f"Nombre total de paramètres: {vocab_size * d_model:,}")
    print(f"Statistiques de la matrice:")
    print(f"  - Moyenne: {np.mean(embedding_matrix):.6f}")
    print(f"  - Écart-type: {np.std(embedding_matrix):.6f}")
    print(f"  - Min: {np.min(embedding_matrix):.6f}")
    print(f"  - Max: {np.max(embedding_matrix):.6f}")
    
    if token_ids is not None:
        print(f"\nExemple de lookup pour token_ids: {token_ids}")
        embeddings = token_embedding_lookup(token_ids, embedding_matrix)
        print(f"Shape des embeddings résultants: {embeddings.shape}")
        if embeddings.ndim == 2:
            print(f"Premier embedding (token {token_ids[0]}):")
            print(f"  {embeddings[0][:5]}... (premiers 5 éléments)")
    
    print("=" * 60)


def validate_embedding_dimensions(vocab_size: int, d_model: int, 
                                  batch_size: int, seq_len: int):
    """
    Valide les dimensions à travers le processus d'embedding complet.
    Fonction pédagogique pour vérifier la compréhension des shapes.
    
    Args:
        vocab_size: Taille du vocabulaire
        d_model: Dimension de l'embedding
        batch_size: Taille du batch
        seq_len: Longueur de la séquence
    """
    print("\n" + "=" * 60)
    print("VALIDATION DES DIMENSIONS - TOKEN EMBEDDING")
    print("=" * 60)
    
    # Étape 1: Création de la matrice d'embedding
    print(f"\n1. Création de la matrice d'embedding:")
    print(f"   vocab_size={vocab_size}, d_model={d_model}")
    embedding_matrix = create_token_embedding_matrix(vocab_size, d_model)
    print(f"   ✓ Shape de la matrice: {embedding_matrix.shape}")
    
    # Étape 2: Création de token IDs aléatoires
    print(f"\n2. Création de token IDs:")
    print(f"   batch_size={batch_size}, seq_len={seq_len}")
    token_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
    print(f"   ✓ Shape des token_ids: {token_ids.shape}")
    
    # Étape 3: Lookup des embeddings
    print(f"\n3. Lookup des embeddings:")
    embeddings = token_embedding_lookup(token_ids, embedding_matrix)
    print(f"   ✓ Shape des embeddings: {embeddings.shape}")
    
    # Étape 4: Vérification finale
    expected_shape = (batch_size, seq_len, d_model)
    print(f"\n4. Vérification finale:")
    print(f"   Shape attendue: {expected_shape}")
    print(f"   Shape obtenue: {embeddings.shape}")
    
    if embeddings.shape == expected_shape:
        print(f"   ✓ SUCCÈS: Les dimensions sont correctes!")
    else:
        print(f"   ✗ ERREUR: Les dimensions ne correspondent pas!")
    
    print("=" * 60 + "\n")
    
    return embeddings


# ============================================
# Exemple d'utilisation pédagogique
# ============================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DÉMONSTRATION: TOKEN EMBEDDING FROM SCRATCH")
    print("=" * 60)
    
    # Configuration
    vocab_size = 1000  # Vocabulaire de 1000 tokens
    d_model = 128      # Vecteurs de dimension 128
    batch_size = 4     # Batch de 4 séquences
    seq_len = 10       # Séquences de longueur 10
    
    print(f"\nConfiguration:")
    print(f"  - Vocabulaire: {vocab_size} tokens")
    print(f"  - Dimension d'embedding: {d_model}")
    print(f"  - Taille du batch: {batch_size}")
    print(f"  - Longueur de séquence: {seq_len}")
    
    # Validation complète des dimensions
    embeddings = validate_embedding_dimensions(vocab_size, d_model, batch_size, seq_len)
    
    # Affichage d'informations détaillées
    embedding_matrix = create_token_embedding_matrix(vocab_size, d_model)
    sample_token_ids = np.array([0, 1, 2, 3, 4])
    print_embedding_info(embedding_matrix, sample_token_ids)
    
    print("\n✓ Démonstration terminée avec succès!")


# ============================================
# IMPLEMENTATION 2: PyTorch (nn.Module)
# ============================================
# Objectif: Code professionnel et efficace avec PyTorch

class TokenEmbedding(nn.Module):
    """
    Implémentation PyTorch professionnelle de l'embedding de tokens.
    
    Cette classe utilise nn.Embedding, une couche PyTorch optimisée qui:
    - Stocke une matrice d'embedding de shape (vocab_size, d_model)
    - Effectue des lookups efficaces sur GPU
    - Supporte la différentiation automatique pour l'entraînement
    - Gère automatiquement les batches et les dimensions
    
    Méthodes PyTorch utilisées:
    - nn.Embedding: Couche d'embedding avec matrice de poids apprenables
      * Paramètres:
        - num_embeddings (vocab_size): Taille du vocabulaire
        - embedding_dim (d_model): Dimension des vecteurs d'embedding
      * Attributs:
        - weight: Matrice d'embedding de shape (vocab_size, d_model)
      * Comportement:
        - Input: LongTensor de shape (batch_size, seq_len) contenant les IDs
        - Output: FloatTensor de shape (batch_size, seq_len, d_model)
    
    Avantages par rapport à l'implémentation NumPy:
    - Calculs GPU automatiques si disponible
    - Gradients calculés automatiquement pour l'entraînement
    - Optimisations internes pour les lookups
    - Intégration native avec le reste du modèle PyTorch
    
    Args:
        vocab_size: Taille du vocabulaire (nombre de tokens uniques)
        d_model: Dimension de l'embedding (taille du vecteur pour chaque token)
    
    Attributes:
        embedding: Couche nn.Embedding contenant la matrice de poids
    
    Shape:
        - Input: (batch_size, seq_len) - LongTensor d'IDs de tokens
        - Output: (batch_size, seq_len, d_model) - FloatTensor d'embeddings
    
    Exemple:
        >>> token_emb = TokenEmbedding(vocab_size=1000, d_model=128)
        >>> token_ids = torch.tensor([[5, 10, 15], [20, 25, 30]])  # (2, 3)
        >>> embeddings = token_emb(token_ids)
        >>> embeddings.shape
        torch.Size([2, 3, 128])
    """
    
    def __init__(self, vocab_size: int, d_model: int):
        """
        Initialise la couche d'embedding de tokens.
        
        Args:
            vocab_size: Taille du vocabulaire
            d_model: Dimension de l'embedding
        """
        super().__init__()
        
        # Validation des dimensions
        assert vocab_size > 0, f"vocab_size doit être positif, reçu: {vocab_size}"
        assert d_model > 0, f"d_model doit être positif, reçu: {d_model}"
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Création de la couche d'embedding PyTorch
        # nn.Embedding initialise automatiquement les poids avec une distribution normale
        self.embedding = nn.Embedding(num_embeddings=vocab_size, 
                                     embedding_dim=d_model)
        
        # Note pédagogique: La matrice de poids est accessible via self.embedding.weight
        # Shape: (vocab_size, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Effectue le lookup des embeddings pour les token IDs fournis.
        
        Opération effectuée:
        - Pour chaque token ID dans x, récupère le vecteur d'embedding correspondant
        - Équivalent à: self.embedding.weight[x]
        - Mais optimisé et avec support des gradients
        
        Args:
            x: Tensor d'IDs de tokens, shape (batch_size, seq_len)
               Doit être de type LongTensor (torch.long)
        
        Returns:
            embeddings: Tensor d'embeddings, shape (batch_size, seq_len, d_model)
        
        Validation des dimensions:
            - Input: (batch_size, seq_len) - LongTensor
            - Output: (batch_size, seq_len, d_model) - FloatTensor
        
        Exemple:
            >>> token_emb = TokenEmbedding(vocab_size=100, d_model=64)
            >>> x = torch.tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
            >>> out = token_emb(x)
            >>> out.shape
            torch.Size([2, 3, 64])
        """
        # Validation du type d'entrée
        assert x.dtype in [torch.long, torch.int, torch.int32, torch.int64], \
            f"Les token IDs doivent être de type entier (long), reçu: {x.dtype}"
        
        # Validation des valeurs
        assert torch.all(x >= 0), "Les token IDs doivent être non-négatifs"
        assert torch.all(x < self.vocab_size), \
            f"Tous les token IDs doivent être < vocab_size ({self.vocab_size}), " \
            f"max trouvé: {torch.max(x).item()}"
        
        # Lookup des embeddings
        # nn.Embedding gère automatiquement le batching et les dimensions
        embeddings = self.embedding(x)
        
        # Vérification de la dimension de sortie
        expected_shape = (*x.shape, self.d_model)
        assert embeddings.shape == expected_shape, \
            f"Shape incorrecte: attendu {expected_shape}, obtenu {embeddings.shape}"
        
        return embeddings
    
    def get_embedding_matrix(self) -> torch.Tensor:
        """
        Retourne la matrice d'embedding complète.
        
        Utile pour:
        - Inspection pédagogique des poids
        - Visualisation de l'espace d'embedding
        - Analyse de similarité entre tokens
        
        Returns:
            weight: Matrice d'embedding, shape (vocab_size, d_model)
        """
        return self.embedding.weight
    
    def extra_repr(self) -> str:
        """
        Représentation textuelle pour print(model).
        
        Returns:
            Description de la couche avec ses paramètres
        """
        return f'vocab_size={self.vocab_size}, d_model={self.d_model}'


# ============================================
# Fonctions utilitaires PyTorch
# ============================================

def check_embedding_shape(token_ids: torch.Tensor, 
                         embeddings: torch.Tensor,
                         expected_d_model: int) -> bool:
    """
    Vérifie que les dimensions des embeddings sont correctes.
    Fonction pédagogique pour valider la compréhension des shapes.
    
    Args:
        token_ids: Tensor d'IDs de tokens, shape (batch_size, seq_len)
        embeddings: Tensor d'embeddings, shape (batch_size, seq_len, d_model)
        expected_d_model: Dimension d'embedding attendue
    
    Returns:
        True si les dimensions sont correctes, False sinon
    """
    batch_size, seq_len = token_ids.shape
    expected_shape = (batch_size, seq_len, expected_d_model)
    
    print("\n" + "=" * 60)
    print("VÉRIFICATION DES DIMENSIONS - TOKEN EMBEDDING (PyTorch)")
    print("=" * 60)
    print(f"Input token_ids shape: {token_ids.shape}")
    print(f"Output embeddings shape: {embeddings.shape}")
    print(f"Expected shape: {expected_shape}")
    
    if embeddings.shape == expected_shape:
        print("✓ SUCCÈS: Les dimensions sont correctes!")
        print("=" * 60 + "\n")
        return True
    else:
        print("✗ ERREUR: Les dimensions ne correspondent pas!")
        print("=" * 60 + "\n")
        return False


def demonstrate_pytorch_token_embedding():
    """
    Démonstration complète de l'utilisation de TokenEmbedding PyTorch.
    Fonction pédagogique pour montrer toutes les fonctionnalités.
    """
    print("\n" + "=" * 60)
    print("DÉMONSTRATION: TOKEN EMBEDDING PYTORCH")
    print("=" * 60)
    
    # Configuration
    vocab_size = 1000
    d_model = 128
    batch_size = 4
    seq_len = 10
    
    print(f"\nConfiguration:")
    print(f"  - Vocabulaire: {vocab_size} tokens")
    print(f"  - Dimension d'embedding: {d_model}")
    print(f"  - Taille du batch: {batch_size}")
    print(f"  - Longueur de séquence: {seq_len}")
    
    # Création de la couche d'embedding
    print(f"\n1. Création de la couche TokenEmbedding:")
    token_emb = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)
    print(f"   ✓ Couche créée: {token_emb}")
    
    # Comptage des paramètres
    num_params = sum(p.numel() for p in token_emb.parameters())
    print(f"   ✓ Nombre de paramètres: {num_params:,} ({vocab_size} × {d_model})")
    
    # Création de token IDs aléatoires
    print(f"\n2. Création de token IDs:")
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"   ✓ Shape des token_ids: {token_ids.shape}")
    print(f"   ✓ Type: {token_ids.dtype}")
    print(f"   ✓ Exemple: {token_ids[0]}")
    
    # Forward pass
    print(f"\n3. Forward pass (lookup des embeddings):")
    embeddings = token_emb(token_ids)
    print(f"   ✓ Shape des embeddings: {embeddings.shape}")
    print(f"   ✓ Type: {embeddings.dtype}")
    print(f"   ✓ Device: {embeddings.device}")
    
    # Vérification des dimensions
    print(f"\n4. Vérification des dimensions:")
    check_embedding_shape(token_ids, embeddings, d_model)
    
    # Statistiques des embeddings
    print(f"5. Statistiques des embeddings:")
    print(f"   - Moyenne: {embeddings.mean().item():.6f}")
    print(f"   - Écart-type: {embeddings.std().item():.6f}")
    print(f"   - Min: {embeddings.min().item():.6f}")
    print(f"   - Max: {embeddings.max().item():.6f}")
    
    # Accès à la matrice d'embedding
    print(f"\n6. Accès à la matrice d'embedding:")
    weight_matrix = token_emb.get_embedding_matrix()
    print(f"   ✓ Shape de la matrice: {weight_matrix.shape}")
    print(f"   ✓ Accessible via: token_emb.get_embedding_matrix()")
    
    # Comparaison avec l'implémentation NumPy
    print(f"\n7. Comparaison avec NumPy:")
    print(f"   - NumPy: Implémentation manuelle, CPU uniquement")
    print(f"   - PyTorch: Optimisé, support GPU, gradients automatiques")
    print(f"   - Résultat: Même fonctionnalité, mais PyTorch est plus efficace")
    
    print("\n" + "=" * 60)
    print("✓ Démonstration terminée avec succès!")
    print("=" * 60 + "\n")
    
    return token_emb, token_ids, embeddings


# ============================================
# Point d'entrée pour les tests
# ============================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DÉMONSTRATION: TOKEN EMBEDDING FROM SCRATCH")
    print("=" * 60)
    
    # Configuration
    vocab_size = 1000  # Vocabulaire de 1000 tokens
    d_model = 128      # Vecteurs de dimension 128
    batch_size = 4     # Batch de 4 séquences
    seq_len = 10       # Séquences de longueur 10
    
    print(f"\nConfiguration:")
    print(f"  - Vocabulaire: {vocab_size} tokens")
    print(f"  - Dimension d'embedding: {d_model}")
    print(f"  - Taille du batch: {batch_size}")
    print(f"  - Longueur de séquence: {seq_len}")
    
    # Validation complète des dimensions
    embeddings = validate_embedding_dimensions(vocab_size, d_model, batch_size, seq_len)
    
    # Affichage d'informations détaillées
    embedding_matrix = create_token_embedding_matrix(vocab_size, d_model)
    sample_token_ids = np.array([0, 1, 2, 3, 4])
    print_embedding_info(embedding_matrix, sample_token_ids)
    
    print("\n✓ Démonstration NumPy terminée avec succès!")
    
    # Démonstration PyTorch
    demonstrate_pytorch_token_embedding()
