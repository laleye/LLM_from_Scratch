"""
Causal Masking Implementation

Ce module implémente le masque causal (triangulaire) utilisé dans les modèles
autorégressifs comme GPT. Le masque causal empêche chaque token de voir les
tokens futurs, ce qui est essentiel pour la génération de texte.

Concept clé:
    - BERT: Attention bidirectionnelle (pas de masque)
    - GPT: Attention causale (masque triangulaire)

Formule du masque:
    mask[i,j] = 1 si j <= i (peut voir le passé et le présent)
    mask[i,j] = 0 si j > i  (ne peut pas voir le futur)

Application dans l'attention:
    scores[mask == 0] = -inf
    → softmax(-inf) = 0
    → attention nulle aux positions futures
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


# ============================================
# IMPLEMENTATION 1: From Scratch (NumPy)
# ============================================
# Objectif: Comprendre le concept de masque triangulaire


def create_causal_mask_from_scratch(seq_len: int) -> np.ndarray:
    """
    Crée un masque causal (triangulaire inférieur) avec NumPy.
    
    Le masque causal est une matrice triangulaire inférieure qui empêche
    chaque token de voir les tokens futurs. C'est le mécanisme fondamental
    des modèles autorégressifs comme GPT.
    
    Concept:
        - Position i peut voir positions 0 à i (passé + présent)
        - Position i ne peut PAS voir positions i+1 à seq_len-1 (futur)
    
    Exemple pour seq_len=5:
        [[1, 0, 0, 0, 0],   ← Token 0 ne voit que lui-même
         [1, 1, 0, 0, 0],   ← Token 1 voit tokens 0 et 1
         [1, 1, 1, 0, 0],   ← Token 2 voit tokens 0, 1, 2
         [1, 1, 1, 1, 0],   ← Token 3 voit tokens 0, 1, 2, 3
         [1, 1, 1, 1, 1]]   ← Token 4 voit tous les tokens
    
    Pourquoi triangulaire inférieur?
        - Ligne i représente le token à la position i
        - Colonne j représente le token à la position j
        - mask[i,j] = 1 signifie "token i peut voir token j"
        - Pour l'autorégressif: on ne peut voir que le passé (j <= i)
    
    Args:
        seq_len: Longueur de la séquence
    
    Returns:
        Masque causal de shape (seq_len, seq_len)
        Valeurs: 1 = autoriser l'attention, 0 = bloquer l'attention
    
    Example:
        >>> mask = create_causal_mask_from_scratch(4)
        >>> print(mask)
        [[1. 0. 0. 0.]
         [1. 1. 0. 0.]
         [1. 1. 1. 0.]
         [1. 1. 1. 1.]]
    """
    # Méthode 1: Utiliser np.tril() (triangular lower)
    # np.tril() retourne la partie triangulaire inférieure d'une matrice
    # C'est la méthode la plus simple et efficace
    mask = np.tril(np.ones((seq_len, seq_len)))
    
    return mask


def create_causal_mask_manual(seq_len: int) -> np.ndarray:
    """
    Crée un masque causal manuellement (version pédagogique).
    
    Cette version montre explicitement comment construire le masque
    avec des boucles pour comprendre la logique.
    
    Args:
        seq_len: Longueur de la séquence
    
    Returns:
        Masque causal de shape (seq_len, seq_len)
    """
    # Initialiser une matrice de zéros
    mask = np.zeros((seq_len, seq_len))
    
    # Remplir la partie triangulaire inférieure
    # Pour chaque position i, autoriser l'attention aux positions 0 à i
    for i in range(seq_len):
        for j in range(seq_len):
            if j <= i:  # Position j est dans le passé ou présent
                mask[i, j] = 1.0
            # else: mask[i, j] reste 0 (futur bloqué)
    
    return mask


# ============================================
# IMPLEMENTATION 2: PyTorch
# ============================================
# Objectif: Code professionnel et efficace


def create_causal_mask(seq_len: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Crée un masque causal (triangulaire inférieur) avec PyTorch.
    
    Cette fonction utilise torch.tril() pour créer efficacement un masque
    triangulaire. Le masque est compatible GPU et peut être utilisé
    directement dans les opérations d'attention.
    
    torch.tril() - Triangular Lower:
        - Retourne la partie triangulaire inférieure d'une matrice
        - Tous les éléments au-dessus de la diagonale sont mis à 0
        - La diagonale et en-dessous restent inchangés
        - Opération très efficace (O(n²) mais optimisée)
    
    Pourquoi utiliser torch.tril()?
        - Optimisé GPU: Implémentation CUDA efficace
        - Différentiable: Compatible avec autograd (même si pas nécessaire ici)
        - Mémoire efficace: Pas de boucles Python
        - Lisible: Intent clair en une ligne
    
    Args:
        seq_len: Longueur de la séquence
        device: Device PyTorch (cpu ou cuda)
    
    Returns:
        Masque causal de shape (seq_len, seq_len)
        Type: torch.Tensor avec dtype=float
        Valeurs: 1.0 = autoriser l'attention, 0.0 = bloquer l'attention
    
    Example:
        >>> mask = create_causal_mask(4)
        >>> print(mask)
        tensor([[1., 0., 0., 0.],
                [1., 1., 0., 0.],
                [1., 1., 1., 0.],
                [1., 1., 1., 1.]])
        
        >>> # Utilisation avec GPU
        >>> mask_gpu = create_causal_mask(4, device=torch.device('cuda'))
        >>> print(mask_gpu.device)  # cuda:0
    
    Notes:
        - Le masque est créé sur le device spécifié (évite les transferts CPU↔GPU)
        - dtype=float pour compatibilité avec les opérations d'attention
        - Peut être broadcasté sur la dimension batch automatiquement
    """
    # torch.tril(): Retourne la partie triangulaire inférieure
    # 
    # Étapes:
    # 1. torch.ones(seq_len, seq_len, device=device): Crée une matrice de 1
    # 2. torch.tril(): Garde seulement la partie inférieure (diagonal incluse)
    # 
    # Résultat: Matrice triangulaire inférieure de 1 et 0
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    
    return mask


# ============================================
# FONCTIONS DE VISUALISATION
# ============================================


def visualize_causal_mask(seq_len: int = 8, save_path: Optional[str] = None):
    """
    Visualise le masque causal sous forme de heatmap.
    
    Cette fonction crée une visualisation claire du masque causal pour
    comprendre comment il bloque l'attention aux positions futures.
    
    Args:
        seq_len: Longueur de la séquence à visualiser
        save_path: Chemin optionnel pour sauvegarder la figure
    
    Example:
        >>> visualize_causal_mask(seq_len=8)
        >>> visualize_causal_mask(seq_len=10, save_path='causal_mask.png')
    """
    # Créer le masque
    mask = create_causal_mask_from_scratch(seq_len)
    
    # Créer la figure
    plt.figure(figsize=(8, 7))
    
    # Créer la heatmap avec seaborn
    # cmap='RdYlGn': Rouge (bloqué) → Jaune → Vert (autorisé)
    # annot=True: Afficher les valeurs dans les cellules
    # fmt='.0f': Format des annotations (entiers)
    # cbar_kws: Personnaliser la barre de couleur
    sns.heatmap(
        mask,
        annot=True,
        fmt='.0f',
        cmap='RdYlGn',
        cbar_kws={'label': 'Attention autorisée'},
        square=True,
        linewidths=0.5,
        linecolor='gray'
    )
    
    plt.xlabel('Position Key (ce qu\'on regarde)', fontsize=12)
    plt.ylabel('Position Query (qui regarde)', fontsize=12)
    plt.title(f'Masque Causal pour seq_len={seq_len}\n'
              f'1 = Attention autorisée, 0 = Attention bloquée',
              fontsize=14, pad=20)
    
    # Ajouter des annotations explicatives
    plt.text(seq_len/2, -0.5, '← Passé | Futur →', 
             ha='center', va='top', fontsize=10, style='italic')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure sauvegardée: {save_path}")
    
    plt.show()


def compare_attention_patterns(seq_len: int = 6):
    """
    Compare les patterns d'attention BERT (bidirectionnel) vs GPT (causal).
    
    Cette fonction crée une visualisation côte à côte montrant la différence
    fondamentale entre les deux architectures.
    
    Args:
        seq_len: Longueur de la séquence à visualiser
    
    Example:
        >>> compare_attention_patterns(seq_len=6)
    """
    # Créer les deux types de masques
    bert_mask = np.ones((seq_len, seq_len))  # Pas de masque (tous 1)
    gpt_mask = create_causal_mask_from_scratch(seq_len)  # Masque causal
    
    # Créer la figure avec deux sous-graphiques
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # BERT: Attention bidirectionnelle
    sns.heatmap(
        bert_mask,
        annot=True,
        fmt='.0f',
        cmap='Greens',
        ax=axes[0],
        cbar_kws={'label': 'Attention'},
        square=True,
        linewidths=0.5
    )
    axes[0].set_xlabel('Position Key')
    axes[0].set_ylabel('Position Query')
    axes[0].set_title('BERT: Attention Bidirectionnelle\n'
                      'Tous les tokens se voient mutuellement',
                      fontsize=12, pad=15)
    
    # GPT: Attention causale
    sns.heatmap(
        gpt_mask,
        annot=True,
        fmt='.0f',
        cmap='RdYlGn',
        ax=axes[1],
        cbar_kws={'label': 'Attention'},
        square=True,
        linewidths=0.5
    )
    axes[1].set_xlabel('Position Key')
    axes[1].set_ylabel('Position Query')
    axes[1].set_title('GPT: Attention Causale (Autorégressif)\n'
                      'Chaque token ne voit que le passé',
                      fontsize=12, pad=15)
    
    plt.suptitle('Comparaison: BERT vs GPT', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


def visualize_mask_effect_on_attention(seq_len: int = 5):
    """
    Visualise l'effet du masque causal sur les poids d'attention.
    
    Cette fonction montre comment le masque transforme les scores d'attention
    en bloquant les positions futures.
    
    Args:
        seq_len: Longueur de la séquence
    
    Example:
        >>> visualize_mask_effect_on_attention(seq_len=5)
    """
    # Créer des scores d'attention aléatoires (avant softmax)
    np.random.seed(42)
    scores = np.random.randn(seq_len, seq_len)
    
    # Créer le masque causal
    mask = create_causal_mask_from_scratch(seq_len)
    
    # Appliquer le masque (remplacer 0 par -inf)
    masked_scores = np.where(mask == 0, -np.inf, scores)
    
    # Calculer les poids d'attention (softmax)
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    attention_no_mask = softmax(scores)
    attention_with_mask = softmax(masked_scores)
    
    # Créer la visualisation
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Scores bruts
    sns.heatmap(scores, annot=True, fmt='.2f', cmap='coolwarm', 
                ax=axes[0, 0], center=0, square=True)
    axes[0, 0].set_title('1. Scores d\'attention bruts (avant masque)')
    axes[0, 0].set_xlabel('Key')
    axes[0, 0].set_ylabel('Query')
    
    # 2. Masque causal
    sns.heatmap(mask, annot=True, fmt='.0f', cmap='RdYlGn',
                ax=axes[0, 1], square=True)
    axes[0, 1].set_title('2. Masque causal (1=autorisé, 0=bloqué)')
    axes[0, 1].set_xlabel('Key')
    axes[0, 1].set_ylabel('Query')
    
    # 3. Attention sans masque
    sns.heatmap(attention_no_mask, annot=True, fmt='.3f', cmap='YlOrRd',
                ax=axes[1, 0], vmin=0, vmax=1, square=True)
    axes[1, 0].set_title('3. Attention SANS masque (BERT)\nTous les tokens visibles')
    axes[1, 0].set_xlabel('Key')
    axes[1, 0].set_ylabel('Query')
    
    # 4. Attention avec masque
    sns.heatmap(attention_with_mask, annot=True, fmt='.3f', cmap='YlOrRd',
                ax=axes[1, 1], vmin=0, vmax=1, square=True)
    axes[1, 1].set_title('4. Attention AVEC masque (GPT)\nFutur bloqué (attention=0)')
    axes[1, 1].set_xlabel('Key')
    axes[1, 1].set_ylabel('Query')
    
    plt.suptitle('Effet du Masque Causal sur l\'Attention', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.show()
    
    # Afficher les statistiques
    print("\n" + "="*60)
    print("STATISTIQUES")
    print("="*60)
    print(f"\nSans masque:")
    print(f"  - Somme par ligne (doit être 1.0): {attention_no_mask.sum(axis=1)}")
    print(f"  - Nombre de poids non-nuls par ligne: {(attention_no_mask > 0).sum(axis=1)}")
    
    print(f"\nAvec masque causal:")
    print(f"  - Somme par ligne (doit être 1.0): {attention_with_mask.sum(axis=1)}")
    print(f"  - Nombre de poids non-nuls par ligne: {(attention_with_mask > 0).sum(axis=1)}")
    print(f"\nObservation:")
    print(f"  - Ligne i a (i+1) poids non-nuls (peut voir positions 0 à i)")
    print(f"  - La partie supérieure droite est nulle (futur bloqué)")


# ============================================
# EXEMPLES D'UTILISATION
# ============================================


def example_causal_mask_numpy():
    """
    Exemple pédagogique du masque causal avec NumPy.
    """
    print("\n" + "="*60)
    print("EXEMPLE: Masque Causal avec NumPy")
    print("="*60 + "\n")
    
    seq_len = 6
    
    # Méthode 1: Avec np.tril()
    print("Méthode 1: Utiliser np.tril() (recommandé)")
    mask_tril = create_causal_mask_from_scratch(seq_len)
    print(f"Shape: {mask_tril.shape}")
    print(mask_tril)
    
    # Méthode 2: Manuelle (pédagogique)
    print("\nMéthode 2: Construction manuelle (pédagogique)")
    mask_manual = create_causal_mask_manual(seq_len)
    print(f"Shape: {mask_manual.shape}")
    print(mask_manual)
    
    # Vérifier qu'elles sont identiques
    print(f"\nLes deux méthodes donnent le même résultat? {np.allclose(mask_tril, mask_manual)}")
    
    # Interpréter le masque
    print("\n" + "-"*60)
    print("INTERPRÉTATION")
    print("-"*60)
    for i in range(seq_len):
        visible_positions = np.where(mask_tril[i] == 1)[0]
        print(f"Token {i} peut voir les positions: {list(visible_positions)} "
              f"(total: {len(visible_positions)} tokens)")


def example_causal_mask_pytorch():
    """
    Exemple pédagogique du masque causal avec PyTorch.
    """
    print("\n" + "="*60)
    print("EXEMPLE: Masque Causal avec PyTorch")
    print("="*60 + "\n")
    
    seq_len = 6
    
    # Créer le masque sur CPU
    print("Masque sur CPU:")
    mask_cpu = create_causal_mask(seq_len, device=torch.device('cpu'))
    print(f"Shape: {mask_cpu.shape}")
    print(f"Device: {mask_cpu.device}")
    print(f"Dtype: {mask_cpu.dtype}")
    print(mask_cpu)
    
    # Créer le masque sur GPU (si disponible)
    if torch.cuda.is_available():
        print("\n" + "-"*60)
        print("Masque sur GPU:")
        mask_gpu = create_causal_mask(seq_len, device=torch.device('cuda'))
        print(f"Shape: {mask_gpu.shape}")
        print(f"Device: {mask_gpu.device}")
        print(f"Dtype: {mask_gpu.dtype}")
        print(mask_gpu)
        
        # Vérifier que CPU et GPU donnent le même résultat
        print(f"\nCPU vs GPU identiques? {torch.allclose(mask_cpu, mask_gpu.cpu())}")
    else:
        print("\nGPU non disponible, test sur CPU uniquement")
    
    # Montrer comment utiliser le masque dans l'attention
    print("\n" + "-"*60)
    print("UTILISATION DANS L'ATTENTION")
    print("-"*60)
    print("\nCode typique:")
    print("""
    # Dans la fonction forward de l'attention:
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Créer le masque causal
    mask = create_causal_mask(seq_len, device=scores.device)
    
    # Appliquer le masque
    scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Softmax (les positions masquées auront attention=0)
    attention_weights = F.softmax(scores, dim=-1)
    """)


if __name__ == "__main__":
    # Exécuter les exemples
    example_causal_mask_numpy()
    example_causal_mask_pytorch()
    
    # Créer les visualisations
    print("\n" + "="*60)
    print("VISUALISATIONS")
    print("="*60 + "\n")
    
    print("1. Visualisation du masque causal...")
    visualize_causal_mask(seq_len=8)
    
    print("\n2. Comparaison BERT vs GPT...")
    compare_attention_patterns(seq_len=6)
    
    print("\n3. Effet du masque sur l'attention...")
    visualize_mask_effect_on_attention(seq_len=5)
