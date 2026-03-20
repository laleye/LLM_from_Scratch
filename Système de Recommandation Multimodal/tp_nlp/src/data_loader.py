"""
Module de chargement et de validation des données produits.

Ce module fournit des fonctions pour :
- Charger le dataset de produits depuis un fichier CSV
- Valider que le dataset respecte les exigences du projet
- Générer des statistiques sur le dataset
- Visualiser la distribution des catégories

Auteur : Master NLP Avancé
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


# ==================== CONSTANTES ====================

# Colonnes obligatoires du dataset
REQUIRED_COLUMNS = [
    'id', 'nom', 'description', 'categorie',
    'sous_categorie', 'prix', 'image_path'
]

# Colonnes recommandées
RECOMMENDED_COLUMNS = ['couleur', 'matiere']

# Exigences du dataset
MIN_PRODUCTS = 300
MIN_CATEGORIES = 8
MIN_IMAGE_SIZE = (224, 224)


# ==================== FONCTIONS DE CHARGEMENT ====================

def load_products(csv_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
    """
    Charge le dataset de produits depuis un fichier CSV.

    Parameters
    ----------
    csv_path : str
        Chemin vers le fichier CSV contenant les données produits
    encoding : str, optional
        Encodage du fichier (default: 'utf-8')

    Returns
    -------
    pd.DataFrame
        DataFrame contenant les données produits

    Raises
    ------
    FileNotFoundError
        Si le fichier CSV n'existe pas
    ValueError
        Si le CSV ne contient pas les colonnes requises

    Examples
    --------
    >>> df = load_products('data/products.csv')
    >>> print(df.head())
       id        nom        description  categorie  ...
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Le fichier {csv_path} n'existe pas.")

    # Charger le CSV
    df = pd.read_csv(csv_path, encoding=encoding)

    # Vérifier les colonnes obligatoires
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Colonnes manquantes dans le CSV: {missing_cols}. "
            f"Colonnes requises: {REQUIRED_COLUMNS}"
        )

    # Ajouter les colonnes recommandées si elles n'existent pas
    for col in RECOMMENDED_COLUMNS:
        if col not in df.columns:
            df[col] = None
            print(f"⚠️  Colonne recommandée '{col}' ajoutée avec des valeurs None")

    print(f"✅ Dataset chargé: {len(df)} produits, {len(df.columns)} colonnes")

    return df


def load_annotations(csv_path: str) -> pd.DataFrame:
    """
    Charge les annotations de similarité pour les produits ancres.

    Parameters
    ----------
    csv_path : str
        Chemin vers le fichier CSV d'annotations

    Returns
    -------
    pd.DataFrame
        DataFrame contenant les annotations (produit_ancre, voisins globaux/visuels/sémantiques)

    Examples
    --------
    >>> annotations = load_annotations('data/annotations.csv')
    >>> print(annotations.head())
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f"⚠️  Fichier d'annotations non trouvé: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    print(f"✅ Annotations chargées: {len(df)} produits ancres")

    return df


# ==================== FONCTIONS DE VALIDATION ====================

def validate_dataset(df: pd.DataFrame,
                     min_products: int = MIN_PRODUCTS,
                     min_categories: int = MIN_CATEGORIES,
                     min_image_size: Tuple[int, int] = MIN_IMAGE_SIZE,
                     images_dir: str = None) -> Tuple[bool, List[str]]:
    """
    Valide que le dataset respecte toutes les exigences du projet.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame des produits à valider
    min_products : int, optional
        Nombre minimum de produits requis (default: 300)
    min_categories : int, optional
        Nombre minimum de catégories requises (default: 8)
    min_image_size : Tuple[int, int], optional
        Taille minimum des images (default: (224, 224))
    images_dir : str, optional
        Répertoire contenant les images. Si None, ne vérifie pas les fichiers

    Returns
    -------
    Tuple[bool, List[str]]
        - bool: True si le dataset est valide, False sinon
        - List[str]: Liste des erreurs/warnings rencontrés

    Examples
    --------
    >>> is_valid, errors = validate_dataset(df, images_dir='data/images')
    >>> if is_valid:
    ...     print("Dataset valide !")
    ... else:
    ...     for error in errors:
    ...         print(error)
    """
    errors = []
    warnings = []

    # Vérification 1: Nombre minimum de produits
    n_products = len(df)
    if n_products < min_products:
        errors.append(
            f"❌ Nombre de produits insuffisant: {n_products} < {min_products}"
        )
    else:
        print(f"✅ Nombre de produits: {n_products} (>= {min_products})")

    # Vérification 2: Nombre minimum de catégories
    n_categories = df['categorie'].nunique()
    if n_categories < min_categories:
        errors.append(
            f"❌ Nombre de catégories insuffisant: {n_categories} < {min_categories}"
        )
    else:
        print(f"✅ Nombre de catégories: {n_categories} (>= {min_categories})")

    # Vérification 3: Valeurs manquantes dans les colonnes obligatoires
    for col in REQUIRED_COLUMNS:
        missing = df[col].isna().sum()
        if missing > 0:
            errors.append(
                f"❌ Valeurs manquantes dans '{col}': {missing} produits"
            )

    # Vérification 4: Vérification des images (si images_dir fourni)
    if images_dir is not None:
        images_dir = Path(images_dir)
        missing_images = []
        invalid_images = []

        for idx, row in df.iterrows():
            img_path = images_dir / row['image_path']

            # Vérifier si le fichier existe
            if not img_path.exists():
                missing_images.append(row['id'])
                continue

            # Vérifier la taille de l'image
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    if width < min_image_size[0] or height < min_image_size[1]:
                        invalid_images.append(
                            f"{row['id']} ({width}x{height} < {min_image_size[0]}x{min_image_size[1]})"
                        )
            except Exception as e:
                invalid_images.append(f"{row['id']} (erreur: {str(e)})")

        if missing_images:
            errors.append(
                f"❌ Images manquantes: {len(missing_images)} produits"
            )
            if len(missing_images) <= 10:
                errors.append(f"   IDs concernés: {missing_images}")

        if invalid_images:
            warnings.append(
                f"⚠️  Images invalides/trop petites: {len(invalid_images)} produits"
            )
            if len(invalid_images) <= 5:
                warnings.append(f"   {invalid_images}")

    # Vérification 5: Doublons d'ID
    duplicate_ids = df[df.duplicated(subset=['id'], keep=False)]
    if len(duplicate_ids) > 0:
        errors.append(
            f"❌ IDs en double: {len(duplicate_ids)} produits"
        )

    # Vérification 6: Descriptions trop courtes
    desc_lengths = df['description'].str.len()
    short_descriptions = df[desc_lengths < 50]
    if len(short_descriptions) > 0:
        warnings.append(
            f"⚠️  Descriptions courtes (<50 caractères): {len(short_descriptions)} produits"
        )

    # Afficher les warnings
    for warning in warnings:
        print(warning)

    is_valid = len(errors) == 0

    return is_valid, errors


# ==================== FONCTIONS DE STATISTIQUES ====================

def get_statistics(df: pd.DataFrame) -> Dict:
    """
    Génère des statistiques descriptives sur le dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame des produits

    Returns
    -------
    Dict
        Dictionnaire contenant les statistiques:
        - n_products: Nombre total de produits
        - n_categories: Nombre de catégories
        - n_subcategories: Nombre de sous-catégories
        - categories_distribution: Distribution par catégorie
        - price_stats: Statistiques de prix
        - color_distribution: Distribution par couleur (si disponible)
        - material_distribution: Distribution par matière (si disponible)

    Examples
    --------
    >>> stats = get_statistics(df)
    >>> print(f"Nombre de produits: {stats['n_products']}")
    >>> print(stats['categories_distribution'])
    """
    stats = {
        'n_products': len(df),
        'n_categories': df['categorie'].nunique(),
        'n_subcategories': df['sous_categorie'].nunique(),
        'categories_distribution': df['categorie'].value_counts().to_dict(),
        'price_stats': {
            'mean': df['prix'].mean(),
            'median': df['prix'].median(),
            'min': df['prix'].min(),
            'max': df['prix'].max(),
            'std': df['prix'].std()
        }
    }

    # Ajouter la distribution des couleurs si disponible
    if 'couleur' in df.columns and df['couleur'].notna().any():
        stats['color_distribution'] = df['couleur'].value_counts().to_dict()

    # Ajouter la distribution des matières si disponible
    if 'matiere' in df.columns and df['matiere'].notna().any():
        stats['material_distribution'] = df['matiere'].value_counts().to_dict()

    return stats


def print_statistics(stats: Dict):
    """
    Affiche les statistiques du dataset de manière lisible.

    Parameters
    ----------
    stats : Dict
        Dictionnaire de statistiques retourné par get_statistics()

    Examples
    --------
    >>> stats = get_statistics(df)
    >>> print_statistics(stats)
    """
    print("\n" + "="*50)
    print("📊 STATISTIQUES DU DATASET")
    print("="*50)

    print(f"\n📦 Nombre de produits: {stats['n_products']}")
    print(f"📁 Nombre de catégories: {stats['n_categories']}")
    print(f"📂 Nombre de sous-catégories: {stats['n_subcategories']}")

    print("\n📈 Distribution par catégorie:")
    for cat, count in stats['categories_distribution'].items():
        bar = "█" * (count // max(1, stats['n_products'] // 30))
        print(f"   {cat:30s} {count:4d} {bar}")

    print("\n💰 Prix:")
    price_stats = stats['price_stats']
    print(f"   Moyenne: {price_stats['mean']:.2f} FCFA")
    print(f"   Médiane: {price_stats['median']:.2f} FCFA")
    print(f"   Min: {price_stats['min']:.2f} FCFA")
    print(f"   Max: {price_stats['max']:.2f} FCFA")

    if 'color_distribution' in stats:
        print("\n🎨 Distribution par couleur:")
        for color, count in list(stats['color_distribution'].items())[:10]:
            print(f"   {color:20s} {count:4d}")

    if 'material_distribution' in stats:
        print("\n🧵 Distribution par matière:")
        for material, count in list(stats['material_distribution'].items())[:10]:
            print(f"   {material:20s} {count:4d}")

    print("\n" + "="*50 + "\n")


# ==================== FONCTIONS DE VISUALISATION ====================

def plot_categories_distribution(df: pd.DataFrame, save_path: str = None):
    """
    Crée un histogramme de la distribution des catégories.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame des produits
    save_path : str, optional
        Chemin pour sauvegarder la figure. Si None, affiche seulement

    Examples
    --------
    >>> plot_categories_distribution(df, save_path='rapport/categories_histogram.png')
    """
    plt.figure(figsize=(12, 6))

    cat_counts = df['categorie'].value_counts().sort_values(ascending=True)

    plt.barh(cat_counts.index, cat_counts.values, color='steelblue')
    plt.xlabel('Nombre de produits')
    plt.ylabel('Catégorie')
    plt.title('Distribution des produits par catégorie')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Histogramme sauvegardé: {save_path}")

    plt.show()


def plot_price_distribution(df: pd.DataFrame, save_path: str = None):
    """
    Crée un histogramme de la distribution des prix.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame des produits
    save_path : str, optional
        Chemin pour sauvegarder la figure

    Examples
    --------
    >>> plot_price_distribution(df, save_path='rapport/price_distribution.png')
    """
    plt.figure(figsize=(10, 6))

    plt.hist(df['prix'], bins=30, color='coral', edgecolor='black')
    plt.xlabel('Prix (FCFA)')
    plt.ylabel('Nombre de produits')
    plt.title('Distribution des prix')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Distribution des prix sauvegardée: {save_path}")

    plt.show()


# ==================== FONCTIONS UTILITAIRES ====================

def get_sample_products(df: pd.DataFrame, n: int = 5, category: str = None) -> pd.DataFrame:
    """
    Retourne un échantillon aléatoire de produits.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame des produits
    n : int, optional
        Nombre de produits à retourner (default: 5)
    category : str, optional
        Filtrer par catégorie spécifique (default: None)

    Returns
    -------
    pd.DataFrame
        DataFrame contenant l'échantillon de produits

    Examples
    --------
    >>> sample = get_sample_products(df, n=3, category='Vêtements')
    >>> print(sample[['nom', 'categorie', 'prix']])
    """
    if category:
        df_filtered = df[df['categorie'] == category]
        return df_filtered.sample(n=min(n, len(df_filtered)))
    else:
        return df.sample(n=min(n, len(df)))


def get_product_by_id(df: pd.DataFrame, product_id: str) -> pd.Series:
    """
    Retourne un produit par son ID.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame des produits
    product_id : str
        ID du produit recherché

    Returns
    -------
    pd.Series
        Série contenant les données du produit, ou None si non trouvé

    Examples
    --------
    >>> product = get_product_by_id(df, 'PROD_001')
    >>> print(product['nom'])
    """
    result = df[df['id'] == product_id]

    if len(result) == 0:
        print(f"⚠️  Produit avec ID '{product_id}' non trouvé")
        return None

    return result.iloc[0]


def search_products(df: pd.DataFrame,
                   query: str,
                   search_in: List[str] = ['nom', 'description']) -> pd.DataFrame:
    """
    Recherche des produits par mot-clé.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame des produits
    query : str
        Mot-clé à rechercher
    search_in : List[str], optional
        Colonnes dans lesquelles chercher (default: ['nom', 'description'])

    Returns
    -------
    pd.DataFrame
        DataFrame filtré contenant les résultats de la recherche

    Examples
    --------
    >>> results = search_products(df, 'robe', search_in=['nom', 'description'])
    >>> print(results[['nom', 'description']])
    """
    query_lower = query.lower()

    mask = pd.Series([False] * len(df), index=df.index)

    for col in search_in:
        if col in df.columns:
            mask |= df[col].str.lower().str.contains(query_lower, na=False)

    results = df[mask]
    print(f"🔍 Recherche '{query}': {len(results)} résultats trouvés")

    return results


# ==================== MAIN ====================

if __name__ == "__main__":
    # Exemple d'utilisation
    print("Module de chargement des données produits")
    print("Importez ce module avec: from src import data_loader")
