"""
Dataset loading and preprocessing utilities for French corpus.

This module provides functions to:
- Download and load French datasets (Wikipedia-fr, Gutenberg, OSCAR)
- Preprocess text data
- Create training batches
- Compute dataset statistics
"""


import os
import re
from typing import List, Tuple, Dict, Optional
from collections import Counter

from .config import CorpusConfig


def download_wikipedia_fr(output_path: str, max_lines: int = 10000) -> None:
    """
    Download a subset of French text corpus.
    
    For the 4-hour workshop, we use OSCAR (Open Super-large Crawled ALMAnaCH coRpus)
    which is available without loading scripts.
    
    Args:
        output_path: Path to save the downloaded corpus
        max_lines: Maximum number of lines to download
    """
    try:
        from datasets import load_dataset
        
        print(f"Downloading French corpus from OSCAR (max {max_lines} samples)...")
        
        # Load OSCAR French corpus from HuggingFace datasets
        dataset = load_dataset(
            "oscar",
            "unshuffled_deduplicated_fr",
            split=f"train[:{max_lines}]",
            streaming=True  # Use streaming to avoid loading everything into memory
        )
        
        # Extract text content
        texts = []
        for item in dataset:
            text = item['text'].strip()
            if len(text) > 50:  # Filter out very short texts
                texts.append(text)
                if len(texts) >= max_lines:
                    break
        
        # Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for text in texts:
                # Clean and write each text
                cleaned = text.replace('\n\n', '\n').strip()
                if cleaned:
                    f.write(cleaned + '\n')
        
        print(f"✓ Downloaded {len(texts)} texts to {output_path}")
        
    except Exception as e:
        print(f"Warning: Failed to download from OSCAR: {e}")
        print("Creating sample French corpus instead...")
        _create_sample_corpus(output_path)


def _create_sample_corpus(output_path: str) -> None:
    """
    Create a small sample French corpus for testing.
    
    This is a fallback when the datasets library is not available.
    """
    sample_texts = [
        "Le chat dort sur le canapé.",
        "La tour Eiffel est un monument emblématique de Paris.",
        "Les étudiants apprennent à construire des modèles de langage.",
        "L'intelligence artificielle transforme notre société.",
        "Le soleil brille dans le ciel bleu.",
        "Les oiseaux chantent dans les arbres du jardin.",
        "La France est connue pour sa gastronomie et sa culture.",
        "Les transformers ont révolutionné le traitement du langage naturel.",
        "Le chat qui était sur le tapis a mangé la souris.",
        "Les réseaux de neurones profonds apprennent des représentations complexes.",
    ] * 100  # Repeat to create more data
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in sample_texts:
            f.write(text + '\n')
    
    print(f"✓ Created sample corpus with {len(sample_texts)} lines at {output_path}")


def load_corpus(config: CorpusConfig) -> List[str]:
    """
    Load corpus from file or download if necessary.
    
    Args:
        config: CorpusConfig with source path and preprocessing options
        
    Returns:
        List of text strings
    """
    # Check if file exists
    if not os.path.exists(config.source):
        print(f"Corpus not found at {config.source}")
        
        # Try to download based on dataset name
        if config.dataset_name == "wikipedia-fr":
            download_wikipedia_fr(config.source, config.max_lines)
        else:
            print(f"Unknown dataset: {config.dataset_name}")
            print("Creating sample corpus...")
            _create_sample_corpus(config.source)
    
    # Load from file
    print(f"Loading corpus from {config.source}...")
    texts = []
    
    with open(config.source, 'r', encoding=config.encoding) as f:
        for i, line in enumerate(f):
            if i >= config.max_lines:
                break
            
            text = line.strip()
            
            # Apply length filters
            if len(text) < config.min_length or len(text) > config.max_length:
                continue
            
            # Apply preprocessing
            text = preprocess_text(text, config)
            
            if text:  # Only add non-empty texts
                texts.append(text)
    
    print(f"✓ Loaded {len(texts)} texts from corpus")
    return texts


def preprocess_text(text: str, config: CorpusConfig) -> str:
    """
    Preprocess text according to configuration.
    
    Args:
        text: Input text string
        config: CorpusConfig with preprocessing options
        
    Returns:
        Preprocessed text string
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Optional: lowercase
    if config.lowercase:
        text = text.lower()
    
    # Optional: remove numbers
    if config.remove_numbers:
        text = re.sub(r'\d+', '', text)
    
    # Optional: remove punctuation
    if config.remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)
    
    return text.strip()


def compute_corpus_statistics(texts: List[str]) -> Dict[str, any]:
    """
    Compute statistics about the corpus.
    
    Args:
        texts: List of text strings
        
    Returns:
        Dictionary with statistics:
        - num_texts: Number of texts
        - total_chars: Total character count
        - avg_length: Average text length
        - min_length: Minimum text length
        - max_length: Maximum text length
        - vocab_size: Number of unique characters
        - word_count: Total word count
        - unique_words: Number of unique words
        - char_freq: Character frequency distribution
        - word_freq: Word frequency distribution (top 100)
    """
    if not texts:
        return {}
    
    # Basic statistics
    lengths = [len(text) for text in texts]
    total_chars = sum(lengths)
    
    # Character-level statistics
    all_chars = ''.join(texts)
    char_freq = Counter(all_chars)
    
    # Word-level statistics (ignore stopwords)
    from stopwords import get_stopwords
    stopwords = set(get_stopwords("fr"))
    all_words = []
    for text in texts:
        words = text.split()
        filtered_words = [w for w in words]
        all_words.extend(filtered_words)
    word_freq = Counter(all_words)
    stats = {
        'num_texts': len(texts),
        'total_chars': total_chars,
        'avg_length': total_chars / len(texts),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'vocab_size': len(char_freq),
        'word_count': len(all_words),
        'unique_words': len(word_freq),
        'char_freq': dict(char_freq.most_common(50)),
        'word_freq': dict(word_freq.most_common(100)),
    }
    return stats


def visualize_corpus_statistics(stats: Dict[str, any]) -> None:
    """
    Visualize corpus statistics with matplotlib.
    
    Args:
        stats: Statistics dictionary from compute_corpus_statistics
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Corpus Statistics', fontsize=16, fontweight='bold')
        
        # 1. Basic statistics (text)
        ax1 = axes[0, 0]
        ax1.axis('off')
        stats_text = f"""
        Nombre de textes: {stats['num_texts']:,}
        Caractères totaux: {stats['total_chars']:,}
        Longueur moyenne: {stats['avg_length']:.1f}
        Longueur min: {stats['min_length']}
        Longueur max: {stats['max_length']}
        
        Taille du vocabulaire: {stats['vocab_size']}
        Nombre de mots: {stats['word_count']:,}
        Mots uniques: {stats['unique_words']:,}
        """
        ax1.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                family='monospace')
        ax1.set_title('Statistiques Générales', fontweight='bold')
        
        # 2. Character frequency (top 20)
        ax2 = axes[0, 1]
        char_items = list(stats['char_freq'].items())[:20]
        chars, freqs = zip(*char_items)
        # Replace space with visible character
        chars = [c if c != ' ' else '␣' for c in chars]
        ax2.bar(range(len(chars)), freqs, color='steelblue')
        ax2.set_xticks(range(len(chars)))
        ax2.set_xticklabels(chars, fontsize=10)
        ax2.set_xlabel('Caractères')
        ax2.set_ylabel('Fréquence')
        ax2.set_title('Fréquence des Caractères (Top 20)', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Word frequency (top 20)
        ax3 = axes[1, 0]
        word_items = list(stats['word_freq'].items())[:20]
        words, word_freqs = zip(*word_items)
        y_pos = np.arange(len(words))
        ax3.barh(y_pos, word_freqs, color='coral')
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(words, fontsize=9)
        ax3.invert_yaxis()
        ax3.set_xlabel('Fréquence')
        ax3.set_title('Fréquence des Mots (Top 20)', fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
        
        # 4. Word frequency distribution (log scale)
        ax4 = axes[1, 1]
        word_freq_values = sorted(stats['word_freq'].values(), reverse=True)
        ax4.plot(range(1, len(word_freq_values) + 1), word_freq_values, 
                color='green', linewidth=2)
        ax4.set_xlabel('Rang du mot')
        ax4.set_ylabel('Fréquence')
        ax4.set_title('Distribution de Fréquence (Loi de Zipf)', fontweight='bold')
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Warning: matplotlib not installed. Cannot visualize statistics.")
        print("Install with: pip install matplotlib")


def create_training_batches(
    texts: List[str],
    tokenizer,
    seq_len: int,
    batch_size: int,
    train_split: float = 0.9
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Create training and validation batches from texts.
    
    Args:
        texts: List of text strings
        tokenizer: Tokenizer with encode() method
        seq_len: Maximum sequence length
        batch_size: Batch size
        train_split: Fraction of data for training
        
    Returns:
        Tuple of (train_batches, val_batches)
        Each batch is a list of token ID sequences
    """
    # Tokenize all texts
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
    
    print(f"Total tokens: {len(all_tokens):,}")
    
    # Split into train and validation
    split_idx = int(len(all_tokens) * train_split)
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]
    
    # Create sequences of length seq_len
    def create_sequences(tokens, seq_len):
        sequences = []
        for i in range(0, len(tokens) - seq_len, seq_len):
            seq = tokens[i:i + seq_len]
            if len(seq) == seq_len:
                sequences.append(seq)
        return sequences
    
    train_sequences = create_sequences(train_tokens, seq_len)
    val_sequences = create_sequences(val_tokens, seq_len)
    
    print(f"Training sequences: {len(train_sequences):,}")
    print(f"Validation sequences: {len(val_sequences):,}")
    
    # Create batches
    def create_batches(sequences, batch_size):
        batches = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            if len(batch) == batch_size:
                batches.append(batch)
        return batches
    
    train_batches = create_batches(train_sequences, batch_size)
    val_batches = create_batches(val_sequences, batch_size)
    
    print(f"Training batches: {len(train_batches)}")
    print(f"Validation batches: {len(val_batches)}")
    
    return train_batches, val_batches


def save_corpus(texts: List[str], output_path: str) -> None:
    """
    Save corpus to file.
    
    Args:
        texts: List of text strings
        output_path: Path to save file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')
    print(f"✓ Saved {len(texts)} texts to {output_path}")


def load_corpus_from_file(file_path: str, max_lines: Optional[int] = None) -> List[str]:
    """
    Load corpus from a text file.
    
    Args:
        file_path: Path to corpus file
        max_lines: Maximum number of lines to load
        
    Returns:
        List of text strings
    """
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            text = line.strip()
            if text:
                texts.append(text)
    return texts
