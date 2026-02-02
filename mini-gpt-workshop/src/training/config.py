"""
Configuration dataclasses for training and corpus management.

This module defines configuration objects for:
- Corpus loading and preprocessing
- Training hyperparameters
- Model architecture parameters
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CorpusConfig:
    """Configuration for corpus loading and preprocessing.
    
    Attributes:
        source: Path or URL to French corpus
        dataset_name: Name of dataset (wikipedia-fr, gutenberg-fr, or oscar-fr)
        max_lines: Maximum number of lines to load (for 4-hour session)
        min_length: Minimum text length to include
        max_length: Maximum text length per sample
        encoding: Text encoding format
        remove_punctuation: Whether to remove punctuation
        lowercase: Whether to convert to lowercase
        remove_numbers: Whether to remove numeric characters
    """
    source: str
    dataset_name: str = "wikipedia-fr"
    max_lines: int = 10000
    min_length: int = 10
    max_length: int = 500
    encoding: str = "utf-8"
    
    # Preprocessing options
    remove_punctuation: bool = False
    lowercase: bool = False
    remove_numbers: bool = False


@dataclass
class TrainingConfig:
    """Configuration for model training.
    
    Attributes:
        batch_size: Number of samples per batch
        seq_len: Maximum sequence length
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        eval_interval: Steps between evaluation
        device: Device to use for training (cuda/cpu)
    """
    batch_size: int = 32
    seq_len: int = 128
    learning_rate: float = 3e-4
    num_epochs: int = 5
    eval_interval: int = 100
    device: str = "cuda"  # Will be set based on availability
    
    def __post_init__(self):
        """Set device based on availability if not explicitly set."""
        import torch
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            print("CUDA not available, using CPU")


@dataclass
class ModelConfig:
    """Configuration for Mini-GPT model architecture.
    
    Attributes:
        vocab_size: Size of vocabulary (determined by tokenizer)
        d_model: Dimension of model embeddings
        num_heads: Number of attention heads
        num_layers: Number of transformer blocks
        d_ff: Dimension of feed-forward network
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
    """
    vocab_size: int
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 4
    d_ff: int = 1024
    max_seq_len: int = 128
    dropout: float = 0.1
    
    def __post_init__(self):
        """Validate configuration."""
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
