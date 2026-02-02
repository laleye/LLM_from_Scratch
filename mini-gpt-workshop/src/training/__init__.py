"""Training module for Mini-GPT.

This module provides:
- Configuration dataclasses for corpus, training, and model
- Dataset loading and preprocessing utilities
- Training loop implementation
"""

from .config import CorpusConfig, TrainingConfig, ModelConfig

__all__ = ['CorpusConfig', 'TrainingConfig', 'ModelConfig']
