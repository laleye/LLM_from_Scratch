"""
Embeddings Module

This module provides token and positional embedding implementations
for the Mini-GPT pedagogical notebook.
"""

from .token_embedding import (
    create_token_embedding_matrix,
    token_embedding_lookup,
    print_embedding_info,
    validate_embedding_dimensions,
)

from .positional_embedding import (
    create_sinusoidal_positional_encoding,
    get_positional_encoding,
    create_position_ids,
    print_positional_encoding_info,
    visualize_positional_encoding_pattern,
    validate_positional_encoding_dimensions,
    combine_token_and_positional_embeddings,
)

__all__ = [
    # Token embedding functions
    'create_token_embedding_matrix',
    'token_embedding_lookup',
    'print_embedding_info',
    'validate_embedding_dimensions',
    # Positional embedding functions
    'create_sinusoidal_positional_encoding',
    'get_positional_encoding',
    'create_position_ids',
    'print_positional_encoding_info',
    'visualize_positional_encoding_pattern',
    'validate_positional_encoding_dimensions',
    'combine_token_and_positional_embeddings',
]
