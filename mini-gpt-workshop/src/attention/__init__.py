"""Attention module - Scaled dot-product, multi-head attention, and causal masking."""

from .scaled_dot_product import (
    softmax_from_scratch,
    scaled_dot_product_attention_from_scratch,
    example_attention_from_scratch,
    ScaledDotProductAttention,
    example_attention_pytorch,
    example_attention_with_causal_mask_pytorch
)

from .masking import (
    create_causal_mask_from_scratch,
    create_causal_mask_manual,
    create_causal_mask,
    visualize_causal_mask,
    compare_attention_patterns,
    visualize_mask_effect_on_attention,
    example_causal_mask_numpy,
    example_causal_mask_pytorch
)

from .multi_head import (
    multi_head_attention_from_scratch,
    example_multi_head_attention_from_scratch,
    example_multi_head_with_causal_mask
)

__all__ = [
    # Scaled dot-product attention
    'softmax_from_scratch',
    'scaled_dot_product_attention_from_scratch',
    'example_attention_from_scratch',
    'ScaledDotProductAttention',
    'example_attention_pytorch',
    'example_attention_with_causal_mask_pytorch',
    # Causal masking
    'create_causal_mask_from_scratch',
    'create_causal_mask_manual',
    'create_causal_mask',
    'visualize_causal_mask',
    'compare_attention_patterns',
    'visualize_mask_effect_on_attention',
    'example_causal_mask_numpy',
    'example_causal_mask_pytorch',
    # Multi-head attention
    'multi_head_attention_from_scratch',
    'example_multi_head_attention_from_scratch',
    'example_multi_head_with_causal_mask'
]
