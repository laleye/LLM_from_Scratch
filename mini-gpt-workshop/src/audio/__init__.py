"""Audio module - Audio preprocessing, wav2vec 2.0 components, and speech recognition utilities.

Covers:
- Audio preprocessing (waveform, spectrogram, Log-Mel)
- Feature encoding (CNN 1D multi-layer encoder)
- Contrastive learning (cosine similarity, contrastive loss, diversity loss)
- Vector quantization (Gumbel-Softmax product quantization)
- Bottleneck adapters for multilingual transfer learning
- Audio feature serialization/deserialization (JSON round-trip)
- ASR evaluation metrics (WER, CER)
"""

from .preprocessing import (
    compute_stft_from_scratch,
    compute_mel_filterbank_from_scratch,
    compute_log_mel_from_scratch,
    compute_spectrogram,
    compute_log_mel,
    plot_audio_representations,
)
from .serialization import serialize_features, deserialize_features
from .feature_encoder import (
    conv1d_from_scratch,
    feature_encoder_from_scratch,
    FeatureEncoder,
)
from .contrastive_loss import (
    cosine_similarity_from_scratch,
    contrastive_loss_from_scratch,
    cosine_similarity,
    sample_negatives,
    contrastive_loss,
    diversity_loss,
)
from .quantization import (
    quantize_from_scratch,
    GumbelVectorQuantizer,
)
from .metrics import (
    levenshtein_distance_from_scratch,
    compute_wer_from_scratch,
    compute_cer_from_scratch,
    compute_wer,
    compute_cer,
)
from .adapter import (
    adapter_from_scratch,
    BottleneckAdapter,
    AdaptedLayer,
    insert_adapters,
)

__all__ = [
    # Preprocessing — from scratch
    "compute_stft_from_scratch",
    "compute_mel_filterbank_from_scratch",
    "compute_log_mel_from_scratch",
    # Preprocessing — librosa
    "compute_spectrogram",
    "compute_log_mel",
    "plot_audio_representations",
    # Serialization
    "serialize_features",
    "deserialize_features",
    # Feature Encoder
    "conv1d_from_scratch",
    "feature_encoder_from_scratch",
    "FeatureEncoder",
    # Contrastive Loss
    "cosine_similarity_from_scratch",
    "contrastive_loss_from_scratch",
    "cosine_similarity",
    "sample_negatives",
    "contrastive_loss",
    "diversity_loss",
    # Quantization
    "quantize_from_scratch",
    "GumbelVectorQuantizer",
    # Metrics
    "levenshtein_distance_from_scratch",
    "compute_wer_from_scratch",
    "compute_cer_from_scratch",
    "compute_wer",
    "compute_cer",
    # Adapter
    "adapter_from_scratch",
    "BottleneckAdapter",
    "AdaptedLayer",
    "insert_adapters",
]
