"""
Audio Preprocessing Module — Spectrogrammes et Log-Mel

Ce module implémente les transformations fondamentales du signal audio :
waveform brute → spectrogramme STFT → spectrogramme Log-Mel.

Deux implémentations sont fournies :
1. From Scratch (NumPy) — pour comprendre chaque étape mathématique
2. Librosa — pour le code professionnel et efficace

Formules clés :
    STFT: X(t,f) = Σ x(n) * w(n-t*H) * e^{-j2πfn/N}
    Mel:  m = 2595 * log10(1 + f/700)
    Log-Mel: log(max(ε, Mel_filterbank @ |STFT|²))
"""

import numpy as np
from typing import Tuple, Optional

try:
    import librosa
    import librosa.display
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================
# IMPLEMENTATION 1: From Scratch (NumPy)
# ============================================
# Objectif: Comprendre le formalisme mathématique


def compute_stft_from_scratch(
    waveform: np.ndarray,
    n_fft: int = 400,
    hop_length: int = 160
) -> np.ndarray:
    """Calcule la STFT manuellement avec fenêtrage de Hann et FFT numpy.

    Étapes explicites :
    1. Découpage en trames avec fenêtre glissante (hop_length)
    2. Application de la fenêtre de Hann à chaque trame
    3. FFT sur chaque trame
    4. Calcul du module (magnitude)

    Args:
        waveform: Signal audio 1D, shape (num_samples,)
        n_fft: Taille de la FFT (largeur de la fenêtre d'analyse)
        hop_length: Pas de la fenêtre glissante (en échantillons)

    Returns:
        Spectrogramme de magnitude, shape (n_fft // 2 + 1, n_frames)
    """
    # --- Étape 0 : Validation et padding ---
    # On s'assure que le signal est assez long pour au moins une trame
    if len(waveform) < n_fft:
        waveform = np.pad(waveform, (0, n_fft - len(waveform)), mode='constant')
    # Shape: waveform (num_samples,)

    # --- Étape 1 : Calcul du nombre de trames ---
    # n_frames = (num_samples - n_fft) // hop_length + 1
    n_frames = (len(waveform) - n_fft) // hop_length + 1
    # print(f"  [STFT] num_samples={len(waveform)}, n_fft={n_fft}, hop={hop_length} → n_frames={n_frames}")

    # --- Étape 2 : Création de la fenêtre de Hann ---
    # w(n) = 0.5 * (1 - cos(2πn / (N-1)))
    # La fenêtre de Hann réduit les fuites spectrales (spectral leakage)
    hann_window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n_fft) / (n_fft - 1)))
    # Shape: hann_window (n_fft,)

    # --- Étape 3 : Découpage en trames + fenêtrage + FFT ---
    n_freq = n_fft // 2 + 1
    spectrogram = np.zeros((n_freq, n_frames))
    # Shape: spectrogram (n_freq, n_frames)

    for t in range(n_frames):
        # Extraire la trame t
        start = t * hop_length
        frame = waveform[start:start + n_fft]
        # Shape: frame (n_fft,)

        # Appliquer la fenêtre de Hann
        windowed_frame = frame * hann_window
        # Shape: windowed_frame (n_fft,)

        # FFT (on ne garde que les fréquences positives : 0 à n_fft//2)
        fft_result = np.fft.rfft(windowed_frame)
        # Shape: fft_result (n_fft // 2 + 1,) — complexe

        # Magnitude : |X(f)| = sqrt(Re² + Im²)
        spectrogram[:, t] = np.abs(fft_result)
        # Shape: spectrogram[:, t] (n_freq,)

    # print(f"  [STFT] Output shape: ({n_freq}, {n_frames})")
    return spectrogram


def compute_mel_filterbank_from_scratch(
    sr: int,
    n_fft: int,
    n_mels: int
) -> np.ndarray:
    """Construit manuellement les filtres triangulaires Mel.

    Étapes explicites :
    1. Conversion Hz → Mel : m = 2595 * log10(1 + f/700)
    2. Espacement linéaire en échelle Mel (n_mels + 2 points)
    3. Conversion Mel → Hz : f = 700 * (10^(m/2595) - 1)
    4. Construction des filtres triangulaires

    Args:
        sr: Fréquence d'échantillonnage (Hz)
        n_fft: Taille de la FFT
        n_mels: Nombre de filtres Mel

    Returns:
        Matrice de filtres Mel, shape (n_mels, n_fft // 2 + 1)
    """
    n_freq = n_fft // 2 + 1

    # --- Étape 1 : Bornes fréquentielles ---
    f_min = 0.0
    f_max = sr / 2.0  # Fréquence de Nyquist

    # --- Étape 2 : Conversion Hz → Mel ---
    # Formule : m = 2595 * log10(1 + f / 700)
    mel_min = 2595.0 * np.log10(1.0 + f_min / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + f_max / 700.0)

    # --- Étape 3 : Espacement linéaire en Mel ---
    # n_mels + 2 points pour définir n_mels filtres triangulaires
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    # Shape: mel_points (n_mels + 2,)

    # --- Étape 4 : Conversion Mel → Hz ---
    # Formule inverse : f = 700 * (10^(m/2595) - 1)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    # Shape: hz_points (n_mels + 2,)

    # --- Étape 5 : Conversion Hz → indices de bins FFT ---
    # bin = floor((n_fft + 1) * f / sr)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    # Shape: bin_points (n_mels + 2,)

    # --- Étape 6 : Construction des filtres triangulaires ---
    filterbank = np.zeros((n_mels, n_freq))
    # Shape: filterbank (n_mels, n_freq)

    for m in range(n_mels):
        f_left = bin_points[m]       # Début du triangle
        f_center = bin_points[m + 1] # Sommet du triangle
        f_right = bin_points[m + 2]  # Fin du triangle

        # Pente montante : de f_left à f_center
        for k in range(f_left, f_center):
            if k < n_freq and f_center != f_left:
                filterbank[m, k] = (k - f_left) / (f_center - f_left)

        # Pente descendante : de f_center à f_right
        for k in range(f_center, f_right):
            if k < n_freq and f_right != f_center:
                filterbank[m, k] = (f_right - k) / (f_right - f_center)

    # print(f"  [Mel Filterbank] Shape: ({n_mels}, {n_freq})")
    return filterbank


def compute_log_mel_from_scratch(
    waveform: np.ndarray,
    sr: int = 16000,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160
) -> np.ndarray:
    """Calcule le spectrogramme Log-Mel from scratch.

    Pipeline complet : STFT → |STFT|² → Mel filterbank → log()

    Args:
        waveform: Signal audio 1D, shape (num_samples,)
        sr: Fréquence d'échantillonnage (Hz)
        n_mels: Nombre de filtres Mel
        n_fft: Taille de la FFT
        hop_length: Pas de la fenêtre glissante

    Returns:
        Spectrogramme Log-Mel, shape (n_mels, n_frames)
    """
    # --- Étape 1 : STFT → magnitude ---
    spectrogram = compute_stft_from_scratch(waveform, n_fft=n_fft, hop_length=hop_length)
    # Shape: spectrogram (n_freq, n_frames) avec n_freq = n_fft // 2 + 1

    # --- Étape 2 : Spectrogramme de puissance |X|² ---
    power_spectrogram = spectrogram ** 2
    # Shape: power_spectrogram (n_freq, n_frames)

    # --- Étape 3 : Application des filtres Mel ---
    mel_filterbank = compute_mel_filterbank_from_scratch(sr, n_fft, n_mels)
    # Shape: mel_filterbank (n_mels, n_freq)

    mel_spectrogram = mel_filterbank @ power_spectrogram
    # Shape: mel_spectrogram (n_mels, n_frames)

    # --- Étape 4 : Échelle logarithmique ---
    # log(max(ε, mel)) pour éviter log(0)
    eps = 1e-10
    log_mel = np.log(np.maximum(eps, mel_spectrogram))
    # Shape: log_mel (n_mels, n_frames)

    return log_mel


# ============================================
# IMPLEMENTATION 2: Librosa
# ============================================
# Objectif: Code professionnel et efficace


def load_audio(
    file_path: str,
    target_sr: int = 16000
) -> Tuple[np.ndarray, int]:
    """Charge un fichier audio et le rééchantillonne via librosa.

    Args:
        file_path: Chemin vers le fichier audio (.wav, .mp3, .flac, etc.)
        target_sr: Fréquence d'échantillonnage cible (Hz)

    Returns:
        Tuple (waveform, sample_rate)
            waveform: Signal audio 1D, shape (num_samples,)
            sample_rate: Fréquence d'échantillonnage effective

    Raises:
        FileNotFoundError: Si le fichier n'existe pas
        RuntimeError: Si le rééchantillonnage échoue
    """
    if not HAS_LIBROSA:
        raise ImportError("librosa is required for load_audio. Install with: pip install librosa")

    import os
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        waveform, sr = librosa.load(file_path, sr=target_sr)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file: {e}")

    return waveform, sr


def compute_spectrogram(
    waveform: np.ndarray,
    n_fft: int = 400,
    hop_length: int = 160
) -> np.ndarray:
    """Calcule le spectrogramme STFT via librosa.

    Args:
        waveform: Signal audio 1D, shape (num_samples,)
        n_fft: Taille de la FFT
        hop_length: Pas de la fenêtre glissante

    Returns:
        Spectrogramme de magnitude, shape (n_fft // 2 + 1, n_frames)
    """
    if not HAS_LIBROSA:
        raise ImportError("librosa is required. Install with: pip install librosa")

    stft = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(stft)
    return spectrogram


def compute_log_mel(
    waveform: np.ndarray,
    sr: int = 16000,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160
) -> np.ndarray:
    """Calcule le spectrogramme Log-Mel via librosa.

    Args:
        waveform: Signal audio 1D, shape (num_samples,)
        sr: Fréquence d'échantillonnage (Hz)
        n_mels: Nombre de filtres Mel
        n_fft: Taille de la FFT
        hop_length: Pas de la fenêtre glissante

    Returns:
        Spectrogramme Log-Mel, shape (n_mels, n_frames)
    """
    if not HAS_LIBROSA:
        raise ImportError("librosa is required. Install with: pip install librosa")

    mel_spec = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    log_mel = np.log(np.maximum(1e-10, mel_spec))
    return log_mel


def plot_audio_representations(
    waveform: np.ndarray,
    sr: int,
    spectrogram: np.ndarray,
    log_mel: np.ndarray
) -> None:
    """Affiche côte à côte waveform, spectrogramme et Log-Mel.

    Args:
        waveform: Signal audio 1D, shape (num_samples,)
        sr: Fréquence d'échantillonnage (Hz)
        spectrogram: Spectrogramme de magnitude, shape (n_freq, n_frames)
        log_mel: Spectrogramme Log-Mel, shape (n_mels, n_frames)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # 1. Waveform
    time_axis = np.arange(len(waveform)) / sr
    axes[0].plot(time_axis, waveform, linewidth=0.5)
    axes[0].set_title("Forme d'onde (Waveform)")
    axes[0].set_xlabel("Temps (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    # 2. Spectrogramme
    axes[1].imshow(
        20 * np.log10(np.maximum(1e-10, spectrogram)),
        aspect='auto', origin='lower', cmap='viridis'
    )
    axes[1].set_title("Spectrogramme (dB)")
    axes[1].set_xlabel("Trame")
    axes[1].set_ylabel("Bin fréquentiel")

    # 3. Log-Mel
    axes[2].imshow(log_mel, aspect='auto', origin='lower', cmap='viridis')
    axes[2].set_title("Spectrogramme Log-Mel")
    axes[2].set_xlabel("Trame")
    axes[2].set_ylabel("Filtre Mel")

    plt.tight_layout()
    plt.show()
