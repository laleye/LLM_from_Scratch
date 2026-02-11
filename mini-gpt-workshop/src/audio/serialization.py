"""
Audio Feature Serialization Module — JSON Round-Trip

Ce module fournit des fonctions pour sérialiser et désérialiser
des dictionnaires de features audio (contenant des numpy arrays)
vers/depuis des chaînes JSON formatées.

Le round-trip serialize → deserialize doit produire un dictionnaire
équivalent (arrays égaux à tolérance flottante près, métadonnées identiques).
"""

import json
import numpy as np
from typing import Any


# Clés dont les valeurs sont des numpy arrays à convertir
_ARRAY_KEYS = {"waveform", "spectrogram", "log_mel"}


class _NumpyEncoder(json.JSONEncoder):
    """Encodeur JSON personnalisé gérant les types numpy."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)


def serialize_features(features: dict) -> str:
    """Sérialise un dictionnaire de features audio en JSON formaté.

    Gère les numpy arrays en les convertissant en listes Python.
    Les métadonnées scalaires (int, float, str) sont conservées telles quelles.

    Args:
        features: Dictionnaire contenant des numpy arrays et des métadonnées.
            Exemple de structure attendue :
            {
                "waveform": np.ndarray,
                "sample_rate": int,
                "spectrogram": np.ndarray,
                "log_mel": np.ndarray,
                "metadata": { ... }
            }

    Returns:
        Chaîne JSON formatée (indentée).

    Raises:
        TypeError: Si le dictionnaire contient des types non sérialisables
            après conversion numpy.
    """
    return json.dumps(features, cls=_NumpyEncoder, indent=2)


def deserialize_features(json_str: str) -> dict:
    """Désérialise une chaîne JSON en dictionnaire de features audio.

    Reconvertit les listes en numpy arrays pour les clés connues
    (waveform, spectrogram, log_mel). Les métadonnées restent en types Python natifs.

    Args:
        json_str: Chaîne JSON produite par serialize_features.

    Returns:
        Dictionnaire de features audio avec numpy arrays.

    Raises:
        json.JSONDecodeError: Si la chaîne JSON est malformée.
        KeyError: Si des champs requis sont absents.
    """
    data = json.loads(json_str)

    # Reconvertir les listes en numpy arrays pour les clés connues
    for key in _ARRAY_KEYS:
        if key in data and isinstance(data[key], list):
            data[key] = np.array(data[key])

    # Traiter récursivement les sous-dictionnaires (ex: metadata)
    # Les valeurs numériques dans metadata restent en types Python natifs
    return data
