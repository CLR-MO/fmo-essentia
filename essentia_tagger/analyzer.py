"""
Audio analysis with Essentia.

Each analyze_* function takes a file path and returns a dict of
attributes and a list of tags.  Results are later posted to CLRMO via
the REST API.
"""

import os
import warnings
from pathlib import Path
from typing import Any

# Suppress TensorFlow noise unless the user explicitly wants it.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

import numpy as np

# Lazy imports so non-TF analysis still works without essentia-tensorflow.
_es = None


def _essentia():
    global _es
    if _es is None:
        import essentia.standard as es
        _es = es
    return _es


# ─── Models ──────────────────────────────────────────────────────────────────

def _models_dir() -> Path:
    return Path(__file__).parent.parent / "models"


_model_cache: dict[str, Any] = {}


def _load_model(name: str, models_dir: Path | None = None) -> Any:
    """Load (and cache) a TensorflowPredictMusiCNN model by short name."""
    if name not in _model_cache:
        d = models_dir or _models_dir()
        pb = d / f"{name}-musicnn-msd-2.pb"
        if not pb.exists():
            raise FileNotFoundError(
                f"Model file not found: {pb}\n"
                "Run `clrmo-download-models` (or `python -m essentia_tagger.download_models`) first."
            )
        es = _essentia()
        _model_cache[name] = es.TensorflowPredictMusiCNN(
            graphFilename=str(pb),
            output="model/Sigmoid",
        )
    return _model_cache[name]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _load_audio_16k(path: str) -> np.ndarray:
    """Load audio resampled to 16 kHz mono (required by MusiCNN models)."""
    es = _essentia()
    return es.MonoLoader(filename=path, sampleRate=16000, resampleQuality=4)()


def _load_audio_44k(path: str) -> np.ndarray:
    """Load audio at 44.1 kHz mono (required by RhythmExtractor / KeyExtractor)."""
    es = _essentia()
    return es.MonoLoader(filename=path, sampleRate=44100)()


def _predict(model_name: str, audio_16k: np.ndarray, models_dir: Path | None = None) -> float:
    """Run a binary MusiCNN classifier; return the positive-class probability (0–1)."""
    model = _load_model(model_name, models_dir)
    activations = model(audio_16k)          # shape: (N_patches, 2)
    return float(np.mean(activations[:, 1]))  # avg over patches, positive class


# ─── Analysis functions ───────────────────────────────────────────────────────

# Mood labels and the threshold above which the tag is applied.
MOOD_MODELS = [
    "mood_happy",
    "mood_sad",
    "mood_relaxed",
    "mood_aggressive",
    "mood_party",
    "mood_acoustic",
    "mood_electronic",
    "danceability",
]

# Tag prefix used in CLRMO (e.g. "/ mood / happy")
TAG_PREFIX = "/ mood /"


def analyze_file(
    path: str,
    models_dir: Path | None = None,
    mood_threshold: float = 0.5,
    mood_thresholds: dict[str, float] | None = None,
    tag_prefix: str = TAG_PREFIX,
    run_bpm: bool = True,
    run_key: bool = True,
    run_mood: bool = True,
    single_pass: bool = False,
) -> dict[str, Any]:
    """
    Analyze one audio file.

    Returns:
        {
            "attributes": {
                "bpm": 120.3,
                "key": "C",
                "scale": "major",
                "mood_happy": 0.82,
                "mood_sad": 0.12,
                ...
            },
            "tags": ["/ mood / happy", "/ mood / acoustic"],
        }
    """
    attributes: dict[str, Any] = {}
    tags: list[str] = []

    # ── BPM + Key (44.1 kHz) ─────────────────────────────────────────────────
    audio_44k = None
    if run_bpm or run_key:
        audio_44k = _load_audio_44k(path)
        es = _essentia()

        if run_bpm:
            bpm, *_ = es.RhythmExtractor2013(method="multifeature")(audio_44k)
            attributes["bpm"] = round(float(bpm), 1)

        if run_key:
            key, scale, strength = es.KeyExtractor()(audio_44k)
            attributes["key"] = key
            attributes["scale"] = scale
            attributes["key_strength"] = round(float(strength), 3)

    # ── Mood / Danceability (16 kHz, TF models) ───────────────────────────────
    if run_mood:
        es = _essentia()
        if single_pass and audio_44k is not None:
            audio_16k = es.Resample(inputSampleRate=44100, outputSampleRate=16000)(audio_44k)
        else:
            audio_16k = _load_audio_16k(path)

        _thresholds = {"mood_electronic": 0.6, **(mood_thresholds or {})}
        for model_name in MOOD_MODELS:
            prob = _predict(model_name, audio_16k, models_dir)
            # Store raw probability as attribute
            attributes[model_name] = round(prob, 3)
            # Add tag if above this mood's threshold (per-mood override or global default)
            threshold = _thresholds.get(model_name, mood_threshold)
            if prob >= threshold:
                label = model_name.replace("mood_", "").replace("_", " ")
                tags.append(f"{tag_prefix} {label}".strip())

    return {"attributes": attributes, "tags": tags}
