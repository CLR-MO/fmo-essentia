"""
Audio analysis with Essentia.

When run as a module (python3 -m essentia_tagger.analyzer), accepts a
JSON-encoded argument dict as the first positional argument, runs analyze_file,
and prints the JSON result to stdout.  Used by cli.py to isolate each file
analysis in a subprocess so that segfaults (e.g. from unsupported codecs) do
not kill the entire run.

Each analyze_* function takes a file path and returns a dict of
attributes and a list of tags.  Results are later posted to CLRMO via
the REST API.
"""

import contextlib
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


# ─── Config ───────────────────────────────────────────────────────────────────

def _load_models_config(config_path: Path | None = None) -> dict:
    from .download_models import load_models_config
    return load_models_config(config_path)


# ─── Model cache ──────────────────────────────────────────────────────────────

_model_cache: dict[str, Any] = {}


def _load_model(name: str, models_dir: Path | None, output_node: str = "model/Sigmoid",
                filename: str | None = None) -> Any:
    """Load (and cache) a TensorflowPredictMusiCNN model."""
    cache_key = f"{name}:{output_node}"
    if cache_key not in _model_cache:
        d = models_dir or _default_models_dir()
        pb = d / (filename or f"{name}-musicnn-msd-2.pb")
        if not pb.exists():
            raise FileNotFoundError(
                f"Model file not found: {pb}\n"
                "Run `clrmo-download-models` first."
            )
        es = _essentia()
        _model_cache[cache_key] = es.TensorflowPredictMusiCNN(
            graphFilename=str(pb),
            output=output_node,
        )
    return _model_cache[cache_key]


def _default_models_dir() -> Path:
    return Path(__file__).parent.parent / "models"


# ─── Helpers ─────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _suppress_c_stderr():
    """Redirect C-level fd 2 to /dev/null.

    Python's warnings filter and redirect_stderr only affect sys.stderr; they
    cannot silence warnings printed directly by C/C++ extensions (Essentia,
    TensorFlow).  Duplicating fd 2 around the call suppresses those too.
    """
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        os.close(devnull_fd)


def _load_audio_16k(path: str) -> np.ndarray:
    """Load audio resampled to 16 kHz mono (required by MusiCNN models)."""
    es = _essentia()
    return es.MonoLoader(filename=path, sampleRate=16000, resampleQuality=4)()


def _load_audio_44k(path: str) -> np.ndarray:
    """Load audio at 44.1 kHz mono (required by RhythmExtractor / KeyExtractor)."""
    es = _essentia()
    return es.MonoLoader(filename=path, sampleRate=44100)()


def _predict_binary(name: str, audio_16k: np.ndarray, models_dir: Path | None,
                    filename: str | None = None) -> float:
    """Run a binary MusiCNN classifier; return the positive-class probability (0–1)."""
    model = _load_model(name, models_dir, output_node="model/Sigmoid", filename=filename)
    # Suppress C-level stderr: Essentia prints "No network created..." once per
    # audio patch on the 2nd+ call of a cached model instance — harmless but very noisy.
    with _suppress_c_stderr():
        activations = model(audio_16k)      # shape: (N_patches, 2)
    return float(np.mean(activations[:, 1]))  # avg over patches, positive class


def _predict_multiclass(name: str, audio_16k: np.ndarray, models_dir: Path | None,
                        output_node: str = "model/Softmax",
                        filename: str | None = None) -> np.ndarray:
    """Run a multi-class MusiCNN classifier; return mean class probabilities."""
    model = _load_model(name, models_dir, output_node=output_node, filename=filename)
    with _suppress_c_stderr():
        activations = model(audio_16k)      # shape: (N_patches, N_classes)
    return np.mean(activations, axis=0)     # shape: (N_classes,)


# ─── Analysis ─────────────────────────────────────────────────────────────────

def analyze_file(
    path: str,
    models_dir: Path | None = None,
    config_path: Path | None = None,
    mood_threshold: float = 0.5,
    mood_thresholds: dict[str, float] | None = None,
    tag_prefix: str = "/ mood /",
    run_bpm: bool = True,
    run_key: bool = True,
    run_mood: bool = True,
    single_pass: bool = False,
) -> dict[str, Any]:
    """
    Analyze one audio file using models enabled in models.toml.

    Returns:
        {
            "attributes": {"bpm": 120.3, "key": "C", "scale": "major",
                           "mood_happy": 0.82, "voice_instrumental": 0.1, ...},
            "tags": ["/ mood / happy", "/ genre / rock"],
        }
    """
    attributes: dict[str, Any] = {}
    tags: list[str] = []

    # ── BPM + Key (44.1 kHz, algorithmic — no model files needed) ────────────
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

    # ── ML models (16 kHz) ────────────────────────────────────────────────────
    if not run_mood:
        return {"attributes": attributes, "tags": tags}

    es = _essentia()
    if single_pass and audio_44k is not None:
        audio_16k = es.Resample(inputSampleRate=44100, outputSampleRate=16000)(audio_44k)
    else:
        audio_16k = _load_audio_16k(path)

    models_cfg = _load_models_config(config_path)

    # Per-model threshold overrides (applied to binary models)
    _thresholds: dict[str, float] = {"mood_electronic": 0.6, **(mood_thresholds or {})}

    for name, spec in models_cfg.items():
        if not spec.get("enabled", True):
            continue

        model_type = spec.get("type", "binary")
        filename = Path(spec["path"]).name
        pb = (models_dir or _default_models_dir()) / filename
        if not pb.exists():
            continue  # model not downloaded — skip silently

        if model_type == "binary":
            prob = _predict_binary(name, audio_16k, models_dir, filename=filename)
            attributes[name] = round(prob, 3)
            threshold = _thresholds.get(name, mood_threshold)
            if prob >= threshold:
                prefix = spec.get("tag_prefix", tag_prefix)
                label = name.replace("mood_", "").replace("_", " ")
                tags.append(f"{prefix} {label}".strip())

        elif model_type == "multiclass":
            classes = spec.get("classes", [])
            if not classes:
                continue
            output_node = spec.get("output", "model/Softmax")
            probs = _predict_multiclass(name, audio_16k, models_dir, output_node, filename=filename)
            # Store top class as attribute
            top_idx = int(np.argmax(probs))
            attributes[name] = classes[top_idx] if top_idx < len(classes) else str(top_idx)
            # For danceability, also store the probability weights
            if name == "danceability" and len(classes) == 2:
                # classes are ["danceable", "not_danceable"]
                # probs[0] = danceable probability, probs[1] = not_danceable probability
                attributes["danceability_weight"] = round(float(probs[0]), 3)
                attributes["not_danceability_weight"] = round(float(probs[1]), 3)
            prefix = spec.get("tag_prefix", "/ genre /")
            threshold = _thresholds.get(name, mood_threshold)
            if len(classes) == 2:
                # Binary-like model (Sigmoid per class): both outputs can exceed the
                # threshold simultaneously, so only tag the winning class.
                tags.append(f"{prefix} {classes[top_idx]}".strip())
            else:
                # Multi-class: tag any class above threshold (e.g. multiple genres)
                for idx, prob in enumerate(probs):
                    if prob >= threshold and idx < len(classes):
                        tags.append(f"{prefix} {classes[idx]}".strip())

    return {"attributes": attributes, "tags": tags}


if __name__ == "__main__":
    import json
    import sys
    args = json.loads(sys.argv[1])
    if args.get("models_dir"):
        args["models_dir"] = Path(args["models_dir"])
    result = analyze_file(**args)
    print(json.dumps(result))
