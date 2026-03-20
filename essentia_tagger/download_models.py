"""Download Essentia TensorFlow models for music analysis."""

import os
import sys
import urllib.request
from pathlib import Path

# Models are MusiCNN classifiers trained on the Million Song Dataset.
# Each outputs [prob_negative, prob_positive] for its label.
MODELS = {
    # ── Mood ──────────────────────────────────────────────────────────────────
    "mood_happy":       "classifiers/mood_happy/mood_happy-musicnn-msd-2.pb",
    "mood_sad":         "classifiers/mood_sad/mood_sad-musicnn-msd-2.pb",
    "mood_relaxed":     "classifiers/mood_relaxed/mood_relaxed-musicnn-msd-2.pb",
    "mood_aggressive":  "classifiers/mood_aggressive/mood_aggressive-musicnn-msd-2.pb",
    "mood_party":       "classifiers/mood_party/mood_party-musicnn-msd-2.pb",
    "mood_acoustic":    "classifiers/mood_acoustic/mood_acoustic-musicnn-msd-2.pb",
    "mood_electronic":  "classifiers/mood_electronic/mood_electronic-musicnn-msd-2.pb",
    # ── Danceability ──────────────────────────────────────────────────────────
    "danceability":     "classifiers/danceability/danceability-musicnn-msd-2.pb",
}

BASE_URL = "https://essentia.upf.edu/models"


def default_models_dir() -> Path:
    """Return the bundled models/ directory next to this file."""
    return Path(__file__).parent.parent / "models"


def download_models(dest: Path | None = None, force: bool = False) -> None:
    dest = dest or default_models_dir()
    dest.mkdir(parents=True, exist_ok=True)

    for name, rel_path in MODELS.items():
        url = f"{BASE_URL}/{rel_path}"
        out = dest / f"{name}-musicnn-msd-2.pb"

        if out.exists() and not force:
            print(f"  [skip] {name} (already exists)")
            continue

        print(f"  [download] {name} ...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, out)
            size_kb = out.stat().st_size // 1024
            print(f"{size_kb} KB")
        except Exception as exc:
            print(f"FAILED — {exc}")
            if out.exists():
                out.unlink()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download Essentia mood models")
    parser.add_argument("--dest", type=Path, default=None,
                        help="Directory to save models (default: tools/essentia-tagger/models/)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if files already exist")
    args = parser.parse_args()

    print(f"Downloading models to: {args.dest or default_models_dir()}")
    download_models(args.dest, args.force)
    print("Done.")


if __name__ == "__main__":
    main()
