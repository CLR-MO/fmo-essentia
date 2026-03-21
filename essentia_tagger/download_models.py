"""Download Essentia TensorFlow models for music analysis."""

import os
import sys
import urllib.request
from pathlib import Path

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]

BASE_URL = "https://essentia.upf.edu/models"


def _repo_models_toml() -> Path:
    """Return the models.toml next to pyproject.toml (one level up from this file)."""
    return Path(__file__).parent.parent / "models.toml"


def load_models_config(config_path: Path | None = None) -> dict:
    """Load models.toml and return the [models] table."""
    path = config_path or _repo_models_toml()
    if tomllib is None:
        raise RuntimeError(
            "TOML support requires Python 3.11+ or: pip install tomli"
        )
    with open(path, "rb") as f:
        return tomllib.load(f)["models"]


def default_models_dir() -> Path:
    """Return the bundled models/ directory next to this file."""
    return Path(__file__).parent.parent / "models"


def download_models(
    dest: Path | None = None,
    force: bool = False,
    config_path: Path | None = None,
    only_enabled: bool = True,
) -> None:
    dest = dest or default_models_dir()
    dest.mkdir(parents=True, exist_ok=True)

    models = load_models_config(config_path)

    for name, spec in models.items():
        if only_enabled and not spec.get("enabled", True):
            print(f"  [skip]  {name} (disabled in models.toml)")
            continue

        rel_path = spec["path"]
        url = f"{BASE_URL}/{rel_path}"
        out = dest / Path(rel_path).name  # use actual filename from path

        if out.exists() and not force:
            print(f"  [skip]  {name} (already exists)")
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
    parser = argparse.ArgumentParser(description="Download Essentia models listed in models.toml")
    parser.add_argument("--dest", type=Path, default=None,
                        help="Directory to save models (default: tools/essentia-tagger/models/)")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to models.toml (default: auto-detected next to pyproject.toml)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if files already exist")
    parser.add_argument("--all", dest="all_models", action="store_true",
                        help="Download all models including disabled ones")
    args = parser.parse_args()

    dest = args.dest or default_models_dir()
    print(f"Downloading models to: {dest}")
    download_models(dest, force=args.force, config_path=args.config, only_enabled=not args.all_models)
    print("Done.")


if __name__ == "__main__":
    main()
