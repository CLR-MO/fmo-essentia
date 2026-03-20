# clrmo-essentia-tagger

Analyze music files with [Essentia](https://essentia.upf.edu) and write the
results back to a [CLRMO](https://github.com/example/clrmo) entity database
via the REST API.

## What it detects

| Feature | Algorithm | Stored as |
|---|---|---|
| BPM | RhythmExtractor2013 | attribute `bpm` |
| Musical key | KeyExtractor | attributes `key`, `scale`, `key_strength` |
| Mood: happy | MusiCNN-MSD classifier | attribute `mood_happy` (0–1), tag `/ mood / happy` |
| Mood: sad | MusiCNN-MSD classifier | attribute `mood_sad`, tag `/ mood / sad` |
| Mood: relaxed | MusiCNN-MSD classifier | attribute `mood_relaxed`, tag `/ mood / relaxed` |
| Mood: aggressive | MusiCNN-MSD classifier | attribute `mood_aggressive`, tag `/ mood / aggressive` |
| Mood: party | MusiCNN-MSD classifier | attribute `mood_party`, tag `/ mood / party` |
| Mood: acoustic | MusiCNN-MSD classifier | attribute `mood_acoustic`, tag `/ mood / acoustic` |
| Mood: electronic | MusiCNN-MSD classifier | attribute `mood_electronic`, tag `/ mood / electronic` |
| Danceability | MusiCNN-MSD classifier | attribute `danceability`, tag `/ mood / danceability` |

Mood tags are only applied when the classifier probability is ≥ the threshold
(default 0.5). Raw probabilities are always stored as attributes so you can
re-threshold later without re-analyzing.

## Installation

**Requirements:** Python 3.9+, pip.

```bash
# Install essentia (includes TensorFlow support)
pip3 install essentia-tensorflow

# Install this package (from the essentia-tagger directory)
pip3 install .
```

Then download the TensorFlow model files (~3 MB each, 8 models total):

```bash
clrmo-download-models
# or without the entry point:
python3 -m essentia_tagger.download_models
```

Models are saved to `models/` inside this directory by default. Pass `--dest`
to save elsewhere.

### pip vs pip3

Use whichever command points to your Python 3 installation. On most Linux
systems `pip3` is explicit; on macOS with Homebrew or in a virtualenv `pip`
usually works too. You can always use `python3 -m pip install …` to be sure
you're using the right interpreter.

### Older pip / setuptools (pip < 23, setuptools < 64)

Editable installs (`pip install -e .`) require pip 23+ and setuptools 64+.
On older systems use a regular install instead:

```bash
pip3 install .
```

If the `clrmo-tag` command isn't found after installing (entry points
occasionally fail on very old pip), run the tool directly:

```bash
python3 -m essentia_tagger.cli --help
python3 -m essentia_tagger.download_models
```

### Running without installing

If you don't want to install at all, just run from this directory:

```bash
# Add the package to the Python path for this shell session
export PYTHONPATH="$PWD:$PYTHONPATH"

python3 -m essentia_tagger.cli --help
python3 -m essentia_tagger.download_models
```

## Usage

**Start the CLRMO REST API server** first: open the app → Tools → Servers →
REST API Server → Start.

```bash
# Tag all audio entities in the active database
clrmo-tag

# Preview without writing (dry run)
clrmo-tag --dry-run

# Only untagged audio entities (no / mood / tag yet)
clrmo-tag --query "-/ mood /"

# Skip BPM/key, only run mood models
clrmo-tag --no-bpm --no-key

# Use a different database
clrmo-tag --db my-other-db

# Custom mood threshold
clrmo-tag --mood-threshold 0.6

# Custom server port
clrmo-tag --url http://127.0.0.1:3283
```

## API used

All results are written via a single REST call per batch:

```http
POST /entities/bulk/update
Content-Type: application/json

{
  "updates": [
    {
      "id": 42,
      "tags": ["/ mood / happy", "/ mood / acoustic"],
      "attributes": {
        "bpm": 120.3,
        "key": "C",
        "scale": "major",
        "mood_happy": 0.83,
        "mood_acoustic": 0.71,
        ...
      }
    },
    ...
  ]
}
```

Tags are **added** (existing tags are preserved). Attributes are **merged**
(existing keys not present in the payload are preserved).

## Models

Models are pre-trained on the
[Million Song Dataset](http://millionsongdataset.com) by the
[Music Technology Group, UPF](https://www.upf.edu/web/mtg).
See [essentia.upf.edu/models](https://essentia.upf.edu/models/) for details.

Models are not bundled with this package. Run `clrmo-download-models` to
fetch them (~25 MB total).

## Extending

`analyzer.py` is designed to be easy to extend:

```python
from essentia_tagger.analyzer import analyze_file

result = analyze_file("/path/to/song.mp3")
print(result["attributes"])  # {"bpm": 120.3, "key": "C", "mood_happy": 0.83, ...}
print(result["tags"])        # ["/ mood / happy", "/ mood / acoustic"]
```
