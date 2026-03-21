# clrmo-essentia-tagger

Analyze music files with [Essentia](https://essentia.upf.edu) and write the
results back to a [CLRMO](https://github.com/example/clrmo) entity database
via the REST API.

## What it detects

### Algorithmic (no model files needed)

| Feature | Algorithm | Stored as |
|---|---|---|
| BPM | RhythmExtractor2013 | attribute `bpm` |
| Musical key | KeyExtractor | attributes `key`, `scale`, `key_strength` |

### ML classifiers (MusiCNN-MSD, ~3.1 MB each)

All classifiers are configured in `models.toml`. Set `enabled = false` on any
model to skip it for both download and analysis.

**Binary** — raw probability (0–1) stored as an attribute; tag applied when probability ≥ threshold:

| Model | Attribute | Tag (if above threshold) |
|---|---|---|
| `mood_happy` | `mood_happy` | `/ mood / happy` |
| `mood_sad` | `mood_sad` | `/ mood / sad` |
| `mood_relaxed` | `mood_relaxed` | `/ mood / relaxed` |
| `mood_aggressive` | `mood_aggressive` | `/ mood / aggressive` |
| `mood_party` | `mood_party` | `/ mood / party` |
| `mood_acoustic` | `mood_acoustic` | `/ mood / acoustic` |
| `mood_electronic` | `mood_electronic` | `/ mood / electronic` |

**Multi-class** — top class stored as string attribute; tagging rules depend on the number of classes:

- **2-class models** (`danceability`, `voice_instrumental`, `gender`, `tonal_atonal`) use a Sigmoid output, so both classes can independently exceed the threshold. Only the winning class (highest probability) is tagged to avoid contradictory tags like `danceable` + `not_danceable`.
- **N-class models** (genre classifiers, `moods_mirex`) use a Softmax or multi-label output; any class above threshold is tagged.

| Model | Attribute | Tag prefix | Classes |
|---|---|---|---|
| `moods_mirex` | `moods_mirex` | `/ mood /` | rousing, cheerful, brooding, quirky, intense |
| `danceability` | `danceability` | `/ music /` | danceable, not_danceable |
| `voice_instrumental` | `voice_instrumental` | `/ music /` | instrumental, vocal |
| `gender` | `gender` | `/ vocal /` | female, male |
| `tonal_atonal` | `tonal_atonal` | `/ music /` | atonal, tonal |
| `genre_dortmund` | `genre_dortmund` | `/ genre /` | alternative, blues, electronic, folkcountry, funksoulrnb, jazz, pop, raphiphop, rock |
| `genre_electronic` | `genre_electronic` | `/ genre / electronic /` | ambient, dnb, house, techno, trance |
| `genre_rosamerica` | `genre_rosamerica` | `/ genre /` | classical, dance, hiphop, jazz, pop, rnb, rock, speech |
| `genre_tzanetakis` | `genre_tzanetakis` | `/ genre /` | blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock |

`fs_loop_ds` (loop role: bass/chords/fx/melody/percussion) is included but
**disabled by default** — only useful for sample/loop libraries.

> **Note on the progress display:** the per-file summary line shortens tags by
> showing only the last path segment (e.g. `/ genre / electronic / dnb` appears
> as `dnb`).  `dnb` = drum and bass from the `genre_electronic` classifier.
> Full tag paths are always written to the database.

Raw probabilities are always stored as attributes so you can re-threshold later
without re-analyzing.

## Installation

**Requirements:** Python 3.9+, pip.

```bash
# Install essentia (includes TensorFlow support)
pip3 install essentia-tensorflow

# Install this package (from the essentia-tagger directory)
pip3 install .
```

Then download the TensorFlow model files (~3.1 MB each, ~50 MB for all enabled models):

```bash
clrmo-download-models
# or without the entry point:
python3 -m essentia_tagger.download_models
```

Models are saved to `models/` inside this directory by default. Pass `--dest`
to save elsewhere.

### Selecting which models to download

Edit `models.toml` and set `enabled = false` on any model you don't want.
Re-run `clrmo-download-models` — disabled models are skipped, already-present
files are skipped unless you pass `--force`.

```bash
# Re-download everything (including already-present files)
clrmo-download-models --force

# Download all models including disabled ones
clrmo-download-models --all
```

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

# Preview without writing (dry run, shows mood scores)
clrmo-tag --dry-run

# Skip files already analyzed (have essentia=1 attribute)
clrmo-tag --skip-analyzed

# Only untagged audio entities (no / mood / tag yet)
clrmo-tag --query "-/ mood /"

# Skip BPM/key, only run ML models
clrmo-tag --no-bpm --no-key

# Skip ML models, only run BPM + key
clrmo-tag --no-mood

# Use a different database
clrmo-tag --db my-other-db

# Custom mood threshold (applies to all binary models)
clrmo-tag --mood-threshold 0.6

# Per-model threshold override
clrmo-tag --threshold electronic=0.75 --threshold danceability=0.4

# Custom server port
clrmo-tag --url http://127.0.0.1:3283

# Load audio once at 44.1 kHz and resample in memory (faster, slightly less accurate)
clrmo-tag --single-pass
```

## Resuming after interruption

Every successfully analyzed file gets an `essentia = 1` attribute written back.
Use `--skip-analyzed` to skip those on the next run:

```bash
clrmo-tag --skip-analyzed
```

This adds `-@essentia:=:1` to the query, excluding any entity that already has
the `essentia` attribute set to 1.  Entities with no `essentia` attribute (not
yet processed) are always included.

## CPU load and thermal management

The ML models run on the CPU via TensorFlow and can sustain near-100% load
across all cores for extended runs. Two flags help keep thermals under control:

```bash
# Limit TensorFlow to N threads (default: half your CPU count)
clrmo-tag --tf-threads 4

# Sleep N seconds between each file (allows the CPU to cool briefly)
clrmo-tag --delay 2

# Combine both for gentler sustained load
clrmo-tag --tf-threads 4 --delay 1

# Remove the thread cap for maximum speed (use at your own risk)
clrmo-tag --tf-threads 0
```

**`--tf-threads`** defaults to half your logical CPU count. This keeps the
tool from saturating every core for the duration of a large run, at roughly
half the throughput. Pass `--tf-threads 0` to remove the cap.

**`--delay`** sleeps between each file. Even 0.5–1 s gives the CPU a
meaningful thermal break without adding much total time on large libraries.
If you are running other CPU-heavy processes concurrently, combine both flags.

## Useful CLRMO searches after tagging

These queries work in the CLRMO search bar. Attributes use `attr:name` syntax;
tags use path notation. Ranges use `>`, `<`, `>=`, `<=`.

### Mood / energy

```
# Calm, low-energy tracks
/ mood / relaxed

# High-energy
/ mood / aggressive

# Happy or party
/ mood / happy | / mood / party

# Electronic feel specifically (uses per-model threshold)
/ mood / electronic
```

### Background music for tasks

Good background music tends to be instrumental, calm, and tonal (not jarring).
Combine attributes for more precision:

```
# Instrumental only
/ music / instrumental

# Calm instrumentals — the sweet spot for focus music
/ mood / relaxed  / music / instrumental

# Calm, tonal, instrumental (tonal = melodic rather than noise/atonal)
/ mood / relaxed  / music / tonal  / music / instrumental

# Not danceable = less rhythmically insistent = less distracting
/ music / not_danceable  / music / instrumental

# Brooding/wistful MIREX cluster — often ambient-adjacent
/ mood / brooding  / music / instrumental
```

### Voice and gender

```
# Instrumental tracks (no prominent vocals)
/ music / instrumental

# Vocal tracks
/ music / vocal

# Female vocals
/ vocal / female

# Male vocals
/ vocal / male
```

### Genre

```
# Broad genre (Dortmund classifier — best general-purpose)
/ genre / rock
/ genre / jazz
/ genre / electronic

# Electronic subgenre
/ genre / electronic / ambient
/ genre / electronic / house

# Multiple genre classifiers agree (more confident result)
/ genre / jazz  attr:genre_rosamerica = jazz

# Disco
/ genre / disco
```

### BPM ranges

```
# Slow (ballad / ambient)
attr:bpm < 80

# Mid-tempo
attr:bpm >= 80  attr:bpm < 120

# Upbeat
attr:bpm >= 120  attr:bpm < 160

# Fast (punk / drum and bass)
attr:bpm >= 160

# Around a specific BPM (±5)
attr:bpm >= 125  attr:bpm <= 135
```

### Key / scale

```
# Minor key tracks (typically darker/moodier)
attr:scale = minor

# Specific key
attr:key = C  attr:scale = major

# High confidence key detection only
attr:key_strength > 0.6
```

### Combining signals

```
# Upbeat happy electronic tracks
/ mood / happy  / mood / electronic  attr:bpm >= 120

# Sad slow minor-key tracks
/ mood / sad  attr:scale = minor  attr:bpm < 80

# Acoustic folk-adjacent (acoustic + not electronic + moderate BPM)
/ mood / acoustic  attr:mood_electronic < 0.2  attr:bpm >= 70  attr:bpm < 130

# Female vocal tracks
/ mood / gender  attr:voice_instrumental > 0.6
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
      "tags": ["/ mood / happy", "/ mood / acoustic", "/ genre / rock"],
      "attributes": {
        "bpm": 120.3,
        "key": "C",
        "scale": "major",
        "mood_happy": 0.83,
        "mood_acoustic": 0.71,
        "voice_instrumental": 0.12,
        "genre_dortmund": "rock",
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
fetch them (~50 MB total for all enabled models). The full list and
enable/disable switches are in `models.toml`.

## Extending

`analyzer.py` is designed to be easy to extend. Add new models directly to
`models.toml` — the analyzer picks them up automatically without code changes.

```python
from essentia_tagger.analyzer import analyze_file

result = analyze_file("/path/to/song.mp3")
print(result["attributes"])  # {"bpm": 120.3, "key": "C", "mood_happy": 0.83, ...}
print(result["tags"])        # ["/ mood / happy", "/ mood / acoustic", "/ genre / rock"]
```
