# fmo-essentia-tagger

Analyze music files with [Essentia](https://essentia.upf.edu) and write the
results back to a [FMO](https://github.com/CLR-MO/fmo-tracker) entity database
via the REST API.


## Usage

```sh
# download ai models (one-time setup)
python3 -m essentia_tagger.download_models
# add confidence scores to entities using ai models
python3 -m essentia_tagger.cli --skip-analyzed

# Optionally,
# based on confidence score attributes, add tags
python3 -m essentia_tagger.cli --tag-analyzed
```


## What it Detects

### Algorithmic (no model files needed)

| Feature | Algorithm | Stored as |
|---|---|---|
| BPM | RhythmExtractor2013 | attribute `bpm` |
| Musical key | KeyExtractor | attributes `key`, `scale`, `key_strength` |

### ML classifiers (MusiCNN-MSD, ~3.1 MB each)

All classifiers are configured in `models.toml`. Set `enabled = false` on any
model to skip it for both download and analysis.

**Confidence scores** are stored as attributes with `_conf` suffix (e.g., `mood_happy_conf`).
**Tags are NOT applied during analysis** — only confidence scores are stored.
Run `--tag-analyzed` separately to apply tags based on these scores.

**Confidence scores** are stored as attributes with `_conf` suffix (e.g., `mood_happy_conf`).
**Tags are NOT applied during analysis** — only confidence scores are stored.
Run `--tag-analyzed` separately to apply tags based on these scores.

**Binary models** — store raw probability (0–1):

| Model | Attribute |
|---|---|
| `mood_happy` | `mood_happy_conf` |
| `mood_sad` | `mood_sad_conf` |
| `mood_relaxed` | `mood_relaxed_conf` |
| `mood_aggressive` | `mood_aggressive_conf` |
| `mood_party` | `mood_party_conf` |
| `mood_acoustic` | `mood_acoustic_conf` |
| `mood_electronic` | `mood_electronic_conf` |

**Multiclass models** — store all class probabilities with `_classname_conf` suffix:

| Model | Attribute pattern | Classes |
|---|---|---|
| `moods_mirex` | `moods_mirex_<class>_conf` | intense, quirky, rousing, cheerful, brooding |
| `danceability` | `danceability_<class>_conf` | danceable, not_danceable |
| `voice_instrumental` | `voice_instrumental_<class>_conf` | instrumental, vocal |
| `gender` | `gender_<class>_conf` | female, male |
| `tonal_atonal` | `tonal_atonal_<class>_conf` | atonal, tonal |
| `genre_dortmund` | `genre_dortmund_<class>_conf` | alternative, blues, electronic, ... |
| `genre_electronic` | `genre_electronic_<class>_conf` | ambient, dnb, house, techno, trance |
| `genre_rosamerica` | `genre_rosamerica_<class>_conf` | classical, dance, hiphop, jazz, ... |
| `genre_tzanetakis` | `genre_tzanetakis_<class>_conf` | blues, classical, country, disco, ... |

`fs_loop_ds` (loop role: bass/chords/fx/melody/percussion) is included but
**disabled by default** — only useful for sample/loop libraries.

> **Note on the progress display:** the per-file summary shortens attribute
> names by removing the `_conf` suffix (e.g., `mood_happy_conf` appears as
> `mood_happy`). Full attribute names are always written to the database.

Confidence scores are always stored so you can re-threshold or change tagging
rules without re-analyzing.


## Tagging Rules (--tag-analyzed)

The `--tag-analyzed` phase applies these rules based on confidence scores:

| Rule | Condition | Tag(s) applied |
|------|-----------|---------------|
| Sad+Aggressive | `mood_sad_conf > mood_happy_conf AND mood_aggressive_conf > 0.7` | `/mood/intense` |
| Happy | `mood_happy_conf > mood_sad_conf AND mood_happy_conf > 0.5` | `/mood/happy` |
| High Aggressive | `mood_aggressive_conf > 0.7` | `/mood/energetic` or `/mood/intense` |
| Party | `mood_party_conf > 0.7` | `/mood/party` |
| Relaxed | `mood_relaxed_conf > 0.6 AND mood_aggressive_conf < 0.3` | `/mood/relaxed`, `/mood/calm` |
| Acoustic | `mood_acoustic_conf > 0.9` with conditions | `/mood/acoustic` variants |
| BPM < 90 | `bpm < 90` | `/mood/slow`, `/mood/ballad` |
| BPM > 160 | `bpm > 160` | `/mood/fast`, `/mood/intense` |
| MIREX classes | class confidence above threshold | `/mood/intense`, `/mood/quirky`, etc. |
| Danceable | `danceability_danceable_conf > 0.6` | `/mood/danceable` |


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
python3 -m essentia_tagger.download_models
python3 -m essentia_tagger.cli --skip-analyzed
```

### Running without installing

If you don't want to install at all, just run from this directory:

```bash
# Add the package to the Python path for this shell session
export PYTHONPATH="$PWD:$PYTHONPATH"

python3 -m essentia_tagger.download_models
python3 -m essentia_tagger.cli --skip-analyzed
```


## CLI Options

```bash
# Tag all audio entities in the active database
clrmo-tag

# Preview without writing (dry run, shows mood scores)
clrmo-tag --dry-run

# Skip files already analyzed (have essentia attribute set)
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

Every successfully analyzed file gets an `essentia = <unix_timestamp>` attribute
written back. Use `--skip-analyzed` to skip those on the next run:

```bash
clrmo-tag --skip-analyzed
```

This adds `-@essentia` to the query, only processing entities that don't
yet have an `essentia` attribute set.  You can also manually query for files
analyzed after a certain time:

```bash
clrmo-tag --query "@essentia:>:1744060800"
```


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
        "mood_happy_conf": 0.83,
        "mood_acoustic_conf": 0.71,
        "genre_electronic_dnb_conf": 0.45,
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
print(result["attributes"])  # {"bpm": 120.3, "key": "C", "mood_happy_conf": 0.83, ...}
print(result["tags"])        # [] (empty - tagging is done separately with --tag-analyzed)
```
