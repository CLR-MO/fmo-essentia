"""
clrmo-tag — analyze music files and write results back to CLRMO via the REST API.

Usage examples:

    # Analyze all audio entities in the active database
    clrmo-tag

    # Custom server URL, dry run to preview without writing
    clrmo-tag --url http://127.0.0.1:3283 --dry-run

    # Only run BPM + key, skip mood models
    clrmo-tag --no-mood

    # Filter by a CLRMO query (e.g. only untagged audio entities)
    clrmo-tag --query "-/ mood /"

    # Skip entities that already have a 'bpm' attribute
    clrmo-tag --skip-analyzed

    # Tag already-analyzed entities with mood tags based on their essentia attributes
    clrmo-tag --tag-analyzed

    # Tag analyzed entities with a custom tag prefix
    clrmo-tag --tag-analyzed --tag-prefix "/ mood /"

    # Post in batches of 50
    clrmo-tag --batch-size 50

    # Use a specific database
    clrmo-tag --db my-music-db
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Tag music entities in CLRMO using Essentia audio analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--url", default="http://127.0.0.1:3283",
                        help="CLRMO REST API base URL (default: http://127.0.0.1:3283)")
    parser.add_argument("--db", default=None,
                        help="Database name or path (uses server active DB if omitted)")
    parser.add_argument("--query", default="",
                        help="CLRMO query to filter entities (default: all audio entities)")
    parser.add_argument("--models-dir", type=Path, default=None,
                        help="Path to directory containing .pb model files")
    parser.add_argument("--mood-threshold", type=float, default=0.5,
                        help="Probability threshold to apply a mood tag (default: 0.5)")
    parser.add_argument("--threshold", metavar="MOOD=VALUE", action="append", default=[],
                        help="Per-mood threshold override, e.g. --threshold electronic=0.75 "
                             "(can be repeated; mood names: happy sad relaxed aggressive party acoustic electronic danceability)")
    parser.add_argument("--tag-prefix", default="/ mood /",
                        help="Prefix for mood tags (default: '/ mood /')")
    parser.add_argument("--no-bpm", action="store_true", help="Skip BPM analysis")
    parser.add_argument("--no-key", action="store_true", help="Skip key detection")
    parser.add_argument("--no-mood", action="store_true", help="Skip mood/danceability models")
    parser.add_argument("--single-pass", action="store_true",
                        help="Load audio once at 44.1 kHz and resample in memory for mood models "
                             "instead of reading the file a second time at 16 kHz")
    parser.add_argument("--skip-analyzed", action="store_true",
                        help="Skip entities that already have a 'bpm' attribute")
    parser.add_argument("--tag-analyzed", action="store_true",
                        help="Tag entities that have already been analyzed. Queries @essentia instead of -@essentia and applies mood tagging rules based on existing attributes.")
    parser.add_argument("--no-overwrite", action="store_true",
                        help="Only add new attributes, don't overwrite existing ones")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Number of results to POST per bulk update call (default: 20)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Analyze files but do not write anything to the API")
    parser.add_argument("--delay", type=float, default=0.0, metavar="SECONDS",
                        help="Sleep this many seconds between files to allow CPU cooldown (default: 0)")
    _default_threads = max(1, (os.cpu_count() or 2) // 2)
    parser.add_argument("--tf-threads", type=int, default=_default_threads, metavar="N",
                        help=f"Limit TensorFlow to N CPU threads (0 = unlimited, "
                             f"default: half your CPU count = {_default_threads}). "
                             "Lower values reduce sustained CPU load at the cost of speed.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Graceful Ctrl+C handling
    interrupted = False
    def sigint_handler(signum, frame):
        nonlocal interrupted
        if interrupted:
            # Second Ctrl+C: force exit
            sys.exit(1)
        interrupted = True
        print("\nInterrupted. Finishing current file, then exiting...", file=sys.stderr)
    signal.signal(signal.SIGINT, sigint_handler)

    # Parse per-mood threshold overrides: "electronic=0.75" → {"mood_electronic": 0.75}
    # "danceability" has no "mood_" prefix in the model name
    mood_thresholds: dict[str, float] = {}
    for spec in args.threshold:
        try:
            name, val = spec.split("=", 1)
            key = name.strip()
            # Normalise short names to model keys
            if key != "danceability" and not key.startswith("mood_"):
                key = f"mood_{key}"
            mood_thresholds[key] = float(val)
        except ValueError:
            print(f"WARNING: ignoring bad --threshold value {spec!r} (expected MOOD=VALUE)", file=sys.stderr)

    # ── Imports (deferred so --help is fast) ─────────────────────────────────
    from .api import ClrmoClient

    def _apply_mood_rules(attrs: dict, tag_prefix: str = "/mood/") -> list[str]:
        """Apply mood tagging rules based on existing essentia attributes.

        Returns a list of mood tags to apply.
        """
        tags = []

        # Extract mood values
        happy = attrs.get("mood_happy_conf")
        sad = attrs.get("mood_sad_conf")
        agg = attrs.get("mood_aggressive_conf")
        relaxed = attrs.get("mood_relaxed_conf")
        party = attrs.get("mood_party_conf")
        acoustic = attrs.get("mood_acoustic_conf")
        danceable = attrs.get("danceability_danceable_conf")
        bpm = attrs.get("bpm")
        mirex_intense = attrs.get("moods_mirex_intense_conf")
        mirex_quirky = attrs.get("moods_mirex_quirky_conf")
        mirex_rousing = attrs.get("moods_mirex_rousing_conf")
        mirex_cheerful = attrs.get("moods_mirex_cheerful_conf")
        mirex_brooding = attrs.get("moods_mirex_brooding_conf")

        # Rule 1: Sad + Aggressive = Intense (NOT sad)
        if sad is not None and happy is not None and agg is not None:
            if sad > happy and agg > 0.7:
                tags.append(f"{tag_prefix}intense")
            elif happy > sad and happy > 0.5:
                tags.append(f"{tag_prefix}happy")

        # Rule 2: High aggressive alone → energetic or intense
        if agg is not None and agg > 0.7:
            if happy is not None and happy > 0.5:
                tags.append(f"{tag_prefix}energetic")
            else:
                tags.append(f"{tag_prefix}intense")

        # Rule 3: Party signal
        if party is not None and party > 0.7:
            tags.append(f"{tag_prefix}party")

        # Rule 4: Relaxed but not aggressive
        if relaxed is not None and relaxed > 0.6 and (agg is None or agg < 0.3):
            tags.append(f"{tag_prefix}relaxed")
            tags.append(f"{tag_prefix}calm")

        # Rule 5: Acoustic signals with conditions
        if acoustic is not None and acoustic > 0.9:
            if agg is not None and agg >= 0.4:
                if party is not None and party > 0.5:
                    tags.append(f"{tag_prefix}acoustic")
                    tags.append(f"{tag_prefix}party")
                elif danceable is not None and danceable > 0.7:
                    tags.append(f"{tag_prefix}acoustic")
                    tags.append(f"{tag_prefix}danceable")
            else:
                tags.append(f"{tag_prefix}acoustic")
                tags.append(f"{tag_prefix}calm")

        # Rule 6: BPM-based tagging
        if bpm is not None:
            if bpm < 90:
                tags.append(f"{tag_prefix}slow")
                tags.append(f"{tag_prefix}ballad")
            elif bpm > 160:
                tags.append(f"{tag_prefix}fast")
                tags.append(f"{tag_prefix}intense")

        # Rule 7: MIREX class rules
        if mirex_intense is not None and mirex_intense > 0.5:
            tags.append(f"{tag_prefix}intense")

        if mirex_quirky is not None and mirex_quirky > 0.5:
            tags.append(f"{tag_prefix}quirky")

        if mirex_rousing is not None and mirex_rousing > 0.3:
            tags.append(f"{tag_prefix}rousing")

        if mirex_cheerful is not None and mirex_cheerful > 0.3:
            tags.append(f"{tag_prefix}cheerful")

        if mirex_brooding is not None and mirex_brooding > 0.6:
            tags.append(f"{tag_prefix}brooding")

        # Rule 8: Danceability
        if danceable is not None and danceable > 0.6:
            tags.append(f"{tag_prefix}danceable")

        # Remove duplicates while preserving order
        seen = set()
        unique_tags = []
        for t in tags:
            if t not in seen:
                seen.add(t)
                unique_tags.append(t)

        return unique_tags

    # Build the environment for subprocess workers.  TF thread limits must be
    # set before TensorFlow initialises, so we pass them via env vars inherited
    # by each subprocess rather than via Python API calls.
    worker_env = os.environ.copy()
    if args.tf_threads != 0:
        worker_env["TF_NUM_INTEROP_THREADS"] = str(args.tf_threads)
        worker_env["TF_NUM_INTRAOP_THREADS"] = str(args.tf_threads)

    def _analyze(path: str) -> dict:
        """Run analyze_file in a subprocess to isolate segfaults from bad files."""
        worker_args = json.dumps({
            "path": path,
            "models_dir": str(args.models_dir) if args.models_dir else None,
            "mood_threshold": args.mood_threshold,
            "mood_thresholds": mood_thresholds or None,
            "tag_prefix": args.tag_prefix,
            "run_bpm": not args.no_bpm,
            "run_key": not args.no_key,
            "run_mood": not args.no_mood,
            "single_pass": args.single_pass,
        })
        proc = subprocess.run(
            [sys.executable, "-m", "essentia_tagger.analyzer", worker_args],
            capture_output=True,
            text=True,
            env=worker_env,
            timeout=300,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.strip().splitlines()
            # Surface the last non-empty line of stderr as the error message
            msg = next((l for l in reversed(stderr) if l.strip()), f"exit code {proc.returncode}")
            raise RuntimeError(msg)
        return json.loads(proc.stdout)

    client = ClrmoClient(base_url=args.url)

    # Verify the server is reachable
    try:
        status = client.status()
        print(f"Connected to CLRMO REST API  db={status.get('db', '?')}")
    except Exception as exc:
        print("", file=sys.stderr)
        print("  The CLRMO REST server is not running.", file=sys.stderr)
        print("  In the app:  Navigate → Tools → Servers  →  turn on REST server", file=sys.stderr)
        print("", file=sys.stderr)
        print(f"  (technical: cannot reach {args.url} — {exc})", file=sys.stderr)
        print("", file=sys.stderr)
        sys.exit(1)

    # ── Collect entities ──────────────────────────────────────────────────────
    query = args.query
    if args.tag_analyzed:
        # @essentia → entities that HAVE essentia attribute (already analyzed)
        # Applies mood tagging rules based on existing attributes.
        essentia_filter = "@essentia"
        query = f"({query}) & {essentia_filter}" if query else essentia_filter
    elif args.skip_analyzed:
        # -@essentia → entities that do NOT have essentia attribute (not yet analyzed)
        # Only processes entities that don't have essentia set yet.
        skip_filter = "-@essentia"
        query = f"({query}) & {skip_filter}" if query else skip_filter

    print(f"Fetching audio entities{' (query: ' + query + ')' if query else ''}...")

    total, entities_iter = client.iter_audio_entities(query=query, db=args.db)
    print(f"Found {total} audio entities.")

    if not total:
        print("Nothing to do.")
        return

    # ── Tag analyzed entities ─────────────────────────────────────────────────
    if args.tag_analyzed:
        tag_prefix = args.tag_prefix
        i = 0
        pending = []
        posted = 0
        failed = 0

        try:
            for entity in entities_iter:
                if interrupted:
                    break

                eid = entity["id"]
                i += 1
                name = entity.get("name") or entity.get("path") or f"id={eid}"
                prefix = f"[{i}/{total}] {name}"

                print(f"{prefix}  fetching attributes...", end="", flush=True)

                try:
                    full_entity = client.get_entity(eid, db=args.db)
                    attrs = full_entity.get("attributes", {})
                except Exception as exc:
                    print(f"  ERROR fetching attributes: {exc}")
                    failed += 1
                    continue

                mood_tags = _apply_mood_rules(attrs, tag_prefix)

                if mood_tags:
                    print(f"  → {', '.join(mood_tags)}")
                    pending.append({"id": eid, "tags": mood_tags})
                else:
                    print(f"  → no mood tags applied")

                # Bulk update every batch_size entities
                if len(pending) >= args.batch_size:
                    if not args.dry_run:
                        r = client.bulk_update(pending, db=args.db)
                        posted += r.get("updated", 0) or len(pending)
                        for err in r.get("errors", []):
                            print(f"  API error for id={err['id']}: {err['error']}", file=sys.stderr)
                    else:
                        posted += len(pending)
                    pending = []

                if args.delay > 0 and i < total:
                    time.sleep(args.delay)

        except KeyboardInterrupt:
            print("\nInterrupted.", file=sys.stderr)

        # Flush remaining
        if pending and not args.dry_run:
            r = client.bulk_update(pending, db=args.db)
            posted += r.get("updated", 0) or len(pending)
        elif pending:
            posted += len(pending)

        mode = "[DRY RUN] " if args.dry_run else ""
        print(f"\n{mode}Done. Tagged: {posted}  Failed: {failed}")
        return

    # ── Analyze + collect updates ─────────────────────────────────────────────
    pending: list[dict] = []
    posted = 0
    failed = 0
    i = 0

    try:
        for entity in entities_iter:
            if interrupted:
                break

            path = entity.get("path")
            name = entity.get("name") or path or ""

            if not path:
                print(f"[{name}]  SKIP (no path)")
                continue

            i += 1
            prefix = f"[{i}/{total}] {name}"

            if not os.path.isfile(path):
                print(f"{prefix}  SKIP (file not found)")
                continue

            eid = entity["id"]

            print(f"{prefix}  analyzing...", end="", flush=True)
            try:
                result = _analyze(path)
            except Exception as exc:
                print(f"  ERROR: {exc}")
                failed += 1
                continue

            attrs = result["attributes"]
            summary_parts = []
            if "bpm" in attrs:
                summary_parts.append(f"bpm={attrs['bpm']}")
            if "key" in attrs:
                summary_parts.append(f"key={attrs['key']} {attrs.get('scale','')}")
            print(f"  {', '.join(summary_parts) or 'no results'}")

            if args.dry_run and not args.no_mood:
                # Show all stored model weights
                mood_parts = []
                for key, value in attrs.items():
                    if key in ("bpm", "key", "scale", "key_strength", "essentia"):
                        continue
                    if not isinstance(value, (int, float)):
                        mood_parts.append(f"{key}={value}")
                    else:
                        mood_parts.append(f"{key}={value:.3f}")
                if mood_parts:
                    print(f"  weights:  {' '.join(mood_parts)}")

            if args.verbose:
                for k, v in attrs.items():
                    print(f"    {k}: {v}")

            attrs["essentia"] = int(time.time())
            if not args.dry_run:
                r = client.bulk_update([{"id": eid, "attributes": attrs}], db=args.db, add_only=args.no_overwrite)
                posted += r.get("updated", 0)
                for err in r.get("errors", []):
                    print(f"  API error for id={err['id']}: {err['error']}", file=sys.stderr)
                    # Fatal errors: database connection issues
                    if "connection" in err["error"].lower() or "not open" in err["error"].lower():
                        print("  Fatal: database connection lost. Exiting.", file=sys.stderr)
                        sys.exit(1)
            else:
                posted += 1

            if args.delay > 0 and i < total:
                time.sleep(args.delay)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)

    mode = "[DRY RUN] " if args.dry_run else ""
    print(f"\n{mode}Done. Updated: {posted}  Failed: {failed}")


if __name__ == "__main__":
    main()
