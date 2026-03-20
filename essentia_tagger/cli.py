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

    # Post in batches of 50
    clrmo-tag --batch-size 50

    # Use a specific database
    clrmo-tag --db my-music-db
"""

import argparse
import os
import sys
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
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Number of results to POST per bulk update call (default: 20)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Analyze files but do not write anything to the API")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

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
    from .analyzer import analyze_file

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
    print(f"Fetching audio entities{' (query: ' + args.query + ')' if args.query else ''}...")

    entities = list(client.iter_audio_entities(query=args.query, db=args.db))

    # Keep only entities with a non-empty path
    entities = [e for e in entities if e.get("path")]
    print(f"Found {len(entities)} audio entities with valid paths.")

    if args.skip_analyzed:
        # Would need full entity details to check attributes; for simplicity we
        # let the user pre-filter via --query instead.
        print("Note: --skip-analyzed not yet implemented; use --query to pre-filter.")

    if not entities:
        print("Nothing to do.")
        return

    # ── Analyze + collect updates ─────────────────────────────────────────────
    total = len(entities)
    pending: list[dict] = []
    posted = 0
    failed = 0

    for i, entity in enumerate(entities, 1):
        eid = entity["id"]
        path = entity["path"]
        name = entity.get("name") or entity.get("path", "")
        prefix = f"[{i}/{total}] {name}"

        if not os.path.isfile(path):
            print(f"{prefix}  SKIP (file not found)")
            continue

        print(f"{prefix}  analyzing...", end="", flush=True)
        try:
            result = analyze_file(
                path,
                models_dir=args.models_dir,
                mood_threshold=args.mood_threshold,
                mood_thresholds=mood_thresholds or None,
                tag_prefix=args.tag_prefix,
                run_bpm=not args.no_bpm,
                run_key=not args.no_key,
                run_mood=not args.no_mood,
                single_pass=args.single_pass,
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            failed += 1
            continue

        attrs = result["attributes"]
        tags = result["tags"]
        summary_parts = []
        if "bpm" in attrs:
            summary_parts.append(f"bpm={attrs['bpm']}")
        if "key" in attrs:
            summary_parts.append(f"key={attrs['key']} {attrs.get('scale','')}")
        if tags:
            summary_parts.append("tags=[" + ", ".join(t.split("/")[-1].strip() for t in tags) + "]")
        print(f"  {', '.join(summary_parts) or 'no results'}")

        if args.dry_run and not args.no_mood:
            MOOD_KEYS = [
                ("mood_happy", "happy"), ("mood_sad", "sad"), ("mood_relaxed", "relaxed"),
                ("mood_aggressive", "aggr"), ("mood_party", "party"), ("mood_acoustic", "acou"),
                ("mood_electronic", "elec"), ("danceability", "dance"),
            ]
            mood_parts = []
            for key, label in MOOD_KEYS:
                if key in attrs:
                    score = attrs[key]
                    t = mood_thresholds.get(key, args.mood_threshold)
                    flag = "*" if score >= t else " "
                    t_note = f">{t}" if t != args.mood_threshold else ""
                    mood_parts.append(f"{label}={score:.2f}{flag}{t_note}")
            if mood_parts:
                print(f"  mood:  {' '.join(mood_parts)}  (default threshold={args.mood_threshold})")

        if args.verbose:
            for k, v in attrs.items():
                print(f"    {k}: {v}")

        pending.append({"id": eid, "tags": tags, "attributes": attrs})

        # Post in batches
        if len(pending) >= args.batch_size:
            if not args.dry_run:
                r = client.bulk_update(pending, db=args.db)
                posted += r.get("updated", 0)
                for err in r.get("errors", []):
                    print(f"  API error for id={err['id']}: {err['error']}", file=sys.stderr)
            else:
                posted += len(pending)
            pending.clear()

    # Flush remaining
    if pending:
        if not args.dry_run:
            r = client.bulk_update(pending, db=args.db)
            posted += r.get("updated", 0)
            for err in r.get("errors", []):
                print(f"  API error for id={err['id']}: {err['error']}", file=sys.stderr)
        else:
            posted += len(pending)

    mode = "[DRY RUN] " if args.dry_run else ""
    print(f"\n{mode}Done. Updated: {posted}  Failed: {failed}")


if __name__ == "__main__":
    main()
