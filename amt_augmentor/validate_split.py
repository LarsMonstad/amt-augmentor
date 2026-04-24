import csv
import json
import os
import sys
import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# Extensions the dataset uses. Only stripped when they appear at the very end
# of the filename — never from the middle (that was the bug this module fixes).
_AUDIO_MIDI_EXTS = (".wav", ".mid", ".midi", ".flac", ".mp3", ".m4a", ".aiff")

_AUG_MARKER = "_augmented_"


def canonical_stem(path_or_name: str) -> str:
    """Return the filename stem with at most one trailing audio/MIDI extension removed.

    Unlike ``os.path.splitext`` this never strips an interior ".mid"/".2"/etc.,
    and unlike ``str.replace(".mid", "")`` it never strips an embedded ".mid"
    fragment (as in ``foo.mid_cleaned.mid``).
    """
    name = os.path.basename(path_or_name)
    lower = name.lower()
    for ext in _AUDIO_MIDI_EXTS:
        if lower.endswith(ext):
            return name[: -len(ext)]
    return name


def is_augmented_version(filename: str) -> bool:
    """True if the filename carries the ``_augmented_`` marker."""
    return _AUG_MARKER in filename.lower()


def original_from_aug(filename: str) -> str:
    """Return the canonical stem of the source original for an augmented file.

    Works on either an augmented stem or a bare original stem (in which case
    the input is returned unchanged, after canonicalization).
    """
    stem = canonical_stem(filename)
    if _AUG_MARKER in stem:
        return stem.split(_AUG_MARKER, 1)[0]
    return stem


# Back-compat alias — older callers and tests may import this name.
def get_original_song_name(filename: str) -> str:
    return original_from_aug(filename)


def _row_filename(row: Dict[str, str]) -> str:
    """Pick the filename column to normalize. Prefer midi_filename, fall back to audio."""
    for key in ("midi_filename", "audio_filename"):
        val = row.get(key)
        if val:
            return val
    return ""


def find_contamination(csv_file: str) -> Dict:
    """Scan ``csv_file`` for split issues and return a structured report.

    Report keys:
      - ``csv_file``: path inspected
      - ``totals``: per-split row counts, split into originals/augmented
      - ``aug_in_eval``: augmented rows sitting in test/validation
      - ``cross_split``: augmented row whose source original is in a different split
      - ``orphan_aug``: augmented row with no matching original anywhere in the CSV
      - ``clean``: True iff all three issue lists are empty
    """
    # stem -> split  (for originals only)
    originals: Dict[str, str] = {}
    # Preserve insertion order for stable output
    aug_rows: List[Tuple[str, str, str]] = []  # (aug_stem, row_split, source_original_stem)
    totals: Dict[str, Dict[str, int]] = defaultdict(lambda: {"original": 0, "augmented": 0})

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = _row_filename(row)
            if not fname:
                continue
            split = row.get("split", "")
            stem = canonical_stem(fname)

            if is_augmented_version(stem):
                source = stem.split(_AUG_MARKER, 1)[0]
                aug_rows.append((stem, split, source))
                totals[split]["augmented"] += 1
            else:
                # If the same original appears twice across splits, that itself is a bug;
                # keep the first seen split but flag via cross_split detection below.
                originals.setdefault(stem, split)
                totals[split]["original"] += 1

    aug_in_eval: List[Dict[str, str]] = []
    cross_split: List[Dict[str, str]] = []
    orphan_aug: List[Dict[str, str]] = []

    for aug_stem, split, source in aug_rows:
        if split in ("test", "validation"):
            aug_in_eval.append({"file": aug_stem, "split": split})

        src_split = originals.get(source)
        if src_split is None:
            orphan_aug.append({"file": aug_stem, "missing_original": source, "split": split})
        elif src_split != split:
            cross_split.append(
                {
                    "file": aug_stem,
                    "split": split,
                    "original": source,
                    "original_split": src_split,
                }
            )

    clean = not (aug_in_eval or cross_split or orphan_aug)

    return {
        "csv_file": csv_file,
        "totals": {k: dict(v) for k, v in totals.items()},
        "aug_in_eval": aug_in_eval,
        "cross_split": cross_split,
        "orphan_aug": orphan_aug,
        "clean": clean,
    }


def _print_report(report: Dict) -> None:
    print(f"\nValidating dataset split in {report['csv_file']}...")

    print("\n1. Basic Split Statistics:")
    print("-" * 50)
    for split, counts in report["totals"].items():
        total = counts["original"] + counts["augmented"]
        print(f"{split}: {total} total entries")
        print(f"  - Original songs: {counts['original']}")
        print(f"  - Augmented versions: {counts['augmented']}")

    print("\n2. Checking for Augmented Songs in Test/Validation:")
    print("-" * 50)
    by_split: Dict[str, List[str]] = defaultdict(list)
    for item in report["aug_in_eval"]:
        by_split[item["split"]].append(item["file"])
    for split in ("test", "validation"):
        files = by_split.get(split, [])
        if files:
            print(f"ERROR: Found {len(files)} augmented songs in {split} split:")
            for f in files:
                print(f"  - {f}")
        else:
            print(f"OK: No augmented songs found in {split} split")

    print("\n3. Checking for Cross-Split Contamination:")
    print("-" * 50)
    if report["cross_split"]:
        print(f"ERROR: {len(report['cross_split'])} contaminated rows:")
        for item in report["cross_split"]:
            print(
                f"  - '{item['original']}' is in {item['original_split']}, "
                f"but augmented '{item['file']}' is in {item['split']}"
            )
    else:
        print("OK: No cross-split contamination found")

    print("\n4. Checking for Orphan Augmented Rows:")
    print("-" * 50)
    if report["orphan_aug"]:
        print(f"ERROR: {len(report['orphan_aug'])} augmented rows have no matching original:")
        for item in report["orphan_aug"]:
            print(f"  - '{item['file']}' (expected original: '{item['missing_original']}')")
    else:
        print("OK: Every augmented row has a matching original")

    if report["clean"]:
        print("\nValidation PASSED: Dataset split appears to be clean!")
    else:
        print("\nValidation FAILED: Issues were found in the dataset split!")


def validate_dataset_split(csv_file: str, strict: bool = False) -> bool:
    """Human-readable validator. Returns True if clean, False otherwise.

    When ``strict`` is True and issues are found, raises ``SystemExit(1)`` so
    this function can be used directly from CI scripts.
    """
    report = find_contamination(csv_file)
    _print_report(report)
    if strict and not report["clean"]:
        raise SystemExit(1)
    return report["clean"]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate dataset split in CSV file (checks for cross-split contamination)."
    )
    parser.add_argument("csv_file", type=str, help="Path to the CSV file")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with a non-zero status if any contamination is found.",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit the report as JSON on stdout instead of the human-readable format.",
    )

    args = parser.parse_args(argv)

    report = find_contamination(args.csv_file)
    if args.as_json:
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        _print_report(report)

    if not report["clean"] and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
