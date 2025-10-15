#!/usr/bin/env python3
"""
Prefix TIFF filenames by channel based on their marker folder.

Rule:
  - If the immediate parent folder is 'DAPI' (case-insensitive) -> prefix 'ch1_'
  - Otherwise -> prefix 'ch2_'

Safety:
  - Dry-run by default (use --apply to actually rename)
  - Skips files already starting with 'ch1_' or 'ch2_'
  - Skips if target filename already exists
  - Handles both .tif and .tiff (any casing)
"""

import argparse
import logging
import os
from pathlib import Path

CH1_PREFIX = "ch1_"
CH2_PREFIX = "ch2_"
ALREADY_PREFIXES = (CH1_PREFIX, CH2_PREFIX)
TIFF_EXTS = {".tif", ".tiff"}

def is_tiff(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in TIFF_EXTS

def decide_prefix(marker_name: str) -> str:
    return CH1_PREFIX if marker_name.strip().lower() == "dapi" else CH2_PREFIX

def main():
    parser = argparse.ArgumentParser(
        description="Prefix TIFF files by channel based on marker folder name."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Top-level directory that contains your batch folders.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rename files (otherwise dry-run).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging (DEBUG).",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    root: Path = args.root
    if not root.exists() or not root.is_dir():
        logging.error("Root path does not exist or is not a directory: %s", root)
        raise SystemExit(1)

    dry_run = not args.apply
    if dry_run:
        logging.info("DRY-RUN mode (no changes will be made). Use --apply to rename.")

    stats = {
        "seen_files": 0,
        "tiff_files": 0,
        "already_prefixed": 0,
        "renamed": 0,
        "skipped_target_exists": 0,
        "errors": 0,
    }

    for dirpath, dirnames, filenames in os.walk(root):
        dir_path = Path(dirpath)

        for name in filenames:
            stats["seen_files"] += 1
            src = dir_path / name
            if not is_tiff(src):
                continue
            stats["tiff_files"] += 1

            marker = src.parent.name  # immediate parent folder is the marker
            prefix = decide_prefix(marker)

            # Skip if already prefixed
            if any(src.name.startswith(p) for p in ALREADY_PREFIXES):
                logging.debug("Already prefixed, skipping: %s", src)
                stats["already_prefixed"] += 1
                continue

            dst = src.with_name(prefix + src.name)

            # Avoid overwriting
            if dst.exists():
                logging.warning("Target exists, skipping: %s -> %s", src, dst)
                stats["skipped_target_exists"] += 1
                continue

            logging.info("Rename: %s  ->  %s", src, dst)
            if not dry_run:
                try:
                    src.rename(dst)
                    stats["renamed"] += 1
                except Exception as e:
                    logging.error("Error renaming %s -> %s: %s", src, dst, e)
                    stats["errors"] += 1

    logging.info("Done.")
    logging.info(
        "Summary | TIFFs: %d | Renamed: %d | Already prefixed: %d | Target exists: %d | Errors: %d",
        stats["tiff_files"],
        stats["renamed"],
        stats["already_prefixed"],
        stats["skipped_target_exists"],
        stats["errors"],
    )
    if dry_run:
        logging.info("No changes made. Re-run with --apply to perform renames.")

if __name__ == "__main__":
    main()
