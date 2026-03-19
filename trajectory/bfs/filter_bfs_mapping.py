#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BFS Mapping Filter Tool

Features:
1. Collect all file paths from the old bfs_mapping (used as a whitelist)
2. Filter the newly generated bfs_mapping, keeping only files present in the whitelist
3. Generate a detailed report showing kept, deleted, and missing files

Use case:
After modifying the selector in fsm.json, regenerate BFS to get the latest selectors
while retaining only the original trajectory scope without adding new trajectories.
"""

import argparse
import sys
from pathlib import Path
from typing import Set, Tuple, List, Dict


def normalize_path(path: str) -> str:
    """
    Normalize a path, keeping only the last two levels (target_state/filename).

    This allows ignoring differences in the top-level directory (e.g. home_initial),
    focusing only on the actual target state and filename.

    Examples:
    - home_initial/SESSION_COMPLETED_SUCCESS/macro_001.json
      -> SESSION_COMPLETED_SUCCESS/macro_001.json
    - SESSION_COMPLETED_SUCCESS/macro_001.json
      -> SESSION_COMPLETED_SUCCESS/macro_001.json

    Args:
        path: Original relative path

    Returns:
        Normalized path (last two levels)
    """
    parts = Path(path).parts
    if len(parts) >= 2:
        return str(Path(parts[-2]) / parts[-1])
    return path


def collect_file_paths(bfs_mapping_dir: Path) -> Set[str]:
    """
    Collect normalized paths of all JSON files under the bfs_mapping directory.

    Args:
        bfs_mapping_dir: Path to the bfs_mapping directory

    Returns:
        Set of normalized file paths (last two levels only: target_state/filename)
        Example: {"SESSION_COMPLETED_SUCCESS/macro_002_00_HOME__SESSION_COMPLETED_SUCCESS.json"}
    """
    if not bfs_mapping_dir.exists():
        print(f"[ERROR] Directory does not exist: {bfs_mapping_dir}")
        return set()

    files = set()
    for json_file in bfs_mapping_dir.rglob("*.json"):
        try:
            rel_path = json_file.relative_to(bfs_mapping_dir)
            normalized = normalize_path(str(rel_path))
            files.add(normalized)
        except ValueError:
            continue

    return files


def filter_bfs_mapping(
    old_mapping: Path,
    new_mapping: Path,
    verbose: bool = True
) -> Tuple[int, int, List[str]]:
    if verbose:
        print(f"[INFO] Collecting file list from old bfs_mapping...")
        print(f"       Old directory: {old_mapping}")

    # Collect old file list (whitelist) - using normalized paths
    old_files = collect_file_paths(old_mapping)

    if not old_files:
        print(f"[ERROR] No files found in old bfs_mapping")
        return 0, 0, []

    if verbose:
        print(f"[INFO] Old bfs_mapping contains {len(old_files)} files")
        print(f"[INFO] Filtering new bfs_mapping...")
        print(f"       New directory: {new_mapping}")

    kept = 0
    removed = 0
    new_files_normalized = set()  # Collect normalized paths of new files

    # Iterate over newly generated files and delete those not in the whitelist
    for json_file in new_mapping.rglob("*.json"):
        try:
            rel_path = str(json_file.relative_to(new_mapping))
            normalized = normalize_path(rel_path)
            new_files_normalized.add(normalized)

            if normalized in old_files:
                # In the whitelist, keep it
                kept += 1
            else:
                # Not in the whitelist, delete it
                json_file.unlink()
                removed += 1
                if verbose and removed <= 5:
                    print(f"       Deleted: {rel_path}")
        except Exception as e:
            print(f"[WARNING] Failed to process file: {json_file} -> {e}")

    # Check which old files are missing from the new mapping
    missing = []
    for old_file in old_files:
        if old_file not in new_files_normalized:
            missing.append(old_file)

    # Print statistics
    print("")
    print("=" * 60)
    print("Filter Results Summary")
    print("=" * 60)
    print(f"✅ Files kept: {kept}")
    print(f"❌ Files deleted: {removed}")
    print(f"⚠️  Files missing: {len(missing)}")

    # Calculate missing rate
    if old_files:
        loss_rate = len(missing) / len(old_files) * 100
        print(f"📊 Missing rate: {loss_rate:.1f}%")

        if loss_rate > 50:
            print("")
            print("⚠️  Warning: missing rate exceeds 50%!")

    # Show missing files (first 20)
    if missing:
        print("")
        print(f"Missing files ({len(missing)} total):")
        for i, f in enumerate(missing[:20], 1):
            print(f"  {i}. {f}")

        if len(missing) > 20:
            print(f"  ... {len(missing) - 20} more files not shown")
    
    print("=" * 60)
    
    return kept, removed, missing


def main():
    parser = argparse.ArgumentParser(
        description="Filter BFS mapping, keeping only files that exist in the reference mapping"
    )
    parser.add_argument(
        "--old-mapping",
        type=Path,
        required=True,
        help="Old bfs_mapping directory (used as whitelist)"
    )
    parser.add_argument(
        "--new-mapping",
        type=Path,
        required=True,
        help="Newly generated bfs_mapping directory (will be filtered)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Quiet mode, show statistics only"
    )

    args = parser.parse_args()

    # Check that directories exist
    if not args.old_mapping.exists():
        print(f"[ERROR] Old bfs_mapping directory does not exist: {args.old_mapping}")
        sys.exit(1)

    if not args.new_mapping.exists():
        print(f"[ERROR] New bfs_mapping directory does not exist: {args.new_mapping}")
        sys.exit(1)

    # Run the filter
    kept, removed, missing = filter_bfs_mapping(
        args.old_mapping,
        args.new_mapping,
        verbose=not args.quiet
    )

    # Return exit code
    if len(missing) > 0:
        sys.exit(2)  # Missing files exist, but not considered an error
    else:
        sys.exit(0)  # Success


if __name__ == "__main__":
    main()

