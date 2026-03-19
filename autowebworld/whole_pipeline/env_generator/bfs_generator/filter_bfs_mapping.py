#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path
from typing import Set, Tuple, List, Dict


def normalize_path(path: str) -> str:
    parts = Path(path).parts
    if len(parts) >= 2:
        return str(Path(parts[-2]) / parts[-1])
    return path


def collect_file_paths(bfs_mapping_dir: Path) -> Set[str]:
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
    new_files_normalized = set()

    for json_file in new_mapping.rglob("*.json"):
        try:
            rel_path = str(json_file.relative_to(new_mapping))
            normalized = normalize_path(rel_path)
            new_files_normalized.add(normalized)

            if normalized in old_files:
                kept += 1
            else:
                json_file.unlink()
                removed += 1
                if verbose and removed <= 5:
                    print(f"       Removed: {rel_path}")
        except Exception as e:
            print(f"[WARNING] Failed to process file: {json_file} -> {e}")

    missing = []
    for old_file in old_files:
        if old_file not in new_files_normalized:
            missing.append(old_file)
    
    print("")
    print("=" * 60)
    print("Filtering summary")
    print("=" * 60)
    print(f"✅ Kept files: {kept}")
    print(f"❌ Removed files: {removed}")
    print(f"⚠️  Missing files: {len(missing)}")
    
    if old_files:
        loss_rate = len(missing) / len(old_files) * 100
        print(f"📊 Loss rate: {loss_rate:.1f}%")
        
        if loss_rate > 50:
            print("")
            print("⚠️  Warning: loss rate exceeds 50%!")
    
    if missing:
        print("")
        print(f"Missing files list (total {len(missing)}):")
        for i, f in enumerate(missing[:20], 1):
            print(f"  {i}. {f}")
        
        if len(missing) > 20:
            print(f"  ... {len(missing) - 20} more files not shown")
    
    print("=" * 60)
    
    return kept, removed, missing


def main():
    parser = argparse.ArgumentParser(
        description="Filter BFS mapping and keep only files present in the reference mapping"
    )
    parser.add_argument(
        "--old-mapping",
        type=Path,
        required=True,
        help="Old bfs_mapping directory (used as a whitelist)"
    )
    parser.add_argument(
        "--new-mapping",
        type=Path,
        required=True,
        help="Newly generated bfs_mapping directory (to be filtered)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Quiet mode; show only summary statistics"
    )
    
    args = parser.parse_args()
    
    if not args.old_mapping.exists():
        print(f"[ERROR] Old bfs_mapping directory does not exist: {args.old_mapping}")
        sys.exit(1)
    
    if not args.new_mapping.exists():
        print(f"[ERROR] New bfs_mapping directory does not exist: {args.new_mapping}")
        sys.exit(1)
    
    kept, removed, missing = filter_bfs_mapping(
        args.old_mapping,
        args.new_mapping,
        verbose=not args.quiet
    )
    
    if len(missing) > 0:
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

