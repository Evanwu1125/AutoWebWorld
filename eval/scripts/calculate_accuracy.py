#!/usr/bin/env python3
"""
Calculate accuracy for WebVoyager evaluation results.

Usage:
    python calculate_accuracy.py <run_directory_path>

Example:
    python calculate_accuracy.py output/webvoyager_batch/gemini-2.5-flash/Apple/Apple_20260118_192502
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple


def find_score_files(run_dir: Path) -> List[Tuple[str, Path]]:
    """
    Find all score.json files under the run directory.

    Expected directory structure:
        run_dir/
          ├── {task_id}/
          │   ├── score.json
          │   ├── result.json
          │   └── ...

    Args:
        run_dir: Path to the run directory

    Returns:
        List of (task_id, score_path) tuples
    """
    score_files = []

    if not run_dir.exists():
        print(f"Error: directory does not exist: {run_dir}")
        return score_files

    # Iterate over all sub-directories in the run directory
    for task_dir in sorted(run_dir.iterdir()):
        if not task_dir.is_dir():
            continue

        # Skip special directories
        if task_dir.name in ["artifacts", "results", "__pycache__"]:
            continue

        task_id = task_dir.name
        score_path = task_dir / "score.json"

        score_files.append((task_id, score_path))

    return score_files


def read_score(score_path: Path) -> float:
    """
    Read the score field from a score.json file.
    
    Args:
        score_path: Path to the score.json file
        
    Returns:
        score value, or -1 if the file is missing or unreadable
    """
    if not score_path.exists():
        return -1
    
    try:
        with open(score_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('score', -1)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: failed to read {score_path}: {e}")
        return -1


def calculate_accuracy(run_dir: str, silent: bool = False) -> Dict:
    """
    Calculate task success accuracy for a run directory.

    Args:
        run_dir: Path to the run directory
        silent: If True, suppress detailed print output

    Returns:
        Dict containing accuracy statistics
    """
    run_path = Path(run_dir)

    if not run_path.exists():
        if not silent:
            print(f"Error: run directory not found: {run_path}")
        return {
            "run_directory": str(run_path),
            "total_tasks": 0,
            "success_count": 0,
            "fail_count": 0,
            "not_found_count": 0,
            "accuracy": 0.0,
            "task_details": [],
            "error": f"run directory not found: {run_path}"
        }

    # Locate all score.json files
    score_files = find_score_files(run_path)

    if not score_files:
        if not silent:
            print(f"Error: no task folders found under {run_path}")
        return {
            "run_directory": str(run_path),
            "total_tasks": 0,
            "success_count": 0,
            "fail_count": 0,
            "not_found_count": 0,
            "accuracy": 0.0,
            "task_details": [],
            "error": f"No task folders found in {run_path}"
        }

    # Tally results
    total_tasks = len(score_files)
    success_count = 0
    fail_count = 0
    not_found_count = 0

    task_details = []

    if not silent:
        print(f"\n{'='*80}")
        print(f"Calculating accuracy")
        print(f"{'='*80}")
        print(f"Run directory: {run_dir}")
        print(f"Total tasks: {total_tasks}")
        print(f"{'='*80}\n")
    
    for task_id, score_path in score_files:
        score = read_score(score_path)
        
        if score == -1:
            status = "NOT_FOUND"
            not_found_count += 1
            symbol = "?"
        elif score == 1.0:
            status = "SUCCESS"
            success_count += 1
            symbol = "+"
        elif score == 0.0:
            status = "FAIL"
            fail_count += 1
            symbol = "-"
        else:
            status = f"UNKNOWN({score})"
            fail_count += 1
            symbol = "!"
        
        task_details.append({
            "task_id": task_id,
            "score": score,
            "status": status,
            "score_path": str(score_path)
        })

        if not silent:
            print(f"[{symbol}] {task_id}: {status} (score={score})")

    # Compute accuracy
    accuracy = (success_count / total_tasks * 100) if total_tasks > 0 else 0.0

    metrics = {
        "run_directory": str(run_path),
        "total_tasks": total_tasks,
        "success_count": success_count,
        "fail_count": fail_count,
        "not_found_count": not_found_count,
        "accuracy": round(accuracy, 2),
        "task_details": task_details
    }

    if not silent:
        print(f"\n{'='*80}")
        print(f"Results")
        print(f"{'='*80}")
        print(f"Total tasks:     {total_tasks}")
        print(f"Success:         {success_count} ({success_count/total_tasks*100:.1f}%)")
        print(f"Fail:            {fail_count} ({fail_count/total_tasks*100:.1f}%)")
        print(f"Not found:       {not_found_count} ({not_found_count/total_tasks*100:.1f}%)")
        print(f"{'='*80}")
        print(f"Accuracy:        {accuracy:.2f}%")
        print(f"{'='*80}\n")

    return metrics


def save_metrics(metrics: Dict, output_path: Path, silent: bool = False):
    """
    Save metrics dict to a JSON file.

    Args:
        metrics: Metrics dictionary to save
        output_path: Output file path
        silent: If True, suppress print output
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    if not silent:
        print(f"Metrics saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python calculate_accuracy.py <run_directory_path>")
        print("\nExample:")
        print("  python calculate_accuracy.py output/webvoyager_batch/gemini-2.5-flash/Apple/Apple_20260118_192502")
        sys.exit(1)

    run_dir = sys.argv[1]

    # Calculate accuracy
    metrics = calculate_accuracy(run_dir)

    # Save to my_metrics.json
    output_path = Path(run_dir) / "my_metrics.json"
    save_metrics(metrics, output_path)

    print(f"\nDone!")
