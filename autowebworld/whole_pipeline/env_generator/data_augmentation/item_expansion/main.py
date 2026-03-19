#!/usr/bin/env python3
"""
Item Expansion - Command Line Interface.

Expands BFS trajectories to generate item-specific trajectories for all items in mockdata.

Usage:
    python -m item_expansion.main --input <bfs_dir> --mockdata <mockdata.json> --output <output_dir>

Examples:
    python -m item_expansion.main \
        --input ./bfs_generator/zillow_mapping/home_initial \
        --mockdata ./mockdata.json \
        --output ./expanded_trajectories
"""
import asyncio
import argparse
from pathlib import Path
from typing import List, Tuple
from tqdm.asyncio import tqdm as atqdm

from .utils import load_json, save_json
from .item_expander import ItemExpander
from .failure_logger import FailureLogger


def find_trajectory_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    # Find all macro_*.json files recursively
    files = list(input_path.rglob("macro_*.json"))
    return sorted(files)


async def process_file(
    expander: ItemExpander,
    traj_file: Path,
    input_path: Path,
    output_dir: Path
) -> Tuple[Path, int]:
    """Process a single trajectory file asynchronously."""
    try:
        # Create output subdirectory maintaining structure
        rel_path = traj_file.relative_to(input_path) if input_path.is_dir() else traj_file.name
        file_output_dir = output_dir / rel_path.parent / rel_path.stem

        generated = await expander.expand_file(traj_file, file_output_dir)
        return (traj_file, len(generated))
    except Exception as e:
        print(f"Error processing {traj_file}: {e}")
        return (traj_file, 0)


async def async_main(args) -> int:
    input_path = Path(args.input)
    mockdata_path = Path(args.mockdata)
    output_dir = Path(args.output)

    # Validate paths
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        return 1

    if not mockdata_path.exists():
        print(f"Error: Mockdata file does not exist: {mockdata_path}")
        return 1

    # Load mockdata
    print(f"Loading mockdata from {mockdata_path}...")
    mockdata = load_json(mockdata_path)

    # Initialize failure logger
    error_log_path = Path(args.error_log) if args.error_log else None
    failure_logger = FailureLogger(error_log_path=error_log_path)

    # Initialize expander
    expander = ItemExpander(
        mockdata=mockdata,
        model=args.model,
        failure_logger=failure_logger
    )

    # Find trajectory files
    trajectory_files = find_trajectory_files(input_path)
    print(f"Found {len(trajectory_files)} trajectory files")

    if not trajectory_files:
        print("No trajectory files found")
        return 1

    # Process files concurrently with semaphore to limit concurrency
    semaphore = asyncio.Semaphore(args.concurrency)

    async def process_with_semaphore(traj_file: Path) -> Tuple[Path, int]:
        async with semaphore:
            failure_logger.increment_total()
            result = await process_file(expander, traj_file, input_path, output_dir)
            if result[1] > 0:
                failure_logger.increment_successful()
            return result

    # Run all tasks concurrently with progress bar
    print(f"Processing {len(trajectory_files)} files with concurrency={args.concurrency}...")
    tasks = [process_with_semaphore(traj_file) for traj_file in trajectory_files]
    results = []
    for coro in atqdm.as_completed(tasks, total=len(tasks), desc="Expanding trajectories"):
        result = await coro
        results.append(result)

    total_generated = sum(count for _, count in results)

    failure_log_path = output_dir.parent / "expansion_failures.json"
    failure_logger.save(failure_log_path)

    usage_stats = expander.get_entity_detector_stats()
    cost_file = output_dir.parent / 'api_cost_stage2_expansion.json'
    save_json(usage_stats, cost_file)

    print(f"\n✅ Expansion complete!")
    print(f"   Processed: {len(trajectory_files)} trajectory files")
    print(f"   Generated: {total_generated} item-specific trajectories")
    print(f"   Output: {output_dir}")
    print(f"   Failures log: {failure_log_path}")
    print(f"\n{'='*60}")
    print(f"API Usage Statistics:")
    print(f"  Input tokens:  {usage_stats['input_tokens']:,}")
    print(f"  Output tokens: {usage_stats['output_tokens']:,}")
    print(f"  Total cost:    ${usage_stats['total_cost_usd']:.6f}")
    print(f"  Cost saved to: {cost_file}")
    print(f"{'='*60}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Expand BFS trajectories to item-specific trajectories'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to BFS trajectory file or directory'
    )
    parser.add_argument(
        '--mockdata', '-m',
        type=str,
        required=True,
        help='Path to mockdata.json file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output directory for expanded trajectories'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='deepseek-v3.2-exp',
        help='LLM model to use (default: deepseek-v3.2-exp)'
    )
    parser.add_argument(
        '--concurrency', '-c',
        type=int,
        default=10,
        help='Max concurrent LLM requests (default: 5)'
    )
    parser.add_argument(
        '--error-log',
        type=str,
        default=None,
        help='Path to unified error log file'
    )

    args = parser.parse_args()
    return asyncio.run(async_main(args))


if __name__ == '__main__':
    exit(main())

