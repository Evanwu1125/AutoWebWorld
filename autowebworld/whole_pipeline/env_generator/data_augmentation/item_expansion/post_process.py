"""
Post-process expanded trajectories to generate queries and fill input_text.

This script:
1. Reads all expanded trajectory files
2. For each file, calls LLM to generate query + input_text_replacements
3. Builds slider from pre-calculated values (from main.py stage)
4. Updates the file with the generated content
"""
import asyncio
from pathlib import Path
import argparse
from tqdm.asyncio import tqdm as atqdm

from .query_generator import QueryGenerator
from .utils import load_json, save_json


def _has_item_id_param(params: dict) -> bool:
    if not params:
        return False
    return any(key.endswith('_id') for key in params.keys())


async def process_file(
    file_path: Path,
    query_generator: QueryGenerator,
    mockdata: dict
) -> bool:
    """
    Process a single expanded trajectory file.

    Args:
        file_path: Path to expanded trajectory JSON
        query_generator: QueryGenerator instance
        mockdata: Complete mockdata dictionary

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load expanded trajectory
        expanded = load_json(file_path)

        # Generate query and input_text_replacements
        result = await query_generator.generate_query(expanded)

        if 'error' in result:
            print(f"Error processing {file_path.name}: {result['error']}")
            return False

        # Update expanded trajectory with generated query
        expanded['query'] = result.get('query', '')

        # Build slider from pre-calculated values (from main.py stage)
        expanded['slider'] = {
            'target_value': expanded.get('target_value'),
            'direction': 'right',  # Fixed: all sliders use 'right'
            'sort_field': expanded.get('filter_field'),
            'sort_order': expanded.get('sort_order'),
            'rank': expanded.get('rank', 1)
        }

        # Fill input_text in trajectory
        _fill_input_text(expanded, result)

        # Save updated file
        save_json(expanded, file_path)
        print(f"✓ Processed: {file_path.name}")
        return True

    except Exception as e:
        print(f"✗ Failed to process {file_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def _fill_input_text(expanded, result):
    """Fill input_text placeholders in trajectory."""
    trajectory = expanded.get('trajectory', [])
    input_text_replacements = result.get('input_text_replacements', [])
    item = expanded.get('item')
    trajectory_type = expanded.get('trajectory_type')

    # Handle None item (for NO_ITEM trajectory type)
    if item is None:
        item = {}
    item_name = item.get('name', '')

    # Build replacement map
    replacement_map = {}
    for replacement in input_text_replacements:
        action_id = replacement.get('action_id', '')
        value = replacement.get('value', '')
        # Replace {ITEM_NAME} in value if present
        value = value.replace('{ITEM_NAME}', item_name)
        replacement_map[action_id] = value

    # Apply replacements to trajectory
    for action in trajectory:
        action_id = action.get('id', '')

        if action_id in replacement_map:
            replacement_value = replacement_map[action_id]
            gui_procedure = action.get('gui_procedure', [])
            for step in gui_procedure:
                if step.get('text') == '{input_text}':
                    step['text'] = replacement_value


async def main():
    parser = argparse.ArgumentParser(description='Post-process expanded trajectories')
    parser.add_argument('--input', required=True, help='Expanded trajectories directory')
    parser.add_argument('--mockdata', required=True, help='Mockdata JSON file')
    parser.add_argument('--model', default='deepseek-v3.2-exp', help='LLM model to use')
    parser.add_argument('--concurrency', type=int, default=20, help='Number of concurrent LLM requests (default: 20)')
    parser.add_argument('--error-log', default=None, help='Path to unified error log file')
    args = parser.parse_args()

    input_dir = Path(args.input)
    mockdata_path = Path(args.mockdata)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return

    if not mockdata_path.exists():
        print(f"Error: Mockdata file not found: {mockdata_path}")
        return

    # Load mockdata
    mockdata = load_json(mockdata_path)
    print(f"Loaded mockdata with {len(mockdata)} keys")

    # Initialize query generator
    query_generator = QueryGenerator(mockdata, model=args.model)

    # Find all expanded trajectory files
    json_files = list(input_dir.rglob('*.json'))
    print(f"Found {len(json_files)} expanded trajectory files")
    print(f"Using concurrency: {args.concurrency}")

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(args.concurrency)

    async def process_with_semaphore(file_path: Path) -> bool:
        """Process file with semaphore to limit concurrency."""
        async with semaphore:
            return await process_file(file_path, query_generator, mockdata)

    # Process files concurrently with progress bar
    print(f"\nProcessing files with {args.concurrency} concurrent requests...")
    tasks = [process_with_semaphore(json_file) for json_file in json_files]
    results = []
    for coro in atqdm.as_completed(tasks, total=len(tasks), desc="Generating queries"):
        try:
            result = await coro
            results.append(result)
        except Exception as e:
            results.append(e)

    # Count successes (handle exceptions from gather)
    success_count = sum(1 for r in results if r is True)
    error_count = sum(1 for r in results if isinstance(r, Exception))

    if error_count > 0:
        print(f"\nWarning: {error_count} files failed with exceptions")

    print(f"\n{'='*60}")
    print(f"Post-processing complete: {success_count}/{len(json_files)} files processed")
    print(f"{'='*60}")

    usage_stats = query_generator.get_usage_stats()
    workspace_dir = input_dir.parent
    cost_file = workspace_dir / 'api_cost_stage2_postprocess.json'
    save_json(usage_stats, cost_file)

    print(f"\n{'='*60}")
    print(f"API Usage Statistics:")
    print(f"  Input tokens:  {usage_stats['input_tokens']:,}")
    print(f"  Output tokens: {usage_stats['output_tokens']:,}")
    print(f"  Total tokens:  {usage_stats['total_tokens']:,}")
    print(f"  Input cost:    ${usage_stats['input_cost_usd']:.6f}")
    print(f"  Output cost:   ${usage_stats['output_cost_usd']:.6f}")
    print(f"  Total cost:    ${usage_stats['total_cost_usd']:.6f}")
    print(f"\nCost saved to: {cost_file}")
    print(f"{'='*60}")


if __name__ == '__main__':
    asyncio.run(main())

