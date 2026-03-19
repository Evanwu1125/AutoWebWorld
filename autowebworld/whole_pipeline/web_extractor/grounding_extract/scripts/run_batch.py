"""Batch grounding extraction script."""

import asyncio
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from batch.batch_runner import run_batch


async def main():
    parser = argparse.ArgumentParser(
        description='Batch BFS Grounding Extraction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-start Vue dev server (default)
  python -m scripts.run_batch input_folder web_dir

  # Use existing server
  python -m scripts.run_batch input_folder web_dir --url http://localhost:5173
        """
    )
    parser.add_argument('input_folder', help='Input folder containing JSON files')
    parser.add_argument('web_dir', help='Path to web project directory')
    parser.add_argument('--url', default=None, help='Web service URL (if not provided, auto-start Vue dev server)')
    parser.add_argument('--output-dir', default='batch_outputs', help='Output directory')
    parser.add_argument('--summary-path', default=None, help='Path to summary.json file containing slider metadata')
    parser.add_argument('--max-workers', type=int, default=3, help='Maximum concurrent workers')
    parser.add_argument('--scroll-step', type=int, default=100, help='Scroll step size in pixels')
    parser.add_argument('--num-slider-steps', type=int, default=3, help='Number of intermediate steps for slider drag')
    parser.add_argument('--headless', action='store_true', help='Run browser in headless mode')
    parser.add_argument('--viewport-width', type=int, default=1280, help='Browser viewport width')
    parser.add_argument('--viewport-height', type=int, default=720, help='Browser viewport height')
    parser.add_argument('--max-scroll-iterations', type=int, default=50, help='Maximum scroll iterations (default: 50)')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    parser.add_argument('--no-verbose', action='store_true', help='Disable detailed output and show only the progress bar and key information')

    args = parser.parse_args()

    verbose = not (args.no_verbose or args.quiet)

    await run_batch(
        input_folder=args.input_folder,
        web_dir=args.web_dir,
        base_url=args.url,
        output_dir=args.output_dir,
        summary_path=args.summary_path,
        max_workers=args.max_workers,
        scroll_step=args.scroll_step,
        num_slider_steps=args.num_slider_steps,
        headless=args.headless,
        viewport_width=args.viewport_width,
        viewport_height=args.viewport_height,
        verbose=verbose,
        max_scroll_iterations=args.max_scroll_iterations
    )


if __name__ == '__main__':
    asyncio.run(main())

