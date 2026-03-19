"""Main entry point for QA Query Generation"""
import asyncio
import argparse
from pathlib import Path

from .qa_generator import QAGenerator
from .config import (
    API_KEY, BASE_URL,
    DEFAULT_LLM_MODEL, DEFAULT_VLM_MODEL,
    DEFAULT_MAX_QUESTIONS, DEFAULT_CONCURRENCY
)


async def async_main(args):
    if not API_KEY:
        print("Error: OPENAI_API_KEY environment variable not set")
        return 1
    
    grounding_dir = Path(args.grounding_dir)
    caption_file = Path(args.caption_file)
    output_dir = Path(args.output_dir)
    
    if not grounding_dir.exists():
        print(f"Error: Grounding directory not found: {grounding_dir}")
        return 1
    
    if not caption_file.exists():
        print(f"Error: Caption file not found: {caption_file}")
        return 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generator = QAGenerator(
        api_key=API_KEY,
        base_url=BASE_URL,
        llm_model=args.llm_model,
        vlm_model=args.vlm_model,
        max_questions=args.max_questions,
        concurrency=args.concurrency,
        skip_vlm=args.skip_vlm
    )
    
    await generator.process_project(
        grounding_dir=grounding_dir,
        caption_file=caption_file,
        output_dir=output_dir
    )
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Generate QA queries from grounding results'
    )
    parser.add_argument(
        '--grounding-dir',
        required=True,
        help='Path to grounding results directory (2_grounding)'
    )
    parser.add_argument(
        '--caption-file',
        required=True,
        help='Path to caption.json file'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for QA queries'
    )
    parser.add_argument(
        '--llm-model',
        default=DEFAULT_LLM_MODEL,
        help=f'LLM model for question generation (default: {DEFAULT_LLM_MODEL})'
    )
    parser.add_argument(
        '--vlm-model',
        default=DEFAULT_VLM_MODEL,
        help=f'VLM model for verification (default: {DEFAULT_VLM_MODEL})'
    )
    parser.add_argument(
        '--max-questions',
        type=int,
        default=DEFAULT_MAX_QUESTIONS,
        help=f'Maximum questions per item (default: {DEFAULT_MAX_QUESTIONS})'
    )
    parser.add_argument(
        '--concurrency',
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f'Number of concurrent requests (default: {DEFAULT_CONCURRENCY})'
    )
    parser.add_argument(
        '--skip-vlm',
        action='store_true',
        help='Skip VLM verification (for debugging)'
    )
    
    args = parser.parse_args()
    return asyncio.run(async_main(args))


if __name__ == '__main__':
    exit(main())

