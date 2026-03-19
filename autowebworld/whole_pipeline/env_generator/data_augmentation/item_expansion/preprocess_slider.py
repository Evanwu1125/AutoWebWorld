#!/usr/bin/env python3
"""
Preprocess BFS trajectories to change slider direction from 'left' to 'right'.

This ensures all slider filters use 'greater than or equal' semantics.
"""
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Replace to_value: left with to_value: right in BFS trajectories'
    )
    parser.add_argument('--input', required=True, help='BFS mapping directory')
    args = parser.parse_args()

    input_dir = Path(args.input)
    modified_count = 0

    for json_file in input_dir.rglob('*.json'):
        content = json_file.read_text(encoding='utf-8')
        if '"to_value": "left"' in content:
            content = content.replace('"to_value": "left"', '"to_value": "right"')
            json_file.write_text(content, encoding='utf-8')
            print(f"Modified: {json_file}")
            modified_count += 1

    print(f"\nTotal modified: {modified_count} files")


if __name__ == '__main__':
    main()

