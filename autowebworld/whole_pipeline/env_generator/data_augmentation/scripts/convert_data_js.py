#!/usr/bin/env python3
"""
Convert data.js (Vue/Pinia store format) to data.json.

This script parses JavaScript data files and extracts the data into JSON format.
It handles common patterns used in Vue/Pinia stores.

Usage:
    python convert_data_js.py <input_data_js> <output_data_json>
    python convert_data_js.py /path/to/src/data.js /path/to/mockdata.json
"""
import re
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional


def extract_js_arrays(js_content: str) -> Dict[str, Any]:
    """
    Extract JavaScript arrays/objects from data.js content.

    Handles patterns like:
        Composition API: const users = ref([...])
        Options API: state: () => ({ users: [...], ... })

    Args:
        js_content: JavaScript file content

    Returns:
        Dictionary with extracted data
    """
    result = {}

    # Pattern 1: Composition API - const {name} = ref([...])
    pattern1 = r'const\s+(\w+)\s*=\s*ref\s*\(\s*(\[[\s\S]*?\])\s*\)'
    matches1 = re.findall(pattern1, js_content)

    for name, array_str in matches1:
        try:
            # Convert JavaScript syntax to JSON
            json_str = js_to_json(array_str)
            data = json.loads(json_str)
            result[name] = data
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse {name} (Composition API): {e}")
            continue

    # Pattern 2: Options API - state: () => ({ ... })
    # Extract the entire state object
    pattern2 = r'state:\s*\(\)\s*=>\s*\(\s*\{([\s\S]*?)\}\s*\)\s*[,}]'
    state_match = re.search(pattern2, js_content)

    if state_match:
        state_content = state_match.group(1)
        # Extract individual fields using bracket counting for proper nesting
        result.update(extract_fields_from_state(state_content))

    return result


def extract_fields_from_state(state_content: str) -> Dict[str, Any]:
    """
    Extract fields from Pinia Options API state object.

    Handles nested arrays and objects by counting brackets.

    Args:
        state_content: Content inside state: () => ({ ... })

    Returns:
        Dictionary with extracted fields
    """
    result = {}
    i = 0
    while i < len(state_content):
        # Skip whitespace and comments
        while i < len(state_content) and state_content[i] in ' \t\n\r':
            i += 1
        if i >= len(state_content):
            break

        # Skip comments
        if i < len(state_content) - 1 and state_content[i:i+2] == '//':
            # Skip to end of line
            while i < len(state_content) and state_content[i] != '\n':
                i += 1
            continue

        # Match field name
        field_match = re.match(r'(\w+)\s*:', state_content[i:])
        if not field_match:
            i += 1
            continue

        field_name = field_match.group(1)
        i += field_match.end()

        # Skip whitespace
        while i < len(state_content) and state_content[i] in ' \t\n\r':
            i += 1

        # Check if this is an array
        if i < len(state_content) and state_content[i] == '[':
            # Find matching closing bracket
            bracket_count = 0
            start = i
            while i < len(state_content):
                if state_content[i] == '[':
                    bracket_count += 1
                elif state_content[i] == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        i += 1
                        break
                i += 1

            array_str = state_content[start:i]

            try:
                # Convert JavaScript syntax to JSON
                json_str = js_to_json(array_str)
                data = json.loads(json_str)
                result[field_name] = data
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse {field_name} (Options API): {e}")

            # Skip comma if present
            while i < len(state_content) and state_content[i] in ' \t\n\r,':
                i += 1
        else:
            # Not an array, skip to next comma or end
            while i < len(state_content) and state_content[i] not in ',}':
                i += 1
            if i < len(state_content) and state_content[i] == ',':
                i += 1

    return result


def js_to_json(js_str: str) -> str:
    """
    Convert JavaScript object/array syntax to valid JSON.

    Handles:
        - Unquoted keys: {key: value} -> {"key": value}
        - Single quotes: 'value' -> "value"
        - Template literals: `value` -> "value"
        - Trailing commas
        - JavaScript booleans (true/false are same in JSON)

    Args:
        js_str: JavaScript array/object string

    Returns:
        Valid JSON string
    """
    # Remove JavaScript comments (but NOT // in URLs!)
    # Only remove // comments that are NOT inside strings
    # We'll do this more carefully after handling quotes
    js_str = re.sub(r'/\*[\s\S]*?\*/', '', js_str)  # Remove /* */ comments first

    # Handle template literals (backticks) FIRST - convert to regular strings
    # Need to escape any double quotes inside the backticks
    def replace_backticks(match):
        content = match.group(1)
        # Escape any double quotes inside the backtick content
        content = content.replace('"', '\\"')
        return f'"{content}"'

    js_str = re.sub(r'`([^`]*)`', replace_backticks, js_str)

    # Replace single quotes with double quotes
    # Use a more careful approach to avoid breaking escaped quotes
    result = []
    i = 0
    while i < len(js_str):
        if js_str[i] == "'":
            # Found a single quote, replace with double quote
            result.append('"')
            i += 1
            # Copy content until closing quote
            while i < len(js_str) and js_str[i] != "'":
                if js_str[i] == '\\' and i + 1 < len(js_str):
                    next_char = js_str[i + 1]
                    if next_char == "'":
                        # \' in single-quoted string becomes just ' in double-quoted string
                        # because single quotes don't need escaping in JSON double-quoted strings
                        result.append("'")
                        i += 2
                    else:
                        # Other escape sequences (\n, \t, \\, \", etc.) remain unchanged
                        result.append(js_str[i:i+2])
                        i += 2
                elif js_str[i] == '"':
                    # Escape double quotes inside the string
                    result.append('\\"')
                    i += 1
                else:
                    result.append(js_str[i])
                    i += 1
            if i < len(js_str):
                result.append('"')
                i += 1
        else:
            result.append(js_str[i])
            i += 1

    js_str = ''.join(result)

    # Now remove // comments (after handling quotes, so URLs are safe)
    # Only remove // that appear outside of strings
    # Since all strings are now double-quoted, we can safely remove // comments
    lines = js_str.split('\n')
    cleaned_lines = []
    for line in lines:
        # Find // that is NOT inside a string
        in_string = False
        escape_next = False
        comment_pos = -1

        for i, char in enumerate(line):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"':
                in_string = not in_string
            elif char == '/' and i + 1 < len(line) and line[i + 1] == '/' and not in_string:
                comment_pos = i
                break

        if comment_pos >= 0:
            cleaned_lines.append(line[:comment_pos])
        else:
            cleaned_lines.append(line)

    js_str = '\n'.join(cleaned_lines)

    # Quote unquoted keys: {key: value} -> {"key": value}
    # This regex now only matches keys that are NOT inside strings
    # We do this by matching keys that come after { or , followed by optional whitespace
    js_str = re.sub(r'(?<=[{,])\s*(\w+)\s*:', r' "\1":', js_str)

    # Remove trailing commas before } or ]
    js_str = re.sub(r',\s*([}\]])', r'\1', js_str)

    return js_str


def convert_data_js_to_json(input_path: Path, output_path: Path) -> Dict[str, Any]:
    """
    Convert data.js file to data.json.
    
    Args:
        input_path: Path to input data.js file
        output_path: Path to output data.json file
        
    Returns:
        Extracted data dictionary
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Read JavaScript file
    with open(input_path, 'r', encoding='utf-8') as f:
        js_content = f.read()
    
    # Extract data
    data = extract_js_arrays(js_content)
    
    if not data:
        print("Warning: No data arrays found in the input file.")
        print("Expected formats:")
        print("  - Composition API: const arrayName = ref([...])")
        print("  - Options API: state: () => ({ arrayName: [...] })")
    
    # Write JSON output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Converted: {input_path} -> {output_path}")
    print(f"📊 Data statistics:")
    for key, value in data.items():
        if isinstance(value, list):
            print(f"   - {key}: {len(value)} items")
        else:
            print(f"   - {key}: {type(value).__name__}")
    
    return data


def main():
    parser = argparse.ArgumentParser(
        description='Convert data.js (Vue/Pinia store) to data.json'
    )
    parser.add_argument(
        'input',
        type=str,
        help='Path to input data.js file'
    )
    parser.add_argument(
        'output',
        type=str,
        help='Path to output data.json file'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    convert_data_js_to_json(input_path, output_path)


if __name__ == '__main__':
    main()

