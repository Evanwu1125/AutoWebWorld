"""
image_gen.py — Post-process image placeholders in generated web projects.

Scans JSX files for {{IMAGE:caption}} placeholders, concurrently generates
images via AsyncOpenAI chat.completions, saves them to public/images/, and
replaces placeholders with actual paths.
"""
from __future__ import annotations

import asyncio
import base64
import re
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI


_PLACEHOLDER_RE = re.compile(r"\{\{IMAGE:(.*?)\}\}")
_B64_IMAGE_RE = re.compile(r"data:image/\w+;base64,([A-Za-z0-9+/=\s]+)")


def extract_placeholders(web_dir: Path) -> list[dict[str, Any]]:
    """Scan all JSX files and extract image placeholders.

    Returns list of dicts: {file, placeholder, caption, img_name}
    """
    results = []
    counter = 0
    src_dir = web_dir / "src"
    if not src_dir.exists():
        return results

    for jsx_file in sorted(src_dir.rglob("*.jsx")):
        content = jsx_file.read_text(encoding="utf-8")
        for match in _PLACEHOLDER_RE.finditer(content):
            counter += 1
            results.append({
                "file": jsx_file,
                "placeholder": match.group(0),       # {{IMAGE:...}}
                "caption": match.group(1).strip(),    # the description
                "img_name": f"img_{counter:03d}.png",
            })
    return results


async def _generate_one(
    entry: dict,
    client: AsyncOpenAI,
    model: str,
    images_dir: Path,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Generate a single image via chat.completions and save it."""
    async with semaphore:
        output_path = images_dir / entry["img_name"]

        print(f"  [img] Generating: {entry['img_name']} — {entry['caption'][:60]}...")
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": f"Generate an image: {entry['caption']}. Return only the image, no text.",
                    }
                ],
            )

            message = response.choices[0].message
            saved = False
            content = message.content if isinstance(message.content, str) else ""

            # Format 1: structured multimodal parts
            if not saved and isinstance(message.content, list):
                for part in message.content:
                    if hasattr(part, "type") and part.type == "image":
                        b64_data = part.image.get("data") or part.image.get("b64_json")
                        if b64_data:
                            output_path.write_bytes(base64.b64decode(b64_data))
                            saved = True
                            break

            # Format 2: markdown inline base64 — ![image](data:image/png;base64,...)
            if not saved and content:
                m = _B64_IMAGE_RE.search(content)
                if m:
                    b64_data = m.group(1).replace("\n", "").replace(" ", "")
                    output_path.write_bytes(base64.b64decode(b64_data))
                    saved = True

            if saved:
                entry["success"] = True
                print(f"  [img] Saved: {entry['img_name']}")
            else:
                entry["success"] = False
                entry["error"] = "no image in response"
                print(f"  [img] No image found in response: {entry['img_name']}")

        except Exception as e:
            entry["success"] = False
            entry["error"] = str(e)
            print(f"  [img] Failed: {entry['img_name']} — {e}")

        return entry


def _replace_placeholders(entries: list[dict]) -> None:
    """Replace placeholders in JSX files with actual image paths."""
    by_file: dict[Path, list[dict]] = {}
    for entry in entries:
        if not entry.get("success"):
            continue
        by_file.setdefault(entry["file"], []).append(entry)

    for file_path, file_entries in by_file.items():
        content = file_path.read_text(encoding="utf-8")
        for entry in file_entries:
            content = content.replace(
                entry["placeholder"],
                f"/images/{entry['img_name']}",
            )
        file_path.write_text(content, encoding="utf-8")


async def process_images(
    web_dir: Path,
    api_key: str,
    base_url: str,
    model: str = "gemini-3.1-flash-image-preview",
    concurrency: int = 5,
) -> list[dict]:
    """Main entry: extract placeholders → generate images → replace.

    Args:
        web_dir: Path to the web project (e.g., fsm_outputs/{theme}/web).
        api_key: API key for the image generation service.
        base_url: Base URL (e.g., https://newapi.deepwisdom.ai/v1).
        model: Image generation model name.
        concurrency: Max concurrent API calls.

    Returns:
        List of entry dicts with success/error status.
    """
    entries = extract_placeholders(web_dir)
    if not entries:
        print("[image_gen] No image placeholders found.")
        return []

    print(f"[image_gen] Found {len(entries)} image placeholders.")

    # Ensure output directory exists
    images_dir = web_dir / "public" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Create shared AsyncOpenAI client
    client_kwargs: dict = {"api_key": api_key}
    if base_url:
        if not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        client_kwargs["base_url"] = base_url
    client = AsyncOpenAI(**client_kwargs)

    # Generate concurrently
    sem = asyncio.Semaphore(concurrency)
    tasks = [
        _generate_one(entry, client, model, images_dir, sem)
        for entry in entries
    ]
    results = await asyncio.gather(*tasks)

    # Replace placeholders
    _replace_placeholders(results)

    success = sum(1 for r in results if r.get("success"))
    print(f"[image_gen] Done. {success}/{len(results)} images generated successfully.")

    return results
