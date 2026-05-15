"""
test_image_gen.py — Test OpenAI SDK async image generation with concurrency.

Uses chat.completions to generate images via gemini-2.5-flash-image-preview,
which returns images as base64 in the response content.

Usage:
    cd whole_pipeline
    python test_image_gen.py
"""
import asyncio
import base64
import os
import re
import time
from pathlib import Path

from openai import AsyncOpenAI

BASE_URL = "https://newapi.deepwisdom.ai/v1"
MODEL = "gemini-3.1-flash-image-preview"

OUTPUT_DIR = Path(__file__).parent / "test_images"

PROMPTS = [
    "A cozy Airbnb living room with modern furniture and warm lighting",
    "A tropical beach villa with an infinity pool overlooking the ocean at sunset",
    "A bustling city street cafe with outdoor seating and string lights at night",
]


async def generate_one(client: AsyncOpenAI, prompt: str, index: int, semaphore: asyncio.Semaphore) -> dict:
    """Generate a single image using chat.completions."""
    async with semaphore:
        img_name = f"test_{index:03d}.png"
        print(f"  [{img_name}] Starting: {prompt[:50]}...")
        t0 = time.time()

        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": f"Generate an image: {prompt}. Return only the image, no text.",
                    }
                ],
            )

            # Extract base64 image from response content
            message = response.choices[0].message
            saved = False
            content = message.content if isinstance(message.content, str) else ""

            # Format 1: structured multimodal parts
            if not saved and isinstance(message.content, list):
                for part in message.content:
                    if hasattr(part, "type") and part.type == "image":
                        b64_data = part.image.get("data") or part.image.get("b64_json")
                        if b64_data:
                            output_path = OUTPUT_DIR / img_name
                            output_path.write_bytes(base64.b64decode(b64_data))
                            saved = True
                            break

            # Format 2: markdown inline base64 — ![image](data:image/png;base64,...)
            if not saved and content:
                m = re.search(r"data:image/\w+;base64,([A-Za-z0-9+/=\s]+)", content)
                if m:
                    b64_data = m.group(1).replace("\n", "").replace(" ", "")
                    output_path = OUTPUT_DIR / img_name
                    output_path.write_bytes(base64.b64decode(b64_data))
                    saved = True

            elapsed = time.time() - t0
            if saved:
                print(f"  [{img_name}] Done in {elapsed:.1f}s")
                return {"img_name": img_name, "success": True, "elapsed": elapsed}
            else:
                print(f"  [{img_name}] No image found in response ({elapsed:.1f}s)")
                return {"img_name": img_name, "success": False, "error": "no image in response", "elapsed": elapsed}

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [{img_name}] Failed in {elapsed:.1f}s: {e}")
            return {"img_name": img_name, "success": False, "error": str(e), "elapsed": elapsed}


async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = AsyncOpenAI(api_key=api_key, base_url=BASE_URL)
    semaphore = asyncio.Semaphore(3)

    print(f"Testing {len(PROMPTS)} concurrent image generations...")
    print(f"Model: {MODEL}, Base URL: {BASE_URL}")
    print()

    t0 = time.time()
    tasks = [
        generate_one(client, prompt, i + 1, semaphore)
        for i, prompt in enumerate(PROMPTS)
    ]
    results = await asyncio.gather(*tasks)
    total = time.time() - t0

    print()
    print(f"Results:")
    for r in results:
        status = "OK" if r["success"] else f"FAIL: {r.get('error', '')}"
        print(f"  {r['img_name']}: {status} ({r['elapsed']:.1f}s)")

    success = sum(1 for r in results if r["success"])
    print(f"\n{success}/{len(results)} succeeded, total time: {total:.1f}s")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
