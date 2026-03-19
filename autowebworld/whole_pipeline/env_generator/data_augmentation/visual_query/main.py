import json
import asyncio
import base64
import re
import shutil
import io
from pathlib import Path
from typing import Dict, Any, Optional
from openai import AsyncOpenAI
from PIL import Image

from .caption_generator import generate_captions, get_caption_usage_stats
from .utils import load_data, save_data
from .prompt.visual_query_prompt import get_visual_query_prompt
from . import config


class LLMConfig:
    def __init__(self, model: str, api_key: str, base_url: str, temperature: float = 0.7):
        self.model = model
        self.key = api_key
        self.base_url = base_url
        self.temperature = temperature


class AsyncLLM:

    def __init__(self, llm_config):
        self.config = llm_config
        self.aclient = AsyncOpenAI(api_key=self.config.key, base_url=self.config.base_url)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

    async def __call__(self, prompt):
        message = []

        if isinstance(prompt, str):
            message.append({"role": "user", "content": prompt})
        elif isinstance(prompt, list):
            message.append({"role": "user", "content": prompt})
        else:
            raise ValueError(f"prompt must be str or list, got {type(prompt)}")

        response = await self.aclient.chat.completions.create(
            model=self.config.model,
            messages=message,
            temperature=self.config.temperature,
        )

        if hasattr(response, 'usage') and response.usage:
            self.total_input_tokens += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens

        ret = response.choices[0].message.content
        return ret

    async def generate_text_to_image(self, prompt: str) -> dict:
        try:
            response = await self(prompt)
            image_b64 = self._extract_image_from_response(response)

            if not image_b64:
                return {
                    "success": False,
                    "image_base64": None,
                    "prompt": prompt,
                    "error": "No image found in response",
                }

            return {
                "success": True,
                "image_base64": image_b64,
                "prompt": prompt,
                "error": None,
            }
        except Exception as e:
            return {
                "success": False,
                "image_base64": None,
                "prompt": prompt,
                "error": str(e),
            }

    def _extract_image_from_response(self, response: str) -> Optional[str]:
        if not isinstance(response, str):
            return None
        match = re.search(r"data:image/[^;]+;base64,([^)]+)", response)
        return match.group(1) if match else None

    def get_usage_summary(self):
        input_price_per_m = 0.3
        output_price_per_m = 30.0

        input_cost = (self.total_input_tokens / 1_000_000) * input_price_per_m
        output_cost = (self.total_output_tokens / 1_000_000) * output_price_per_m
        total_cost = input_cost + output_cost

        return {
            "model": self.config.model,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost": round(total_cost, 6),
            "price_per_million": {
                "input": input_price_per_m,
                "output": output_price_per_m
            }
        }


def generate_caption(mockdata: str, output_path: str = 'caption.txt', model: str = None):
    mockdata = load_data(mockdata)
    result = generate_captions(mockdata, model=model)
    save_data(result, output_path)
    return result


async def generate_images(caption_data: dict, output_dir: str, image_model: str = "gemini-2.5-flash-image", max_concurrent: int = 20):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    llm_config = LLMConfig(
        model=image_model,
        api_key=config.API_KEY,
        base_url=config.BASE_URL,
        temperature=config.TEMPERATURE
    )
    llm = AsyncLLM(llm_config)

    tasks = []
    for collection_name, items in caption_data.items():
        if isinstance(items, list):
            for item in items:
                if 'caption' in item and 'id' in item:
                    tasks.append(_generate_single_image(llm, item['id'], item['caption'], output_dir, image_model))

    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_task(task):
        async with semaphore:
            return await task

    results = await asyncio.gather(*[bounded_task(t) for t in tasks])
    success_count = len([r for r in results if r])
    print(f"Generated {success_count}/{len(tasks)} images")
    return results


async def _generate_single_image(llm, item_id: str, caption: str, output_dir: str, model: str):
    try:
        result = await llm.generate_text_to_image(caption)

        if not result["success"]:
            print(f"Failed to generate image for {item_id}: {result['error']}")
            return None

        image_data = base64.b64decode(result["image_base64"])

        img = Image.open(io.BytesIO(image_data))
        img_resized = img.resize(
            (config.IMAGE_RESIZE_WIDTH, config.IMAGE_RESIZE_HEIGHT),
            Image.Resampling.LANCZOS
        )

        output_path = Path(output_dir) / f"{item_id}.jpg"
        img_resized.save(output_path, quality=config.IMAGE_QUALITY, optimize=True)

        print(f"Generated {item_id}.jpg (resized to {config.IMAGE_RESIZE_WIDTH}x{config.IMAGE_RESIZE_HEIGHT})")
        return str(output_path)

    except Exception as e:
        print(f"Failed to generate image for {item_id}: {e}")
        return None


_image_llm_instance = None
_vq_total_input_tokens = 0
_vq_total_output_tokens = 0

async def generate_images_with_category(caption_data: dict, output_dir: str, image_model: str = "gemini-2.5-flash-image", max_concurrent: int = 20):
    global _image_llm_instance
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    llm_config = LLMConfig(
        model=image_model,
        api_key=config.API_KEY,
        base_url=config.BASE_URL,
        temperature=config.TEMPERATURE
    )
    llm = AsyncLLM(llm_config)
    _image_llm_instance = llm

    tasks = []
    for collection_name, items in caption_data.items():
        if isinstance(items, list):
            for item in items:
                if 'caption' in item and 'id' in item:
                    image_name = f"{collection_name}_{item['id']}.jpg"
                    tasks.append(_generate_single_image_with_name(
                        llm, image_name, item['caption'], output_dir, image_model
                    ))

    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_task(task):
        async with semaphore:
            return await task

    results = await asyncio.gather(*[bounded_task(t) for t in tasks])
    success_count = len([r for r in results if r])
    print(f"Generated {success_count}/{len(tasks)} images")
    return results


async def _generate_single_image_with_name(llm, image_name: str, caption: str, output_dir: str, model: str):
    try:
        result = await llm.generate_text_to_image(caption)

        if not result["success"]:
            print(f"Failed to generate image for {image_name}: {result['error']}")
            return None

        image_data = base64.b64decode(result["image_base64"])

        img = Image.open(io.BytesIO(image_data))
        img_resized = img.resize(
            (config.IMAGE_RESIZE_WIDTH, config.IMAGE_RESIZE_HEIGHT),
            Image.Resampling.LANCZOS
        )

        output_path = Path(output_dir) / image_name
        img_resized.save(output_path, quality=config.IMAGE_QUALITY, optimize=True)

        print(f"Generated {image_name} (resized to {config.IMAGE_RESIZE_WIDTH}x{config.IMAGE_RESIZE_HEIGHT})")
        return str(output_path)

    except Exception as e:
        print(f"Failed to generate image for {image_name}: {e}")
        return None


async def generate_visual_query(original_query: str, item_name: str, caption: str, keyword: str, same_category_data: list):
    global _vq_total_input_tokens, _vq_total_output_tokens
    client = AsyncOpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)
    prompt = get_visual_query_prompt(original_query, item_name, caption, keyword, same_category_data)

    response = await client.chat.completions.create(
        model=config.MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS
    )

    if hasattr(response, 'usage') and response.usage:
        _vq_total_input_tokens += response.usage.prompt_tokens
        _vq_total_output_tokens += response.usage.completion_tokens

    return response.choices[0].message.content.strip()


async def rephase_navi_queries(bfs_json_path: str, caption_data: dict, output_path: str):
    bfs_data = load_data(bfs_json_path)

    item_id = bfs_data['item_id']
    item = None
    item_collection_name = None

    for collection_name, items in caption_data.items():
        if isinstance(items, list):
            for i in items:
                if i.get('id') == item_id:
                    item = i
                    item_collection_name = collection_name
                    break
            if item:
                break

    same_category_data = caption_data.get(item_collection_name, []) if item_collection_name else []

    original_query = bfs_data.get('query', '')
    bfs_data['visual_query'] = await generate_visual_query(
        original_query,
        item.get('name', ''),
        item['caption'],
        item.get('keyword', ''),
        same_category_data
    )
    bfs_data['original_query'] = original_query

    save_data(bfs_data, output_path)
    return bfs_data



def update_data_js_image_paths(data_js_path: str, caption_data: dict, images_dir: str, output_path: str):
    data = load_data(data_js_path)

    for collection_name, items in data.items():
        if isinstance(items, list):
            for item in items:
                if 'id' in item and 'image' in item:
                    new_image_path = f"{images_dir}/{collection_name}_{item['id']}.jpg"
                    item['image'] = new_image_path

    save_data(data, output_path)
    return data


def update_data_js_image_paths_inplace(data_js_path: str, caption_data: dict) -> int:
    """
    Update image paths in data.js file while preserving the original format.
    Uses regex to replace image paths without changing the file structure.

    Args:
        data_js_path: Path to data.js file
        caption_data: Caption data dictionary with collection names and item IDs

    Returns:
        Number of image paths updated
    """
    # Read the entire file content
    with open(data_js_path, 'r', encoding='utf-8') as f:
        content = f.read()

    updated_count = 0

    # For each collection and item, replace the image path
    for collection_name, items in caption_data.items():
        if isinstance(items, list):
            for item in items:
                if 'id' in item:
                    item_id = item['id']
                    new_image_path = f"/images/{collection_name}_{item_id}.jpg"

                    # Pattern to match: id: 'e1' ... image: '/images/xxx.jpg'
                    # We need to find the item with this id and replace its image field
                    # Using a regex that matches the item object containing both id and image

                    # Pattern explanation:
                    # - Find "id: 'item_id'" or 'id: "item_id"'
                    # - Then find "image: " followed by a quoted string
                    # - Replace the image value

                    # Support both single and double quotes
                    pattern = rf"(id:\s*['\"]?{re.escape(item_id)}['\"]?.*?image:\s*['\"])[^'\"]*(['\"])"
                    replacement = rf"\1{new_image_path}\2"

                    new_content, count = re.subn(pattern, replacement, content, flags=re.DOTALL)

                    if count > 0:
                        content = new_content
                        updated_count += count

    # Write back to file
    with open(data_js_path, 'w', encoding='utf-8') as f:
        f.write(content)

    return updated_count


def move_images_and_update_data_js(data_js_path: str, new_images_dir: str, caption_data: dict):
    """
    Move images to project's public/images directory and update data.js with new paths.
    Preserves the original format of data.js file.

    Args:
        data_js_path: Path to original data.js file
        new_images_dir: Directory containing generated images (e.g., outs/JD/new_images)
        caption_data: Caption data dictionary

    Returns:
        Number of updated paths
    """
    # 1. Derive project's public/images path
    # data_js_path: .../vue_template/src/stores/data.js
    # public_images: .../vue_template/public/images
    data_js_path_obj = Path(data_js_path)
    project_root = data_js_path_obj.parent.parent.parent  # Go up 3 levels from src/stores/data.js
    public_images_dir = project_root / "public" / "images"

    # 2. Backup original data.js to initial_data.js (only if not already backed up)
    initial_data_js_path = data_js_path_obj.parent / "initial_data.js"
    if not initial_data_js_path.exists():
        print(f"Creating backup: {initial_data_js_path}")
        shutil.copy2(data_js_path, initial_data_js_path)
        print(f"✓ Backed up original data.js to initial_data.js")
    else:
        print(f"✓ Backup already exists: {initial_data_js_path}")

    # 3. Ensure target directory exists
    public_images_dir.mkdir(parents=True, exist_ok=True)
    print(f"Target directory: {public_images_dir}")

    # 4. Copy image files
    print(f"Copying images from {new_images_dir} to {public_images_dir}")
    copied_count = 0
    skipped_count = 0

    for collection_name, items in caption_data.items():
        if isinstance(items, list):
            for item in items:
                if 'id' in item:
                    image_name = f"{collection_name}_{item['id']}.jpg"
                    source_file = Path(new_images_dir) / image_name
                    target_file = public_images_dir / image_name

                    if source_file.exists():
                        shutil.copy2(source_file, target_file)
                        copied_count += 1
                        print(f"  ✓ Copied {image_name}")
                    else:
                        skipped_count += 1
                        print(f"  ✗ Skipped {image_name} (not found)")

    print(f"✓ Copied {copied_count} images to {public_images_dir}")
    if skipped_count > 0:
        print(f"⚠ Skipped {skipped_count} images (not found)")

    # 5. Update data.js image paths (preserving original format)
    print(f"Updating image paths in {data_js_path}")
    updated_count = update_data_js_image_paths_inplace(data_js_path, caption_data)
    print(f"✓ Updated {updated_count} image paths in data.js")

    return updated_count


async def main(data_js_path: str, bfs_dir: str, output_dir: str, stage: int = 1, skip_stages: str = '', 
               caption_model: str = None, image_model: str = None, visual_query_model: str = None):
    data_js_path = str(Path(data_js_path) / "src/stores/data.js")

    skip_list = [int(s.strip()) for s in skip_stages.split(',') if s.strip()]

    data_path = Path(data_js_path).parent.parent.parent
    parent_dir_name = data_path.name

    if '-' in parent_dir_name:
        parts = parent_dir_name.split('-')
        if len(parts) >= 2:
            theme_name = parts[1]
        else:
            theme_name = parent_dir_name
    else:
        theme_name = parent_dir_name

    print(f"Theme: {theme_name}")
    print(f"Data file: {data_js_path}")
    print(f"Starting from Stage {stage}")
    if skip_list:
        print(f"Skipping stages: {skip_list}")

    output_path = Path(output_dir)
    theme_output_dir = output_path / theme_name
    theme_output_dir.mkdir(parents=True, exist_ok=True)

    caption_path = theme_output_dir / "caption.json"

    # Stage 1: Generate captions
    if stage <= 1 and 1 not in skip_list:
        print("\n=== Stage 1: Generating captions ===")

        if caption_path.exists():
            caption_path.unlink()
            print(f"Removed existing caption.json")

        caption_data = generate_caption(data_js_path, str(caption_path), model=caption_model)
        print(f"✓ Captions saved to {caption_path}")

        caption_stats = get_caption_usage_stats()
        caption_cost_file = theme_output_dir / 'api_cost_stage1_caption.json'
        save_data(caption_stats, str(caption_cost_file))
    elif 1 in skip_list:
        print(f"\n=== Stage 1: Skipped (skip_stages) ===")
        if caption_path.exists():
            caption_data = load_data(str(caption_path))
            print(f"✓ Loaded existing captions from {caption_path}")
        else:
            raise FileNotFoundError(f"Caption file not found: {caption_path}. Please run stage 1 first or remove it from skip_stages.")
    else:
        print(f"\n=== Stage 1: Skipped (loading existing caption.json) ===")
        if not caption_path.exists():
            raise FileNotFoundError(f"Caption file not found: {caption_path}. Please run from stage 1 first.")
        caption_data = load_data(str(caption_path))
        print(f"✓ Loaded captions from {caption_path}")

    # Stage 2: Generate images
    images_dir = theme_output_dir / "new_images"
    if stage <= 2 and 2 not in skip_list:
        print("\n=== Stage 2: Generating images ===")

        if images_dir.exists():
            shutil.rmtree(images_dir)
            print(f"Removed existing images directory")

        use_image_model = image_model if image_model is not None else "gemini-2.5-flash-image"
        print(f"Using image model: {use_image_model}")
        
        await generate_images_with_category(caption_data, str(images_dir), image_model=use_image_model)
        print(f"✓ Images saved to {images_dir}")

        if _image_llm_instance:
            image_stats = _image_llm_instance.get_usage_summary()
            image_cost_file = theme_output_dir / 'api_cost_stage1_image.json'
            save_data(image_stats, str(image_cost_file))
    else:
        if 2 in skip_list:
            print("\n=== Stage 2: Skipped (skip_stages) ===")
        else:
            print("\n=== Stage 2: Skipped ===")
        print(f"Using existing images directory: {images_dir}")

    # Stage 3: Generate visual queries
    visual_queries_dir = theme_output_dir / "visual_queries"
    if stage <= 3 and 3 not in skip_list:
        print("\n=== Stage 3: Generating visual queries ===")

        if not bfs_dir or not Path(bfs_dir).exists():
            print(f"ERROR: bfs_dir is required for Stage 3 but not provided or doesn't exist: {bfs_dir}")
            print("Skipping Stage 3. Please run item expansion first or provide valid bfs_dir.")
            results = []
        else:
            print(f"BFS directory: {bfs_dir}")

            if visual_queries_dir.exists():
                shutil.rmtree(visual_queries_dir)
                print(f"Removed existing visual_queries directory")

            bfs_files = list(Path(bfs_dir).rglob("*.json"))
            print(f"Found {len(bfs_files)} trajectory files")

            if len(bfs_files) > 10000:
                print(f"WARNING: Found {len(bfs_files)} files, which seems abnormal!")
                print(f"Please verify the bfs_dir path is correct: {bfs_dir}")

            tasks = []
            skipped_count = 0
            search_skipped_count = 0
            type_error_count = 0
            failed_trajectory_count = 0

            for bfs_file in bfs_files:
                try:
                    bfs_data = load_data(str(bfs_file))
                except Exception as e:
                    print(f"Failed to load {bfs_file}: {e}")
                    skipped_count += 1
                    continue

                if not isinstance(bfs_data, dict):
                    type_error_count += 1
                    continue

                if bfs_data.get('item') is None:
                    skipped_count += 1
                    continue

                if bfs_data.get('trajectory_type') == 'SEARCH':
                    search_skipped_count += 1
                    continue

                if bfs_file.name == 'bfs.json':
                    result_file = bfs_file.parent / 'result.json'
                    if not result_file.exists():
                        failed_trajectory_count += 1
                        continue

                relative_path = bfs_file.relative_to(bfs_dir)
                output_file = visual_queries_dir / relative_path
                output_file.parent.mkdir(parents=True, exist_ok=True)

                tasks.append(rephase_navi_queries(str(bfs_file), caption_data, str(output_file)))

            if type_error_count > 0:
                print(f"Skipped {type_error_count} files with wrong type (not dict)")
            print(f"Skipped {skipped_count} trajectories with null item")
            print(f"Skipped {search_skipped_count} trajectories with SEARCH type")
            if failed_trajectory_count > 0:
                print(f"Skipped {failed_trajectory_count} failed trajectories (no result.json)")
            print(f"Processing {len(tasks)} trajectories")

            results = await asyncio.gather(*tasks, return_exceptions=True)

            success_count = len([r for r in results if not isinstance(r, Exception)])
            error_count = len(results) - success_count

            print(f"✓ Generated {success_count} visual queries")
            if error_count > 0:
                print(f"✗ {error_count} files failed")

            vq_stats = {
                'model': visual_query_model,
                'input_tokens': _vq_total_input_tokens,
                'output_tokens': _vq_total_output_tokens,
                'total_tokens': _vq_total_input_tokens + _vq_total_output_tokens,
                'input_cost_usd': round((_vq_total_input_tokens / 1_000_000) * 0.21, 6),
                'output_cost_usd': round((_vq_total_output_tokens / 1_000_000) * 0.32, 6),
                'total_cost_usd': round(((_vq_total_input_tokens / 1_000_000) * 0.21) + ((_vq_total_output_tokens / 1_000_000) * 0.32), 6),
                'price_per_million': {'input': 0.21, 'output': 0.32}
            }
            vq_cost_file = theme_output_dir / 'api_cost_stage4_visual_query.json'
            save_data(vq_stats, str(vq_cost_file))
    else:
        if 3 in skip_list:
            print("\n=== Stage 3: Skipped (skip_stages) ===")
        else:
            print("\n=== Stage 3: Skipped ===")
        results = []

    # Stage 4: Move images and update data.js
    if stage <= 4 and 4 not in skip_list:
        print("\n=== Stage 4: Moving images and updating data.js ===")
        move_images_and_update_data_js(
            data_js_path,
            str(images_dir),
            caption_data
        )
        print("✓ Images moved to project and data.js updated")
    else:
        if 4 in skip_list:
            print("\n=== Stage 4: Skipped (skip_stages) ===")
        else:
            print("\n=== Stage 4: Skipped ===")

    print(f"\n✅ All outputs saved to: {output_path}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate visual queries from data.js and BFS trajectories'
    )
    parser.add_argument('--data-path', required=True, help='Path to data.js directory')
    parser.add_argument('--bfs-dir', default='', help='Path to BFS trajectories directory (required for stage 3)')
    parser.add_argument('--output-dir', required=True, help='Path to output directory')
    parser.add_argument(
        '--stage',
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help=(
            'Starting stage: '
            '1=from caption generation, '
            '2=from image generation, '
            '3=from visual query generation, '
            '4=only move images and update data.js '
            '(default: 1)'
        )
    )
    parser.add_argument(
        '--skip-stages',
        type=str,
        default='',
        help='Comma-separated list of stages to skip (e.g., "3" or "2,3")'
    )
    parser.add_argument(
        '--caption-model',
        type=str,
        default=None,
        help='Model for caption generation (Stage 1). If not specified, uses config.MODEL'
    )
    parser.add_argument(
        '--visual-query-model',
        type=str,
        default=None,
        help='Model for visual query generation (Stage 3). If not specified, uses deepseek-chat'
    )
    parser.add_argument(
        '--image-model',
        type=str,
        default=None,
        help='Model for image generation (Stage 2). If not specified, uses gemini-2.5-flash-image'
    )

    args = parser.parse_args()

    asyncio.run(main(args.data_path, args.bfs_dir, args.output_dir, args.stage, args.skip_stages,
                     caption_model=args.caption_model, image_model=args.image_model, 
                     visual_query_model=args.visual_query_model))
