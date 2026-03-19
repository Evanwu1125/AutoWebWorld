import json
from typing import Dict, Any

from openai import OpenAI

from . import config
from .prompt.prompt import get_caption_prompt


_total_input_tokens = 0
_total_output_tokens = 0
_model_used = None

def generate_captions(data: Dict[str, Any], model: str = None) -> Dict[str, Any]:
    global _total_input_tokens, _total_output_tokens, _model_used

    use_model = model if model is not None else config.MODEL
    _model_used = use_model
    
    # Step 1: Collect all items with images (no deduplication)
    unique_data = {}  # Data to send to LLM
    total_items = 0
    collection_stats = {}

    for collection_name, items in data.items():
        if isinstance(items, list):
            unique_data[collection_name] = []
            collection_count = 0

            for item in items:
                image_field = item.get('image') or item.get('avatar')

                if image_field:
                    total_items += 1
                    collection_count += 1
                    unique_data[collection_name].append(item.copy())

            if collection_count > 0:
                collection_stats[collection_name] = collection_count

    # Remove empty collections
    unique_data = {k: v for k, v in unique_data.items() if v}

    print(f"📊 Summary: {total_items} items with images; generating captions for all items (no deduplication)")
    print(f"📋 Collection statistics:")
    for coll_name, count in collection_stats.items():
        print(f"   - {coll_name}: {count} items")

    # Step 2: Call LLM to generate captions
    unique_data_json = json.dumps(unique_data, indent=2, ensure_ascii=False)
    prompt = get_caption_prompt(unique_data_json)

    print(f"🔄 Calling the LLM to generate {total_items} captions...")
    print(f"   Model: {use_model}")

    import time
    start_time = time.time()

    try:
        result = _call_llm(prompt, use_model)
        captioned_data = json.loads(result)

        elapsed_time = time.time() - start_time
        print(f"✅ Caption generation completed!")
        print(f"⏱️  Elapsed time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        print(f"📊 Average per item: {elapsed_time/total_items:.2f} seconds")
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        with open("debug_llm_response.json", "w", encoding="utf-8") as f:
            f.write(result)
        print(f"The full response has been saved to debug_llm_response.json")
        raise

    # Step 3: Create item ID to caption mapping
    item_to_caption = {}
    llm_response_stats = {}
    for collection_name, items in captioned_data.items():
        if isinstance(items, list):
            llm_response_stats[collection_name] = len(items)
            for item in items:
                if 'id' in item and 'caption' in item:
                    item_to_caption[item['id']] = {
                        'caption': item['caption'],
                        'keyword': item.get('keyword', '')
                    }

    print(f"📥 LLM response statistics:")
    for coll_name, count in llm_response_stats.items():
        print(f"   - {coll_name}: {count} items")

    # Step 4: Assign captions to all items in original data by item ID
    for collection_name, items in data.items():
        if isinstance(items, list):
            for item in items:
                if 'id' in item and item['id'] in item_to_caption:
                    item['caption'] = item_to_caption[item['id']]['caption']
                    item['keyword'] = item_to_caption[item['id']]['keyword']

    print(f"✅ Successfully generated captions for {len(item_to_caption)} items")

    missing_captions = []
    for collection_name, items in data.items():
        if isinstance(items, list):
            for item in items:
                if 'id' in item:
                    image_field = item.get('image') or item.get('avatar')
                    if image_field and 'caption' not in item:
                        missing_captions.append(f"{collection_name}:{item['id']}")

    if missing_captions:
        print(f"⚠️  Warning: the following {len(missing_captions)} items were not processed by the LLM:")
        for item_id in missing_captions[:10]:
            print(f"   - {item_id}")
        if len(missing_captions) > 10:
            print(f"   ... and {len(missing_captions) - 10} more")

    return data


def _call_llm(prompt: str, model: str = None) -> str:
    global _total_input_tokens, _total_output_tokens
    import time
    
    use_model = model if model is not None else config.MODEL

    print(f"📤 Sending request to the LLM...")
    print(f"   Prompt length: {len(prompt)} characters")

    start_time = time.time()
    client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)

    response = client.chat.completions.create(
        model=use_model,
        messages=[
            {"role": "system", "content": "You are an expert at generating image descriptions."},
            {"role": "user", "content": prompt}
        ],
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS
    )

    api_time = time.time() - start_time
    print(f"📥 Received response, API call took: {api_time:.2f} seconds")

    if hasattr(response, 'usage'):
        usage = response.usage
        _total_input_tokens += usage.prompt_tokens
        _total_output_tokens += usage.completion_tokens
        print(f"📊 Token usage:")
        print(f"   Input: {usage.prompt_tokens:,} tokens")
        print(f"   Output: {usage.completion_tokens:,} tokens")
        print(f"   Total: {usage.prompt_tokens + usage.completion_tokens:,} tokens")
        if usage.completion_tokens > 0:
            tokens_per_sec = usage.completion_tokens / api_time
            print(f"   Generation speed: {tokens_per_sec:.1f} tokens/sec")

    if isinstance(response, str):
        result = response.strip()
    else:
        result = response.choices[0].message.content.strip()
    return _clean_json_response(result)


def _clean_json_response(result: str) -> str:
    result = result.strip()

    if result.startswith("```json"):
        result = result[7:]
    elif result.startswith("```"):
        result = result[3:]

    if result.endswith("```"):
        result = result[:-3]

    result = result.strip()

    if not result.endswith('}') and not result.endswith(']'):
        last_brace = result.rfind('}')
        last_bracket = result.rfind(']')
        last_pos = max(last_brace, last_bracket)

        if last_pos > 0:
            result = result[:last_pos + 1]
            print(f"⚠️  Detected truncated JSON; trimmed to the last complete object")

    return result

def get_caption_usage_stats() -> Dict[str, Any]:
    input_price_per_m = 0.3
    output_price_per_m = 0.252
    input_cost = (_total_input_tokens / 1_000_000) * input_price_per_m
    output_cost = (_total_output_tokens / 1_000_000) * output_price_per_m
    return {
        'model': _model_used,
        'input_tokens': _total_input_tokens,
        'output_tokens': _total_output_tokens,
        'total_tokens': _total_input_tokens + _total_output_tokens,
        'input_cost_usd': round(input_cost, 6),
        'output_cost_usd': round(output_cost, 6),
        'total_cost_usd': round(input_cost + output_cost, 6),
        'price_per_million': {
            'input': input_price_per_m,
            'output': output_price_per_m
        }
    }




