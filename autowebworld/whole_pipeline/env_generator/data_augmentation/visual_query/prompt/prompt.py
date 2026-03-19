CAPTION_GENERATION_PROMPT = """You are an image description expert. I will provide you with a JSON file.

Please add two new fields to ALL items that have image fields (avatar, image, etc.):
1. "keyword": 3-4 English keywords connected by `-`
2. "caption": A complete English description (15-25 words)

Requirements:
- Generate captions for ALL items with image/avatar fields, regardless of collection type
- Determine the image type based on item content (portrait/icon/illustration)
- **COLOR is the PRIMARY distinguishing feature - always include color information**
- Keywords must use concrete visual features, avoid vague adjectives (e.g., charming, shining):
  * For people: hair color/length, smile, glasses, skin tone, gender, age, etc.
  * For objects: **COLOR (first priority)**, shape, material, pattern, texture, etc.
  * Use simple color words (e.g., "black", "white", "red", "blue", "gray", "yellow")
  * For composite colors, describe them naturally (e.g., "black and white", "red and blue striped")
  * Multiple items can have the same color - additional visual features will be used for differentiation in query generation
- Caption should include keywords and specify the image type
- Different items' keywords can share 1-2 keywords, but cannot be completely identical
- **Ensure color information is prominent in both keywords and captions to enable easy item identification**

Input JSON:
{data_json}

Output the complete JSON with the same format as input."""


def get_caption_prompt(data_json: str) -> str:
    return CAPTION_GENERATION_PROMPT.format(data_json=data_json)

