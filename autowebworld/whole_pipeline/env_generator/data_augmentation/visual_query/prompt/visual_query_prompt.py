VISUAL_QUERY_GENERATION_PROMPT = """Your task is to rewrite a navigation query by replacing item references with visual descriptions, while keeping ALL other parts of the query COMPLETELY UNCHANGED.

**Input Information:**
Original Query: {original_query}
Item Name: {item_name}
Visual Caption: {caption}
Visual Keywords: {keyword}

**Same Category Items (all items in this category for reference):**
{same_category_data}

**Step 1: Identify what to replace**

Determine which pattern the query follows:

Pattern A - SEARCH/SCROLL (item name appears directly):
- Query contains the item name explicitly
- Example: "Find iPhone 15 Pro Max, buy 3 pieces"
- Replace: The item name → visual description

Pattern B - SLIDER (position + filter condition):
- Query contains position words like "first/second/third product/item" AND filter conditions
- Example: "Buy the first product with price above $2"
- Replace: Position reference ("the first product") → visual description
- KEEP: Filter condition ("with price above $2")

Pattern C - SORT (sorting + ordinal position):
- Query contains "sorted by..." and ordinal positions like "seventh one", "third item"
- Example: "From items sorted by price low to high, get the seventh one"
- Replace: Ordinal position ("the seventh one") → visual description
- KEEP: Sorting condition ("From items sorted by price low to high")

**Step 2: Create a specific visual description**

**IMPORTANT: Review the "Same Category Items" above to determine uniqueness:**
- Check if other items in the category share similar colors or features
- Use the SIMPLEST description that uniquely identifies this item among the category
- If your item's color is unique in the category → Use Level 1 (Color + Item Type)
- If multiple items share similar features → Add distinguishing details (Level 2 or 3)

Requirements for the visual description:
1. Must be SPECIFIC enough to uniquely identify this item among the same category items listed above
2. Extract the most DISTINCTIVE features from the caption and keywords
3. Use the SIMPLEST description that achieves unique identification
4. Length: 3-15 words (shorter is better if it's unique enough)
5. Use concrete, observable visual details

**Strategy for creating descriptions (use the simplest description that uniquely identifies the item):**

Level 1 - Color + Item Type (PREFERRED if color is unique):
- "black smartphone" (if it's the only black smartphone in the category)
- "yellow bananas" (if it's the only yellow fruit in the category)
- "blue detergent bottle" (if it's the only blue bottle in the category)
- "silver laptop" (if it's the only silver laptop in the category)

Level 2 - Color + Item Type + One Key Feature (if color alone is not unique):
- "black smartphone with triple camera" (if there are multiple black smartphones)
- "ripe yellow organic bananas" (if there are other yellow items)
- "large blue detergent bottle" (if there are multiple blue bottles)

Level 3 - Detailed Description (only if needed for unique identification):
- "sleek black smartphone with glossy display and triple camera array"
- "black gaming laptop with glowing RGB backlit keyboard"

**Examples of descriptions that are TOO VAGUE (avoid these):**
- "smartphone" (missing color - not specific enough)
- "item" (too generic)
- "product" (too generic)

**Step 3: Preserve ALL other parts**

You MUST keep these elements EXACTLY as they appear in the original query:
- All actions: buy, ship, deliver, select, find, scroll, order, etc.
- All quantities: "3 pieces", "2 bunches", etc.
- All addresses: "usual address", "regular address", "saved address", etc.
- All payment methods: "cash on delivery", "card payment", etc.
- All filter conditions: "with price above $X", "organic", "fresh", etc.
- All sorting conditions: "sorted by price/rating/sales", "low to high", "high to low", etc.
- All option selections: "first option", "first available option", etc.
- All other details: "from supermarket section", etc.

**Examples:**

Example 1 (SEARCH type):
Original: "Find Sony WH-1000XM5 headphones, buy 3 pieces, ship to my usual address with cash on delivery"
Output: "Find black over-ear headphones with comfortable padded earcups, buy 3 pieces, ship to my usual address with cash on delivery"

Example 2 (SLIDER type):
Original: "Buy 3 pieces of the first product with price above $2 from the supermarket section, choose the first option, ship to my usual address and pay cash on delivery"
Output: "Buy 3 pieces of a bunch of ripe yellow organic bananas with price above $2 from the supermarket section, choose the first option, ship to my usual address and pay cash on delivery"

Example 3 (SORT type):
Original: "From items sorted by price low to high, get the seventh one, 3 pieces, deliver to usual address, cash on delivery"
Output: "From items sorted by price low to high, get the [specific visual description based on caption], 3 pieces, deliver to usual address, cash on delivery"

**CRITICAL: Output the COMPLETE rewritten query. Do NOT truncate or omit any parts. The output must be a full, grammatically correct sentence with ALL elements from the original query preserved.**

Return ONLY the rewritten query, nothing else."""


def get_visual_query_prompt(original_query: str, item_name: str, caption: str, keyword: str, same_category_data: list) -> str:
    import json

    same_category_data_str = json.dumps(same_category_data, indent=2, ensure_ascii=False)

    return VISUAL_QUERY_GENERATION_PROMPT.format(
        original_query=original_query,
        item_name=item_name,
        caption=caption,
        keyword=keyword,
        same_category_data=same_category_data_str
    )

