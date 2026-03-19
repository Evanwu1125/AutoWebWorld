"""
Filter Condition Parser.

Parses filter conditions from selectors using rules and optionally LLM.
"""
import os
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from openai import AsyncOpenAI


@dataclass
class FilterCondition:
    """Represents a parsed filter condition."""
    field: str                    # Field name in mockdata
    operator: str                 # Comparison operator: ==, !=, >, <, >=, <=
    value: Any                    # Value to compare against
    is_boolean: bool = False      # Whether this is a boolean field filter
    selector: str = ""            # Original selector
    parsed_by: str = "rule"       # "rule" or "llm"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "operator": self.operator,
            "value": self.value,
            "is_boolean": self.is_boolean,
            "selector": self.selector,
            "parsed_by": self.parsed_by
        }


class FilterParser:
    """
    Parses filter conditions from CSS selectors.
    
    Uses rule-based parsing first, falls back to LLM for complex cases.
    """
    
    def __init__(
        self,
        mockdata_schema: Dict[str, List[str]],
        use_llm: bool = False,
        model: str = "deepseek-v3.2-exp",
        base_url: str = ""
    ):
        """
        Initialize the filter parser.
        
        Args:
            mockdata_schema: Schema of mockdata {entity_type: [field_names]}
            use_llm: Whether to use LLM for parsing when rules fail
            model: LLM model to use
            base_url: LLM API base URL
        """
        self.mockdata_schema = mockdata_schema
        self.use_llm = use_llm
        self.model = model
        self.base_url = base_url
        self._client: Optional[AsyncOpenAI] = None
    
    def parse(
        self,
        selector: str,
        entity_type: Optional[str] = None
    ) -> Optional[FilterCondition]:
        """
        Parse filter condition from selector.
        
        Args:
            selector: CSS selector string (e.g., "#filter-beds-3plus-checkbox")
            entity_type: Optional entity type to help with field matching
            
        Returns:
            FilterCondition or None if parsing fails
        """
        # Try rule-based parsing first
        condition = self._parse_by_rules(selector, entity_type)
        
        if condition:
            return condition
        
        # LLM parsing is async, return None for now
        # Use parse_async for LLM parsing
        return None
    
    async def parse_async(
        self,
        selector: str,
        entity_type: Optional[str] = None
    ) -> Optional[FilterCondition]:
        """
        Parse filter condition asynchronously (supports LLM).
        
        Args:
            selector: CSS selector string
            entity_type: Optional entity type
            
        Returns:
            FilterCondition or None
        """
        # Try rule-based first
        condition = self._parse_by_rules(selector, entity_type)
        
        if condition:
            return condition
        
        # Try LLM if enabled
        if self.use_llm:
            condition = await self._parse_by_llm(selector, entity_type)
            return condition
        
        return None
    
    def _parse_by_rules(
        self,
        selector: str,
        entity_type: Optional[str] = None
    ) -> Optional[FilterCondition]:
        """Parse using rule-based patterns."""
        
        # Pattern: #filter-{field}-{N}plus-checkbox (e.g., beds-3plus)
        match = re.search(
            r'#filter-([a-z_]+)-(\d+)plus-checkbox',
            selector,
            re.IGNORECASE
        )
        if match:
            field = match.group(1).replace('_', '')
            value = int(match.group(2))
            return FilterCondition(
                field=self._match_field(field, entity_type),
                operator=">=",
                value=value,
                selector=selector
            )
        
        # Pattern: #filter-{field}-max{N}-checkbox (e.g., price-max500)
        match = re.search(
            r'#filter-([a-z_]+)-max(\d+)-checkbox',
            selector,
            re.IGNORECASE
        )
        if match:
            field = match.group(1)
            value = int(match.group(2))
            return FilterCondition(
                field=self._match_field(field, entity_type),
                operator="<=",
                value=value,
                selector=selector
            )
        
        # Pattern: #filter-{field}-checkbox (boolean filter)
        match = re.search(
            r'#filter-([a-z_]+)-checkbox',
            selector,
            re.IGNORECASE
        )
        if match:
            field = match.group(1)
            return FilterCondition(
                field=self._match_field(field, entity_type),
                operator="==",
                value=True,
                is_boolean=True,
                selector=selector
            )
        
        return None
    
    def _match_field(self, field_name: str, entity_type: Optional[str]) -> str:
        """Match field name to actual field in mockdata schema."""
        if not entity_type or entity_type not in self.mockdata_schema:
            return field_name
        
        schema_fields = self.mockdata_schema[entity_type]
        
        # Exact match
        if field_name in schema_fields:
            return field_name
        
        # Case-insensitive match
        for f in schema_fields:
            if f.lower() == field_name.lower():
                return f
        
        # Partial match (field is prefix or suffix)
        for f in schema_fields:
            if field_name in f.lower() or f.lower() in field_name:
                return f
        
        # Common boolean field patterns
        bool_prefixes = ['is_', 'has_', 'can_', 'should_']
        for prefix in bool_prefixes:
            candidate = prefix + field_name
            if candidate in schema_fields:
                return candidate

        return field_name

    async def _get_client(self) -> AsyncOpenAI:
        """Get or create OpenAI client."""
        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("Missing OPENAI_API_KEY environment variable")
            self._client = AsyncOpenAI(api_key=api_key, base_url=self.base_url)
        return self._client

    async def _parse_by_llm(
        self,
        selector: str,
        entity_type: Optional[str] = None
    ) -> Optional[FilterCondition]:
        """Parse filter condition using LLM."""
        try:
            client = await self._get_client()

            schema_info = ""
            if entity_type and entity_type in self.mockdata_schema:
                fields = self.mockdata_schema[entity_type]
                schema_info = f"Available fields for {entity_type}: {fields}"

            prompt = f"""Parse this CSS selector to extract the filter condition.

Selector: {selector}
{schema_info}

Return a JSON object with:
- field: the field name being filtered
- operator: comparison operator (==, !=, >, <, >=, <=)
- value: the value to compare against
- is_boolean: true if this is a boolean field

Example: "#filter-beds-3plus-checkbox" -> {{"field": "beds", "operator": ">=", "value": 3, "is_boolean": false}}
Example: "#filter-direct-checkbox" -> {{"field": "is_direct", "operator": "==", "value": true, "is_boolean": true}}

Return only the JSON object, no explanation."""

            resp = await client.chat.completions.create(
                model=self.model,
                max_tokens=256,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )

            import json
            content = resp.choices[0].message.content.strip()
            # Remove markdown code blocks if present
            if content.startswith('```'):
                content = re.sub(r'^```\w*\n?', '', content)
                content = re.sub(r'\n?```$', '', content)

            data = json.loads(content)
            return FilterCondition(
                field=data.get('field', ''),
                operator=data.get('operator', '=='),
                value=data.get('value'),
                is_boolean=data.get('is_boolean', False),
                selector=selector,
                parsed_by="llm"
            )
        except Exception as e:
            print(f"LLM parsing failed for {selector}: {e}")
            return None

