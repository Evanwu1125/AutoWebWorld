"""
Query Generator - Generate natural language queries for expanded trajectories.

This module takes expanded trajectories and generates:
1. Natural language query
2. Slider configuration (target_value, direction, rank)
3. Input text replacements
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI


class QueryGenerator:
    """Generate queries and configurations for expanded trajectories."""

    def __init__(
        self,
        mockdata: Dict[str, Any],
        model: str = "deepseek-v3.2-exp",
        base_url: str = "https://newapi.deepwisdom.ai/v1"
    ):
        self.mockdata = mockdata
        self.model = model
        self.base_url = base_url
        self._client: Optional[AsyncOpenAI] = None
        self._prompt_template = self._load_prompt_template()

        # Token usage statistics
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _load_prompt_template(self) -> str:
        prompt_path = Path(__file__).parent / "prompts" / "navigation_query_generator_prompt.txt"
        if prompt_path.exists():
            return prompt_path.read_text()
        return ""

    async def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("Missing OPENAI_API_KEY environment variable")
            self._client = AsyncOpenAI(api_key=api_key, base_url=self.base_url)
        return self._client

    async def generate_query(
        self,
        expanded_trajectory: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate query and configuration for an expanded trajectory.

        Args:
            expanded_trajectory: Dict containing item, trajectory, trajectory_type, etc.

        Returns:
            Dict with query, slider, input_text_replacements
        """
        try:
            client = await self._get_client()

            # Extract info from expanded trajectory
            item = expanded_trajectory.get('item')
            item_id = expanded_trajectory.get('item_id')
            trajectory = expanded_trajectory.get('trajectory', [])
            trajectory_type = expanded_trajectory.get('trajectory_type')
            filter_field = expanded_trajectory.get('filter_field')
            mockdata_key = expanded_trajectory.get('mockdata_key')

            # Extract pre-calculated values
            rank = expanded_trajectory.get('rank', 1)
            target_value = expanded_trajectory.get('target_value')
            sort_order = expanded_trajectory.get('sort_order')

            # Handle None item (for NO_ITEM trajectory type)
            if item is None:
                item = {}
            item_name = item.get('name', '')

            # Get terminal state (last action's 'to' field)
            terminal_state = "UNKNOWN"
            if trajectory:
                terminal_state = trajectory[-1].get('to', 'UNKNOWN')

            # Get mockdata for this key
            mockdata_items = self.mockdata.get(mockdata_key, []) if mockdata_key else []

            prompt = self._prompt_template.format(
                trajectory_json=json.dumps(trajectory, indent=2, ensure_ascii=False),
                mockdata_json=json.dumps(mockdata_items, indent=2, ensure_ascii=False),
                item_id=item_id,
                item_name=item_name,
                item_json=json.dumps(item, indent=2, ensure_ascii=False),
                trajectory_type=trajectory_type,
                filter_field=filter_field or "null",
                rank=rank,
                target_value=target_value if target_value is not None else "null",
                sort_order=sort_order if sort_order else "null",
                terminal_state=terminal_state
            )

            resp = await client.chat.completions.create(
                model=self.model,
                max_tokens=2048,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )

            # Extract usage statistics
            if hasattr(resp, 'usage') and resp.usage:
                self.total_input_tokens += resp.usage.prompt_tokens
                self.total_output_tokens += resp.usage.completion_tokens

            if isinstance(resp, str):
                content = resp.strip()
            elif hasattr(resp, 'choices'):
                content = resp.choices[0].message.content.strip()
            else:
                return self._empty_result("Invalid LLM response format")

            result = self._parse_llm_response(content)

            if not result:
                return self._empty_result("Failed to parse LLM response")

            print(f"[QueryGen] Generated query for item {item_id}: {result.get('query', '')[:50]}...")

            return result

        except Exception as e:
            print(f"Query generation failed: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_result(str(e))

    def _parse_llm_response(self, content: str) -> Optional[Dict[str, Any]]:
        # Remove markdown code block if present
        if '```json' in content:
            start = content.find('```json') + 7
            end = content.find('```', start)
            content = content[start:end].strip()
        elif '```' in content:
            start = content.find('```') + 3
            end = content.find('```', start)
            content = content[start:end].strip()

        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
            return None
        except json.JSONDecodeError:
            # Try to find JSON object in content
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(content[start:end])
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass
            return None

    def _empty_result(self, reason: str) -> Dict[str, Any]:
        """Return empty result with error reason."""
        return {
            'query': '',
            'slider': None,
            'input_text_replacements': [],
            'error': reason
        }

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get token usage statistics.

        Returns:
            Dict with input_tokens, output_tokens, and cost information
        """
        # Price per million tokens
        input_price_per_m = 0.21
        output_price_per_m = 0.32

        # Calculate costs
        input_cost = (self.total_input_tokens / 1_000_000) * input_price_per_m
        output_cost = (self.total_output_tokens / 1_000_000) * output_price_per_m
        total_cost = input_cost + output_cost

        return {
            'model': self.model,
            'input_tokens': self.total_input_tokens,
            'output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'input_cost_usd': round(input_cost, 6),
            'output_cost_usd': round(output_cost, 6),
            'total_cost_usd': round(total_cost, 6),
            'price_per_million': {
                'input': input_price_per_m,
                'output': output_price_per_m
            }
        }

