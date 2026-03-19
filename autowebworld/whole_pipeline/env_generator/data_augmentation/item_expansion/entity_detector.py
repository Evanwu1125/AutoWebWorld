import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI


class EntityDetector:

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
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _load_prompt_template(self) -> str:
        prompt_path = Path(__file__).parent / "prompts" / "entity_key_prompt.txt"
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

    async def detect(
        self,
        trajectory: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Use LLM to analyze trajectory and detect type, key, filter field, and input replacements."""
        return await self._detect_by_llm(trajectory)

    async def _detect_by_llm(self, trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Use LLM to analyze trajectory and detect type, key, and filter field."""
        try:
            client = await self._get_client()

            # Pass complete mockdata to LLM so it can analyze actual field values
            prompt = self._prompt_template.format(
                trajectory_json=json.dumps(trajectory, indent=2, ensure_ascii=False),
                mockdata_json=json.dumps(self.mockdata, indent=2, ensure_ascii=False)
            )

            resp = await client.chat.completions.create(
                model=self.model,
                max_tokens=1024,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )

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

            print(f"[LLM] Detected: type={result.get('trajectory_type')}, "
                  f"key={result.get('mockdata_key')}, field={result.get('filter_field')}")

            mockdata_key = result.get('mockdata_key')
            if mockdata_key and mockdata_key in self.mockdata:
                result['items'] = self.mockdata[mockdata_key]
            else:
                result['items'] = []

            result['llm_raw_response'] = content
            return result

        except Exception as e:
            print(f"LLM detection failed: {e}")
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
            # Handle case where LLM returns a list instead of dict
            if isinstance(parsed, list) and len(parsed) > 0:
                parsed = parsed[0]
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
            'trajectory_type': None,
            'mockdata_key': None,
            'filter_field': None,
            'items': [],
            'reason': reason
        }

    def get_usage_stats(self) -> Dict[str, Any]:
        input_price_per_m = 0.21
        output_price_per_m = 0.32
        input_cost = (self.total_input_tokens / 1_000_000) * input_price_per_m
        output_cost = (self.total_output_tokens / 1_000_000) * output_price_per_m
        total_cost = input_cost + output_cost
        return {
            'input_tokens': self.total_input_tokens,
            'output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'input_cost_usd': round(input_cost, 6),
            'output_cost_usd': round(output_cost, 6),
            'total_cost_usd': round(total_cost, 6)
        }

    def get_item_id_field(self, items: List[Dict[str, Any]]) -> str:
        if not items:
            return "id"
        
        first_item = items[0]
        
        # Common ID field names in order of preference
        id_fields = ['id', 'item_id', 'product_id', '_id', 'uuid']
        
        for field in id_fields:
            if field in first_item:
                return field
        
        # Look for any field ending with _id
        for key in first_item.keys():
            if key.endswith('_id') or key.endswith('Id'):
                return key
        
        return "id"

