"""VLM verification for QA pairs"""
import base64
import json
from pathlib import Path
from typing import Dict, Any, List
from openai import AsyncOpenAI

from .config import DEFAULT_VLM_MODEL, PRICING
from .utils import calculate_cost


class VLMVerifier:
    
    def __init__(self, api_key: str, base_url: str, model: str = DEFAULT_VLM_MODEL):
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.prompt_template = self._load_prompt_template()
        
        self.total_input_tokens = 0
        self.total_output_tokens = 0
    
    def _load_prompt_template(self) -> str:
        prompt_path = Path(__file__).parent / "prompts" / "vlm_verification_prompt.txt"
        return prompt_path.read_text(encoding='utf-8')
    
    @staticmethod
    def encode_image(image_path: Path) -> str:
        """Encode image to base64"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    async def verify_qa(self, screenshot_path: Path, question: str, ground_truth: str) -> Dict:
        """Verify if ground truth answer is visible in screenshot"""
        image_b64 = self.encode_image(screenshot_path)
        
        prompt = self.prompt_template.format(
            question=question,
            ground_truth=ground_truth
        )
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                        }
                    ]
                }
            ],
            temperature=0.3,
            max_tokens=512
        )
        
        if hasattr(response, 'usage') and response.usage:
            self.total_input_tokens += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens
        
        content = response.choices[0].message.content.strip()
        return self._parse_verification_result(content)
    
    async def batch_verify(self, screenshot_path: Path, qa_pairs: List[Dict]) -> List[Dict]:
        """Verify multiple QA pairs for the same screenshot"""
        verified = []
        
        for qa in qa_pairs:
            result = await self.verify_qa(
                screenshot_path,
                qa['question'],
                qa['answer']
            )
            
            if result.get('answerable', False):
                verified.append({
                    **qa,
                    'vlm_verification': result
                })
        
        return verified
    
    def _parse_verification_result(self, content: str) -> Dict:
        """Parse VLM verification response"""
        if '```json' in content:
            start = content.find('```json') + 7
            end = content.find('```', start)
            content = content[start:end].strip()
        elif '```' in content:
            start = content.find('```') + 3
            end = content.find('```', start)
            content = content[start:end].strip()
        
        try:
            result = json.loads(content)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
        
        return {
            "answerable": False,
            "reason": "Failed to parse VLM response",
            "confidence": "low",
            "location": ""
        }
    
    def get_usage_stats(self) -> Dict:
        """Get API usage statistics"""
        costs = calculate_cost(
            self.total_input_tokens,
            self.total_output_tokens,
            self.model,
            PRICING
        )
        
        return {
            "model": self.model,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            **costs
        }

