"""Question generation using LLM"""
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI

from .config import DEFAULT_LLM_MODEL, PRICING
from .utils import calculate_cost


class QuestionGenerator:
    
    def __init__(self, api_key: str, base_url: str, model: str = DEFAULT_LLM_MODEL):
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.prompt_template = self._load_prompt_template()
        
        self.total_input_tokens = 0
        self.total_output_tokens = 0
    
    def _load_prompt_template(self) -> str:
        prompt_path = Path(__file__).parent / "prompts" / "question_generation_prompt.txt"
        return prompt_path.read_text(encoding='utf-8')
    
    async def generate_questions(self, item: Dict, max_questions: int = 9) -> List[Dict]:
        """Generate questions for an item using LLM"""
        valid_features = {
            k: v for k, v in item.items() 
            if k not in ['id', 'image'] and v is not None
        }
        
        prompt = self.prompt_template.format(
            item_json=json.dumps(valid_features, indent=2, ensure_ascii=False),
            caption=item.get('caption', ''),
            keyword=item.get('keyword', '')
        )
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2048
        )
        
        if hasattr(response, 'usage') and response.usage:
            self.total_input_tokens += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens
        
        content = response.choices[0].message.content.strip()
        questions = self._parse_llm_response(content)
        
        if questions:
            validated = self._validate_answers(questions, valid_features)
            return validated[:max_questions]
        
        return []
    
    def _parse_llm_response(self, content: str) -> List[Dict]:
        """Parse LLM response to extract questions"""
        if '```json' in content:
            start = content.find('```json') + 7
            end = content.find('```', start)
            content = content[start:end].strip()
        elif '```' in content:
            start = content.find('```') + 3
            end = content.find('```', start)
            content = content[start:end].strip()
        
        try:
            questions = json.loads(content)
            if isinstance(questions, list):
                return questions
        except json.JSONDecodeError:
            pass
        
        return []
    
    def _validate_answers(self, questions: List[Dict], valid_features: Dict) -> List[Dict]:
        """Validate that answers contain exact feature values"""
        validated = []
        
        for q in questions:
            is_valid = True
            
            for feature_name in q.get('features_used', []):
                if feature_name in valid_features:
                    feature_value = str(valid_features[feature_name])
                    
                    if feature_name in ['caption', 'keyword']:
                        continue
                    
                    if feature_value not in q.get('answer', ''):
                        is_valid = False
                        break
            
            if is_valid:
                validated.append(q)
        
        return validated
    
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

