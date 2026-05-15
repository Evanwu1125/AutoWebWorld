from __future__ import annotations
import os
import re
import json
import time
import traceback
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from openai import AsyncOpenAI
from datetime import datetime
from pathlib import Path


class BaseAgent(ABC):
    def __init__(self,
                 model: str = "gpt-5",
                 temperature: float = 1,
                 max_tokens: int = 128000,
                 base_url: str = "",
                 debug_output_dir: Optional[str] = None):

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.debug_output_dir = debug_output_dir
        self._client: Optional[AsyncOpenAI] = None

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.call_count = 0

        self.pricing = {
            "gpt-5": {"input": 1.25, "output": 10.0},
            "gpt-5-chat": {"input": 1.5, "output": 12.0},
            "claude-4-sonnet": {"input": 3.0, "output": 15.0},
            "claude-sonnet-4-5": {"input": 3.0, "output": 15.0},
            "gemini-2.5-flash": {"input": 0.3, "output": 0.252},
            "deepseek-v3.2-exp": {"input": 0.25, "output": 0.37}
        }

    async def _get_client(self) -> AsyncOpenAI:

        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("Missing OPENAI_API_KEY environment variable")

            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=self.base_url
            )
        return self._client

    def _force_json(self, text: str) -> Dict[str, Any]:

        try:
            return json.loads(text)
        except Exception:
            cleaned_text = text.strip()
            if cleaned_text.startswith('```json'):
                cleaned_text = cleaned_text[7:]
            elif cleaned_text.startswith('```'):
                cleaned_text = cleaned_text[3:]
            if cleaned_text.endswith('```'):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()

            try:
                return json.loads(cleaned_text)
            except Exception:
                json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                matches = re.findall(json_pattern, cleaned_text, re.DOTALL)

                for match in matches:
                    try:
                        return json.loads(match)
                    except Exception:
                        continue

                start = cleaned_text.find('{')
                if start != -1:
                    brace_count = 0
                    for i, char in enumerate(cleaned_text[start:], start):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                try:
                                    return json.loads(cleaned_text[start:i+1])
                                except Exception:
                                    break

                raise ValueError(f"Unable to extract a valid JSON object from text: {text[:200]}...")

    def _load_text(self, path: str) -> str:
        # If path is relative, resolve it relative to the env_generator directory
        if not os.path.isabs(path):
            # Get the directory where this file (base_agent.py) is located
            current_dir = Path(__file__).parent
            # Go up to env_generator directory
            env_generator_dir = current_dir.parent
            # Resolve the path relative to env_generator
            path = str(env_generator_dir / path)

        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {path}")
        except UnicodeDecodeError as e:
            raise ValueError(f"File encoding error: {path}") from e

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        model_key = self.model
        if model_key not in self.pricing:
            for key in self.pricing.keys():
                if self.model.startswith(key):
                    model_key = key
                    break
            else:
                print(f"⚠️  Model {self.model} not found in pricing table, using default price")
                return (input_tokens / 1_000_000 * 3.0) + (output_tokens / 1_000_000 * 15.0)

        prices = self.pricing[model_key]
        input_cost = (input_tokens / 1_000_000) * prices["input"]
        output_cost = (output_tokens / 1_000_000) * prices["output"]
        return input_cost + output_cost

    def _update_usage_stats(self, input_tokens: int, output_tokens: int, cost: float):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost
        self.call_count += 1

    def get_usage_summary(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "total_calls": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost_usd": round(self.total_cost, 4),
            "total_cost_cny": round(self.total_cost * 7.3, 2),
            "average_cost_per_call_usd": round(self.total_cost / self.call_count, 4) if self.call_count > 0 else 0
        }

    def print_usage_summary(self):
        summary = self.get_usage_summary()
        print("\n" + "=" * 60)
        print(f"💰 {self.__class__.__name__} Usage Statistics")
        print("=" * 60)
        print(f"Model: {summary['model']}")
        print(f"Total calls: {summary['total_calls']}")
        print(f"Input tokens: {summary['total_input_tokens']:,}")
        print(f"Output tokens: {summary['total_output_tokens']:,}")
        print(f"Total tokens: {summary['total_tokens']:,}")
        print(f"Total cost: ${summary['total_cost_usd']} (¥{summary['total_cost_cny']})")
        print(f"Average cost per call: ${summary['average_cost_per_call_usd']}")
        print("=" * 60 + "\n")

    def reset_usage_stats(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.call_count = 0

    async def _call_llm(self,
                        system_prompt: str,
                        user_prompt: str,
                        response_format: Optional[Dict[str, str]] = None,
                        image_b64: Optional[str] = None) -> str:

        client = await self._get_client()
        start_time = time.time()

        # Build user content — plain text or multimodal list when an image is provided
        if image_b64:
            user_content = [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ]
        else:
            user_content = user_prompt

        if self.model.startswith("gpt-"):
            request_params = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ]
            }

        elif self.model.startswith("claude-"):
            request_params = {
                "model": self.model,
                "max_completion_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ]
            }

        else:
            # Default to OpenAI-compatible format (suitable for gemini and other models)
            request_params = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ]
            }

        if response_format and self.model.startswith("gpt-"):
            request_params["response_format"] = response_format

        try:
            resp = await client.chat.completions.create(**request_params)
            duration = time.time() - start_time
            content = resp.choices[0].message.content
            usage = resp.usage
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            cost = self._calculate_cost(input_tokens, output_tokens)
            self._update_usage_stats(input_tokens, output_tokens, cost)

            print(f"🤖 {self.__class__.__name__}: LLM call completed")
            print(f"   Duration: {duration:.2f}s")
            print(f"   Input tokens: {input_tokens:,}")
            print(f"   Output tokens: {output_tokens:,}")
            print(f"   This call cost: ${cost:.4f} (¥{cost * 7.3:.2f})")
            print(f"   Cumulative cost: ${self.total_cost:.4f} (¥{self.total_cost * 7.3:.2f})")

            self._save_llm_debug_record({
                "agent": self.__class__.__name__,
                "model": self.model,
                "base_url": self.base_url,
                "duration_seconds": round(duration, 3),
                "request": request_params,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response_text": content,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_usd": round(cost, 6),
                },
            })

            return content

        except Exception as e:
            self._save_llm_debug_record({
                "agent": self.__class__.__name__,
                "model": self.model,
                "base_url": self.base_url,
                "request": request_params,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })
            print(f"❌ {self.__class__.__name__}: LLM call failed: {e}")
            print(f"Full error traceback: {traceback.format_exc()}")
            raise

    def _save_llm_debug_record(self, payload: Dict[str, Any]) -> None:
        if not self.debug_output_dir:
            return
        try:
            os.makedirs(self.debug_output_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{ts}_{self.__class__.__name__.lower()}.json"
            path = os.path.join(self.debug_output_dir, filename)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️  {self.__class__.__name__}: Failed to save LLM debug record: {e}")

    @abstractmethod
    async def call(self, **kwargs) -> Dict[str, Any]:

        pass

    def save_output(self,
                   data: Dict[str, Any],
                   theme: str,
                   process_id: int,
                   output_dir: str = "outputs",
                   file_suffix: str = "") -> str:

        os.makedirs(output_dir, exist_ok=True)
        clean_theme = re.sub(r'[^\w\-_]', '_', theme.lower())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        agent_name = self.__class__.__name__.lower().replace("agent", "")
        filename = f"{clean_theme}_process_{process_id:03d}_{agent_name}{file_suffix}_{timestamp}.json"
        file_path = os.path.join(output_dir, filename)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"💾 {self.__class__.__name__}: File saved to {file_path}")
            return file_path

        except Exception as e:
            print(f"❌ {self.__class__.__name__}: File save failed: {e}")
            raise OSError(f"Unable to save file to {file_path}: {e}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model}, temperature={self.temperature})"
