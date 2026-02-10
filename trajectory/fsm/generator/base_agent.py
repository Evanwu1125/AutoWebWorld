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
                 base_url: str = "https://newapi.deepwisdom.ai/v1"):

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
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
                raise RuntimeError("缺少 OPENAI_API_KEY 环境变量")

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

                raise ValueError(f"无法从文本中提取有效的 JSON 对象: {text[:200]}...")

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
            raise FileNotFoundError(f"找不到文件: {path}")
        except UnicodeDecodeError:
            raise UnicodeDecodeError(f"文件编码错误: {path}")

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        model_key = self.model
        if model_key not in self.pricing:
            for key in self.pricing.keys():
                if self.model.startswith(key):
                    model_key = key
                    break
            else:
                print(f"⚠️  模型 {self.model} 未在价格表中，使用默认价格")
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
        print(f"💰 {self.__class__.__name__} 使用统计")
        print("=" * 60)
        print(f"模型: {summary['model']}")
        print(f"总调用次数: {summary['total_calls']}")
        print(f"输入 tokens: {summary['total_input_tokens']:,}")
        print(f"输出 tokens: {summary['total_output_tokens']:,}")
        print(f"总 tokens: {summary['total_tokens']:,}")
        print(f"总成本: ${summary['total_cost_usd']} (¥{summary['total_cost_cny']})")
        print(f"平均每次调用成本: ${summary['average_cost_per_call_usd']}")
        print("=" * 60 + "\n")

    def reset_usage_stats(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.call_count = 0

    async def _call_llm(self,
                        system_prompt: str,
                        user_prompt: str,
                        response_format: Optional[Dict[str, str]] = None) -> str:

        client = await self._get_client()
        start_time = time.time()
        if self.model.startswith("gpt-"):
            request_params = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }

        elif self.model.startswith("claude-"):
            request_params = {
                "model": self.model,
                "max_completion_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }

        else:
            # 默认使用 OpenAI 兼容格式（适用于 gemini 等其他模型）
            request_params = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
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

            print(f"🤖 {self.__class__.__name__}: LLM 调用完成")
            print(f"   耗时: {duration:.2f}s")
            print(f"   输入 tokens: {input_tokens:,}")
            print(f"   输出 tokens: {output_tokens:,}")
            print(f"   本次成本: ${cost:.4f} (¥{cost * 7.3:.2f})")
            print(f"   累计成本: ${self.total_cost:.4f} (¥{self.total_cost * 7.3:.2f})")

            return content

        except Exception as e:
            print(f"❌ {self.__class__.__name__}: LLM 调用失败: {e}")
            print(f"完整错误信息: {traceback.format_exc()}")
            raise

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

            print(f"💾 {self.__class__.__name__}: 文件已保存到 {file_path}")
            return file_path

        except Exception as e:
            print(f"❌ {self.__class__.__name__}: 文件保存失败: {e}")
            raise OSError(f"无法保存文件到 {file_path}: {e}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model}, temperature={self.temperature})"
