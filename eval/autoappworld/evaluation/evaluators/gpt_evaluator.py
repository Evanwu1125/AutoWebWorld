"""
GPT Evaluator for WebVoyager-style task evaluation

This module provides a GPT-4 based evaluator that assesses whether a web task
was successfully completed by analyzing the task description, final answer,
and screenshots from the execution.
"""

from __future__ import annotations

import os
import glob
import base64
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

from litellm import acompletion

# Try to import ModelConfig from easyagent
try:
    from easyagent.config.base import ModelConfig
    HAS_EASYAGENT = True
except ImportError:
    HAS_EASYAGENT = False


# WebVoyager evaluation prompts
SYSTEM_PROMPT = """As an evaluator, you will be presented with three primary components to assist you in your role:

1. Web Task Instruction: This is a clear and specific directive provided in natural language, detailing the online activity to be carried out. These requirements may include conducting searches, verifying information, comparing prices, checking availability, or any other action relevant to the specified web service (such as Amazon, Apple, ArXiv, BBC News, Booking etc).

2. Result Screenshots: This is a visual representation of the screen showing the result or intermediate state of performing a web task. It serves as visual proof of the actions taken in response to the instruction.

3. Result Response: This is a textual response obtained after the execution of the web task. It serves as textual result in response to the instruction. If the screenshot already contains sufficient information to answer the task, prioritize the screenshot content and consider the response correct, without overemphasizing the textual response.

-- You DO NOT NEED to interact with web pages or perform actions such as booking flights or conducting searches on websites.
-- You SHOULD NOT make assumptions based on information not presented in the screenshot when comparing it to the instructions.
-- Your primary responsibility is to conduct a thorough assessment of the web task instruction against the outcome depicted in the screenshot and in the response, evaluating whether the actions taken align with the given instructions.
-- NOTE that the instruction may involve more than one task, for example, locating the garage and summarizing the review. Failing to complete either task, such as not providing a summary, should be considered unsuccessful.
-- NOTE that the screenshot is authentic, but the response provided by LLM is generated at the end of web browsing, and there may be discrepancies between the text and the screenshots.
-- Note the difference: 1) Result response may contradict the screenshot, then the content of the screenshot prevails. If the screenshot contains all information needed to answer the query, the response should be considered correct, 2) The content in the Result response is not mentioned on the screenshot, choose to believe the content.

You should elaborate on how you arrived at your final evaluation and then provide a definitive verdict on whether the task has been successfully accomplished, either as 'SUCCESS' or 'NOT SUCCESS'."""

USER_PROMPT_TEMPLATE = """TASK: {task}
Result Response: {answer}
{num_screenshots} screenshots at the end:"""


class GPTEvaluator:
    """
    GPT-4 based evaluator for web task completion assessment.
    
    This evaluator uses multimodal GPT-4 to analyze:
    - Task description
    - Final answer from the agent
    - Screenshots from the execution process
    
    Returns a binary score (0.0 or 1.0) and detailed reasoning.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        max_screenshots: int = 5,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        use_model_config: bool = True,
    ):
        """
        Initialize the GPT evaluator.

        Args:
            model: LiteLLM model name (e.g., "gpt-4o", "gpt-4o-mini")
            max_screenshots: Maximum number of screenshots to include (uses last N)
            api_key: OpenAI API key (optional, uses env var if not provided)
            api_base: API base URL (optional)
            use_model_config: Whether to load config from model_config.yaml (default: True)
        """
        self.max_screenshots = max_screenshots

        # 🔧 Try to load config from model_config.yaml
        self.model_config = None
        self.litellm_model = model  # Default to original model name

        if use_model_config and HAS_EASYAGENT:
            try:
                config = ModelConfig.load()
                model_cfg = config.get_model(model)

                # 🔑 Key: in the config returned by easyagent, model is already in "api_type/model_name" format
                # e.g.: gemini-3-flash-preview -> openai/gemini-3-flash-preview
                self.litellm_model = model_cfg.get("model", model)

                # Extract API info from config
                if not api_key and "api_key" in model_cfg:
                    api_key = model_cfg["api_key"]
                if not api_base and "api_base" in model_cfg:
                    api_base = model_cfg["api_base"]

                # Save full config for later use
                self.model_config = model_cfg
            except Exception as e:
                # If loading fails, use default params
                print(f"⚠️ Failed to load model config for '{model}': {e}")
                pass

        self.model = model  # Save original model name
        self.api_key = api_key
        self.api_base = api_base
    
    async def evaluate_async(
        self,
        task: str,
        answer: str,
        screenshot_dir: str,
        return_request: bool = False,
    ) -> Tuple[float, str] | Tuple[float, str, Dict[str, Any]]:
        """
        Asynchronously evaluate task completion using GPT-4.

        Args:
            task: Task description
            answer: Final answer from the agent
            screenshot_dir: Directory containing screenshots (input/*.png)
            return_request: If True, also return the request sent to the evaluator

        Returns:
            If return_request is False:
                Tuple of (score, reasoning):
                    - score: 1.0 (success) or 0.0 (failure)
                    - reasoning: GPT's detailed explanation
            If return_request is True:
                Tuple of (score, reasoning, request_info):
                    - score: 1.0 (success) or 0.0 (failure)
                    - reasoning: GPT's detailed explanation
                    - request_info: Dict containing the request details
        """
        # 📂 Step 1: Collect screenshots from input/ directory
        screenshot_paths = self._collect_screenshots(screenshot_dir)

        if not screenshot_paths:
            # No screenshots available, evaluate based on text only
            return await self._evaluate_text_only(task, answer, return_request=return_request)
        
        # 🖼️ Step 2: Prepare image URLs for LiteLLM
        # LiteLLM expects images in base64 format for better compatibility
        image_contents = []
        for path in screenshot_paths[-self.max_screenshots:]:  # Use last N screenshots
            # Read and encode image as base64
            try:
                with open(path, 'rb') as image_file:
                    image_data = image_file.read()
                    base64_image = base64.b64encode(image_data).decode('utf-8')
                    # Determine image format from file extension
                    ext = os.path.splitext(path)[1].lower()
                    if ext == '.png':
                        mime_type = 'image/png'
                    elif ext in ['.jpg', '.jpeg']:
                        mime_type = 'image/jpeg'
                    elif ext == '.webp':
                        mime_type = 'image/webp'
                    else:
                        mime_type = 'image/png'  # default to png

                    image_contents.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
                    })
            except Exception as e:
                print(f"⚠️ Failed to encode image {path}: {e}")
                continue
        
        # 📝 Step 3: Build user prompt
        user_prompt = USER_PROMPT_TEMPLATE.format(
            task=task,
            answer=answer,
            num_screenshots=len(image_contents)
        )
        
        # 💬 Step 4: Construct messages for LiteLLM
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    *image_contents
                ]
            },
            {
                "role": "user",
                "content": "Your verdict:\n."
            }
        ]
        
        # 🤖 Step 5: Call GPT-4 via LiteLLM
        kwargs = self._build_litellm_kwargs(messages)

        response = await acompletion(**kwargs)

        # 📄 Step 6: Extract response text
        gpt_response_text = response.choices[0].message.content

        # 🔍 Step 7: Parse verdict
        verdict = self._parse_verdict(gpt_response_text)

        # 📦 Step 8: Build request info if requested
        if return_request:
            # 🔥 Truncate base64 image data in messages for logging
            # Only keep first 100 characters of each base64 string to reduce log size
            messages_for_log = self._truncate_base64_in_messages(messages)

            request_info = {
                "model": self.litellm_model,
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": user_prompt,
                "task": task,
                "answer": answer,
                "num_screenshots": len(image_contents),
                "screenshot_paths": [p for p in screenshot_paths[-self.max_screenshots:]],
                "messages": messages_for_log,  # Messages with truncated base64 data
            }
            return verdict, gpt_response_text, request_info

        return verdict, gpt_response_text

    async def _evaluate_text_only(
        self,
        task: str,
        answer: str,
        return_request: bool = False,
    ) -> Tuple[float, str] | Tuple[float, str, Dict[str, Any]]:
        """Evaluate based on text only (no screenshots)."""
        user_prompt = USER_PROMPT_TEMPLATE.format(
            task=task,
            answer=answer,
            num_screenshots=0
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "user", "content": "Your verdict:\n."}
        ]

        kwargs = self._build_litellm_kwargs(messages)

        response = await acompletion(**kwargs)
        gpt_response_text = response.choices[0].message.content
        verdict = self._parse_verdict(gpt_response_text)

        # 📦 Build request info if requested
        if return_request:
            # No need to truncate for text-only (no images)
            request_info = {
                "model": self.litellm_model,
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": user_prompt,
                "task": task,
                "answer": answer,
                "num_screenshots": 0,
                "screenshot_paths": [],
                "messages": messages,
            }
            return verdict, gpt_response_text, request_info

        return verdict, gpt_response_text

    def _collect_screenshots(self, screenshot_dir: str) -> List[str]:
        """
        Collect screenshot paths from the input/ directory.

        Args:
            screenshot_dir: Base directory (e.g., "artifacts/task_xxx/")

        Returns:
            Sorted list of screenshot paths
        """
        input_dir = os.path.join(screenshot_dir, "input")
        if not os.path.exists(input_dir):
            return []

        # Find all PNG files in input/ directory
        pattern = os.path.join(input_dir, "*.png")
        screenshots = glob.glob(pattern)

        # Sort by filename (001.png, 002.png, ...)
        screenshots.sort()

        return screenshots

    def _build_litellm_kwargs(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build LiteLLM call kwargs.

        Args:
            messages: List of messages

        Returns:
            LiteLLM kwargs dict
        """
        # 🔑 Use litellm_model (may be in "openai/gemini-3-flash-preview" format)
        kwargs: Dict[str, Any] = {
            "model": self.litellm_model,
            "messages": messages,
        }

        # Add API key
        if self.api_key:
            kwargs["api_key"] = self.api_key

        # Add base URL
        if self.api_base:
            kwargs["api_base"] = self.api_base

        return kwargs

    def _truncate_base64_in_messages(self, messages: List[Dict[str, Any]], max_length: int = 100) -> List[Dict[str, Any]]:
        """
        Truncate base64 image data in messages for logging purposes.

        Args:
            messages: Original messages with full base64 data
            max_length: Maximum length of base64 string to keep (default: 100)

        Returns:
            Messages with truncated base64 data
        """
        import copy
        truncated_messages = copy.deepcopy(messages)

        for message in truncated_messages:
            if "content" in message and isinstance(message["content"], list):
                for item in message["content"]:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        if "image_url" in item and "url" in item["image_url"]:
                            url = item["image_url"]["url"]
                            # Check if it's a base64 data URL
                            if url.startswith("data:image/"):
                                # Extract the base64 part
                                if ";base64," in url:
                                    prefix, base64_data = url.split(";base64,", 1)
                                    # Truncate base64 data
                                    truncated_base64 = base64_data[:max_length] + f"...[truncated {len(base64_data) - max_length} chars]"
                                    item["image_url"]["url"] = f"{prefix};base64,{truncated_base64}"

        return truncated_messages

    def _parse_verdict(self, gpt_response: str) -> float:
        """
        Parse GPT response to extract verdict.

        Args:
            gpt_response: GPT's response text

        Returns:
            1.0 if SUCCESS, 0.0 otherwise
        """
        if "NOT SUCCESS" in gpt_response:
            return 0.0
        elif "SUCCESS" in gpt_response:
            return 1.0
        else:
            # Unable to determine, default to failure
            return 0.0

    def evaluate(
        self,
        task: str,
        answer: str,
        screenshot_dir: str,
        return_request: bool = False,
    ) -> Tuple[float, str] | Tuple[float, str, Dict[str, Any]]:
        """
        Synchronous wrapper for evaluate_async.

        Args:
            task: Task description
            answer: Final answer from the agent
            screenshot_dir: Directory containing screenshots
            return_request: If True, also return the request sent to the evaluator

        Returns:
            If return_request is False:
                Tuple of (score, reasoning)
            If return_request is True:
                Tuple of (score, reasoning, request_info)
        """
        import asyncio
        return asyncio.run(self.evaluate_async(task, answer, screenshot_dir, return_request=return_request))

