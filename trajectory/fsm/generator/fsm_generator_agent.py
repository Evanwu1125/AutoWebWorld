import traceback
import json

from typing import Dict, Any, Optional
from .base_agent import BaseAgent
from .utils import normalize_complexity_profile


class FSMGeneratorAgent(BaseAgent):
    def __init__(self,
                 system_prompt_path: str = "prompts/fsm_system_prompt.txt",
                 instruction_prompt_path: str = "prompts/fsm_instruction_prompt.txt",
                 **kwargs):

        super().__init__(**kwargs)
        self.system_prompt_path = system_prompt_path
        self.instruction_prompt_path = instruction_prompt_path

        self._system_prompt = None
        self._instruction_template = None

    def _get_system_prompt(self) -> str:
        if self._system_prompt is None:
            self._system_prompt = self._load_text(self.system_prompt_path)
        return self._system_prompt

    def _get_instruction_template(self) -> str:
        if self._instruction_template is None:
            self._instruction_template = self._load_text(self.instruction_prompt_path)
        return self._instruction_template

    def _build_instruction_prompt(self, theme: str, complexity_profile: Optional[Dict[str, Any]] = None) -> str:
        template = self._get_instruction_template()
        normalized_profile = normalize_complexity_profile(complexity_profile)
        profile_json_str = json.dumps(normalized_profile, ensure_ascii=False, indent=2)

        instruction = (
            template
            .replace("{theme}", theme)
            .replace("{COMPLEXITY_PROFILE_JSON}", profile_json_str)
        )

        return instruction

    async def call(self, theme: str, **kwargs) -> Dict[str, Any]:

        print(f"🚀 FSMGeneratorAgent: Starting FSM generation")
        print(f"   Theme: {theme}")

        normalized_profile = normalize_complexity_profile(kwargs.get('complexity_profile'))
        system_prompt = self._get_system_prompt()
        instruction_prompt = self._build_instruction_prompt(theme, normalized_profile)

        try:
            response = await self._call_llm(
                system_prompt=system_prompt,
                user_prompt=instruction_prompt,
                response_format={"type": "json_object"} if self.model.startswith("gpt-") else None
            )

            fsm_data = self._force_json(response)

            process_id = kwargs.get('process_id')
            meta = fsm_data.setdefault('meta', {})
            meta['complexity_profile'] = normalized_profile

            if process_id is not None:
                output_dir = kwargs.get('output_dir', 'outputs')
                self.save_output(
                    data=fsm_data,
                    theme=theme,
                    process_id=process_id,
                    output_dir=output_dir,
                    file_suffix="_fsm"
                )

            return fsm_data

        except Exception as e:
            print(f"❌ FSMGeneratorAgent: Generation failed: {e}")
            print(f"Full error traceback: {traceback.format_exc()}")
            raise


    def __repr__(self) -> str:
        return f"FSMGeneratorAgent(model={self.model}, system_prompt={self.system_prompt_path})"
