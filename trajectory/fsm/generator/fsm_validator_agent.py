from typing import Dict, Any, Optional
from .base_agent import BaseAgent
import json

class FSMValidatorAgent(BaseAgent):

    
    def __init__(self, 
                 system_prompt_path: str = "prompts/validation_system_prompt.txt",
                 instruction_prompt_path: str = "prompts/validation_instruction_prompt.txt",
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

    def _build_instruction_prompt(self, fsm_data: Dict[str, Any], complexity_profile: Optional[Dict[str, Any]] = None) -> str:
        template = self._get_instruction_template()
        fsm_json_str = json.dumps(fsm_data, ensure_ascii=False, indent=2)
        profile_json_str = json.dumps(complexity_profile, ensure_ascii=False, indent=2) if complexity_profile is not None else "{}"
        instruction = (
            template
            .replace("{fsm_json_here}", fsm_json_str)
            .replace("{COMPLEXITY_PROFILE_JSON}", profile_json_str)
        )
        return instruction

    async def call(self, fsm_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        if not fsm_data or not isinstance(fsm_data, dict):
            raise ValueError("fsm_data 必须是非空字典")

        print(f"🔍 FSMValidatorAgent: 开始验证 FSM")
        print(f"   页面数: {len(fsm_data.get('pages', []))}")

        system_prompt = self._get_system_prompt()
        instruction_prompt = self._build_instruction_prompt(fsm_data, kwargs.get('complexity_profile'))

        try:
            response = await self._call_llm(
                system_prompt=system_prompt,
                user_prompt=instruction_prompt,
                response_format={"type": "json_object"} if self.model.startswith("gpt-") else None
            )

            validation_report = self._force_json(response)
            score = validation_report.get("score", 0)
            print(f"✅ FSMValidatorAgent: 验证完成，评分: {score}")
            return validation_report

        except Exception as e:
            print(f"❌ FSMValidatorAgent: 验证失败: {e}")
            raise

    def get_score(self, validation_report: Dict[str, Any]) -> int:
        return validation_report.get("score", 0)

    def get_issues(self, validation_report: Dict[str, Any]) -> list:
        return validation_report.get("issues", [])
