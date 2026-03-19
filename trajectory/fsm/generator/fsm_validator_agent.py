from typing import Dict, Any, Optional
from .base_agent import BaseAgent
from .reachability_checker import build_reachability_issues, run_bfs_summary
from .scoring import apply_reachability_score
from .utils import normalize_complexity_profile
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
        normalized_profile = normalize_complexity_profile(complexity_profile)
        profile_json_str = json.dumps(normalized_profile, ensure_ascii=False, indent=2)
        instruction = (
            template
            .replace("{fsm_json_here}", fsm_json_str)
            .replace("{COMPLEXITY_PROFILE_JSON}", profile_json_str)
        )
        return instruction

    async def call(self, fsm_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        if not fsm_data or not isinstance(fsm_data, dict):
            raise ValueError("fsm_data must be a non-empty dict")

        print(f"🔍 FSMValidatorAgent: Starting FSM validation")
        print(f"   Pages: {len(fsm_data.get('pages', []))}")

        normalized_profile = normalize_complexity_profile(kwargs.get('complexity_profile'))
        system_prompt = self._get_system_prompt()
        instruction_prompt = self._build_instruction_prompt(fsm_data, normalized_profile)

        try:
            response = await self._call_llm(
                system_prompt=system_prompt,
                user_prompt=instruction_prompt,
                response_format={"type": "json_object"} if self.model.startswith("gpt-") else None
            )

            validation_report = self._force_json(response)
            bfs_summary = run_bfs_summary(fsm_data=fsm_data, max_depth=40)
            local_reachability_issues = build_reachability_issues(bfs_summary)

            validation_report = apply_reachability_score(
                base_report=validation_report,
                local_reachability_issues=local_reachability_issues,
                reachability=bfs_summary.get("reachability", {}),
            )
            validation_report["_bfs_summary"] = bfs_summary

            score = validation_report.get("score", 0)
            print(f"✅ FSMValidatorAgent: Validation complete, score: {score}")
            return validation_report

        except Exception as e:
            print(f"❌ FSMValidatorAgent: Validation failed: {e}")
            raise

    def get_score(self, validation_report: Dict[str, Any]) -> int:
        raw_score = validation_report.get("score", 0)
        try:
            return int(float(raw_score))
        except (TypeError, ValueError):
            return 0

    def get_issues(self, validation_report: Dict[str, Any]) -> list:
        return validation_report.get("issues", [])
