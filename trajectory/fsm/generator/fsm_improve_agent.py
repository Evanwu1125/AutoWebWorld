import json
from typing import Dict, Any, Optional
from .base_agent import BaseAgent


class FSMImproveAgent(BaseAgent):

    def __init__(self,
                 system_prompt_path: str = "prompts/improve_system_prompt.txt",
                 instruction_prompt_path: str = "prompts/improve_instruction_prompt.txt",
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

    def _build_instruction_prompt(self, original_fsm: Dict[str, Any], validation_report: Dict[str, Any], complexity_profile: Optional[Dict[str, Any]] = None) -> str:

        template = self._get_instruction_template()
        original_fsm_str = json.dumps(original_fsm, ensure_ascii=False, indent=2)
        validation_report_str = json.dumps(validation_report, ensure_ascii=False, indent=2)

        profile_json_str = json.dumps(complexity_profile, ensure_ascii=False, indent=2) if complexity_profile is not None else "{}"
        instruction = (
            template
            .replace("{ORIGINAL_FSM_JSON}", original_fsm_str)
            .replace("{VALIDATION_JSON}", validation_report_str)
            .replace("{COMPLEXITY_PROFILE_JSON}", profile_json_str)
        )

        return instruction

    async def call(self, original_fsm: Dict[str, Any], validation_report: Dict[str, Any], **kwargs) -> Dict[str, Any]:

        if not original_fsm or not isinstance(original_fsm, dict):
            raise ValueError("original_fsm 必须是非空字典")

        if not validation_report or not isinstance(validation_report, dict):
            raise ValueError("validation_report 必须是非空字典")

        score = validation_report.get("score", 0)
        issues = validation_report.get("issues", [])

        print(f"🔧 FSMImproveAgent: 开始改进 FSM")
        print(f"   当前评分: {score}")
        print(f"   问题数量: {len(issues)}")

        system_prompt = self._get_system_prompt()
        instruction_prompt = self._build_instruction_prompt(original_fsm, validation_report, kwargs.get('complexity_profile'))

        try:
            response = await self._call_llm(
                system_prompt=system_prompt,
                user_prompt=instruction_prompt,
                response_format={"type": "json_object"} if self.model.startswith("gpt-") else None
            )

            improved_fsm = self._force_json(response)

            self._validate_improved_fsm(improved_fsm, original_fsm)

            print(f"✅ FSMImproveAgent: FSM 改进完成")
            print(f"   原始页面数: {len(original_fsm.get('pages', []))}")
            print(f"   改进页面数: {len(improved_fsm.get('pages', []))}")

            process_id = kwargs.get('process_id')
            if process_id is not None:
                output_dir = kwargs.get('output_dir', 'outputs')
                theme = kwargs.get('theme', 'unknown')
                self.save_output(
                    data=improved_fsm,
                    theme=theme,
                    process_id=process_id,
                    output_dir=output_dir,
                    file_suffix="_improved"
                )

            return improved_fsm

        except Exception as e:
            print(f"❌ FSMImproveAgent: 改进失败: {e}")
            raise

    def _validate_improved_fsm(self, improved_fsm: Dict[str, Any], original_fsm: Dict[str, Any]) -> None:

        required_fields = ["meta", "pages"]
        for field in required_fields:
            if field not in improved_fsm:
                raise ValueError(f"改进后的 FSM 缺少必要字段: {field}")

        meta = improved_fsm["meta"]
        required_meta_fields = ["app", "initial_page_id", "terminal_pages"]
        for field in required_meta_fields:
            if field not in meta:
                raise ValueError(f"改进后的 FSM meta 缺少必要字段: {field}")

        pages = improved_fsm["pages"]
        if not isinstance(pages, list) or len(pages) == 0:
            raise ValueError("改进后的 FSM pages 必须是非空列表")

        original_page_count = len(original_fsm.get("pages", []))
        improved_page_count = len(pages)

        if improved_page_count < original_page_count * 0.5:
            print(f"⚠️  警告: 改进后页面数量大幅减少 ({original_page_count} -> {improved_page_count})")

        for i, page in enumerate(pages):
            if not isinstance(page, dict):
                raise ValueError(f"改进后页面 {i} 必须是字典对象")

            required_page_fields = ["id", "signature_schema", "actions"]
            for field in required_page_fields:
                if field not in page:
                    raise ValueError(f"改进后页面 {i} 缺少必要字段: {field}")

        print(f"✅ 改进后 FSM 基础结构验证通过")

    def __repr__(self) -> str:
        return f"FSMImproveAgent(model={self.model}, system_prompt={self.system_prompt_path})"

