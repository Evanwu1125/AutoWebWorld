import os
import asyncio
import time
from typing import Dict, Any, List, Tuple, Optional
from .fsm_generator_agent import FSMGeneratorAgent
from .fsm_validator_agent import FSMValidatorAgent
from .fsm_improve_agent import FSMImproveAgent
from .utils import save_json, save_cost_report, slugify_theme


class FSMPerfectGenerator:
    def __init__(self,
                model: str = "gpt-5",
                target_score: int = 100,
                concurrent_count: int = 4,
                complexity_profile: Optional[Dict[str, Any]] = None,
                debug_output_dir: Optional[str] = None):

        self.model = model
        self.target_score = target_score
        self.concurrent_count = concurrent_count
        self.complexity_profile = complexity_profile
        self.debug_output_dir = debug_output_dir

        self.generator = FSMGeneratorAgent(model=model)
        self.validator = FSMValidatorAgent(model=model)
        self.improver = FSMImproveAgent(model=model)

    async def generate_initial_fsms(self, theme: str, output_dir: str) -> List[Dict[str, Any]]:
        print(f"🚀 开始并发生成 {self.concurrent_count} 个初始 FSM...")
        print(f"   主题: {theme}")

        tasks = [
            self.generator.call(
                theme=theme,
                process_id=i + 1,
                output_dir=f"{output_dir}/initial",
                complexity_profile=self.complexity_profile
            )
            for i in range(self.concurrent_count)
        ]

        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        duration = time.time() - start_time

        valid_fsms = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"❌ FSM {i+1} 生成失败: {result}")
            else:
                valid_fsms.append(result)
                print(f"✅ FSM {i+1} 生成成功")

        print(f"🎉 初始生成完成，耗时 {duration:.2f}s，成功 {len(valid_fsms)}/{self.concurrent_count}")
        return valid_fsms

    async def validate_all_fsms(self,
                                fsms: List[Dict[str, Any]],
                                theme: str,
                                output_dir: str) -> Dict[int, Dict[str, Any]]:
        print(f"\n🧪 开始验证 {len(fsms)} 个 FSM...")

        tasks = [
            self.validator.call(
                fsm_data=fsm,
                theme=theme,
                process_id=(i + 1),
                output_dir=f"{output_dir}/validation_{i+1}",
                complexity_profile=self.complexity_profile
            )
            for i, fsm in enumerate(fsms)
        ]

        start_time = time.time()
        val_results = await asyncio.gather(*tasks, return_exceptions=True)
        duration = time.time() - start_time

        validation_reports: Dict[int, Dict[str, Any]] = {}
        for i, rep in enumerate(val_results):
            if isinstance(rep, Exception):
                print(f"❌ FSM {i+1} 验证失败: {rep}")
                continue

            validation_dir = f"{output_dir}/validation_{i+1}"
            save_json(rep, f"{validation_dir}/validation_report.json")

            score = self.validator.get_score(rep)
            pages = len(fsms[i].get('pages', []))
            print(f"   FSM {i+1}: 分数 {score}, 页面数 {pages}")

            validation_reports[i] = rep

        print(f"✅ 验证完成，耗时 {duration:.2f}s")
        return validation_reports

    async def improve_fsm_once(self,
                            fsm_data: Dict[str, Any],
                            validation_report: Dict[str, Any],
                            fsm_id: int,
                            theme: str,
                            output_dir: str) -> Dict[str, Any]:
        print(f"� 开始改进 FSM {fsm_id}...")

        improved_fsm = await self.improver.call(
            original_fsm=fsm_data,
            validation_report=validation_report,
            theme=theme,
            process_id=fsm_id,
            output_dir=f"{output_dir}/fsm_{fsm_id}/improved",
            complexity_profile=self.complexity_profile
        )

        save_json(improved_fsm, f"{output_dir}/fsm_{fsm_id}/improved/fsm.json")
        print(f"✅ FSM {fsm_id} 改进完成")

        return improved_fsm

    def select_best_fsm(self,
                    fsms: List[Dict[str, Any]],
                    validation_reports: Dict[int, Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], int, int]:
        best_score = 0
        best_fsm = None
        best_fsm_id = 0

        print(f"\n📊 最终结果统计:")
        for i, rep in validation_reports.items():
            score = self.validator.get_score(rep)
            pages = len(fsms[i].get('pages', []))
            print(f"   FSM {i+1}: 分数 {score}, 页面数 {pages}")

            if score > best_score or (score == best_score and pages > len(best_fsm.get('pages', []))):
                best_score = score
                best_fsm = fsms[i]
                best_fsm_id = i + 1

        print(f"\n   最高分数: {best_score}")
        return best_fsm, best_score, best_fsm_id

    async def find_perfect_fsm(self,
                            theme: str,
                            output_dir: str = "fsm_perfect_outputs") -> Optional[Dict[str, Any]]:
        os.makedirs(output_dir, exist_ok=True)

        print("=" * 80)
        print(f"🎯 FSM 完美生成器启动")
        print(f"   主题: {theme}")
        _cp = self.complexity_profile or {}
        _pp = (_cp.get('pages') or {}).get('min')
        _tt = (_cp.get('terminals') or {}).get('count')
        print(f"   最小页面数: {_pp if _pp is not None else 'N/A'}")
        print(f"   最小终端页面数: {_tt if _tt is not None else 'N/A'}")
        print(f"   并发数量: {self.concurrent_count}")
        print(f"   目标分数: {self.target_score}")

        theme_slug = slugify_theme(theme)
        themed_output_dir = os.path.join(output_dir, theme_slug)
        os.makedirs(themed_output_dir, exist_ok=True)
        print(f"   输出目录: {themed_output_dir}")
        print("=" * 80)

        initial_fsms = await self.generate_initial_fsms(theme, themed_output_dir)
        if not initial_fsms:
            print("❌ 没有成功生成任何初始FSM")
            return None

        initial_validation_reports = await self.validate_all_fsms(
            initial_fsms, theme, f"{themed_output_dir}/initial"
        )

        for i, rep in initial_validation_reports.items():
            score = self.validator.get_score(rep)
            if score >= self.target_score:
                pages = len(initial_fsms[i].get('pages', []))
                final_path = f"{themed_output_dir}/perfect_fsm_{theme_slug}_{score}.json"
                save_json(initial_fsms[i], final_path)
                print(f"🏆 初始FSM已达标，直接采纳")
                print(f"   分数: {score}, 页面数: {pages}")
                print(f"   已保存: {final_path}")
                save_cost_report(
                    themed_output_dir, theme,
                    self.generator.get_usage_summary(),
                    self.validator.get_usage_summary(),
                    self.improver.get_usage_summary()
                )
                return initial_fsms[i]


        print(f"\n🔧 开始并发改进 {len(initial_fsms)} 个 FSM...")
        improve_tasks = [
            self.improve_fsm_once(
                fsm_data=initial_fsms[i],
                validation_report=initial_validation_reports[i],
                fsm_id=i + 1,
                theme=theme,
                output_dir=themed_output_dir
            )
            for i in initial_validation_reports.keys()
        ]

        improved_fsms_results = await asyncio.gather(*improve_tasks, return_exceptions=True)

        improved_fsms = []
        for i, result in enumerate(improved_fsms_results):
            if isinstance(result, Exception):
                print(f"❌ FSM {i+1} 改进失败: {result}")
                improved_fsms.append(initial_fsms[i])
            else:
                improved_fsms.append(result)

        improved_validation_reports = await self.validate_all_fsms(
            improved_fsms, theme, f"{themed_output_dir}/improved"
        )

        best_fsm, best_score, best_fsm_id = self.select_best_fsm(improved_fsms, improved_validation_reports)

        if best_fsm:
            final_path = f"{themed_output_dir}/best_fsm_{theme_slug}_{best_score}.json"
            save_json(best_fsm, final_path)
            print(f"\n🏆 最佳FSM已保存: {final_path}")
            print(f"   来源: FSM {best_fsm_id}")
            print(f"   分数: {best_score}")
        else:
            print("❌ 没有找到任何有效的FSM")

        save_cost_report(
            themed_output_dir, theme,
            self.generator.get_usage_summary(),
            self.validator.get_usage_summary(),
            self.improver.get_usage_summary()
        )

        return best_fsm


async def generate_perfect_fsm(theme: str,
                            model: str = "claude-4-sonnet",
                            concurrent_count: int = 16,
                            output_dir: str = "fsm_perfect_outputs",
                            complexity_profile: Optional[Dict[str, Any]] = None,
                            debug_llm_calls: bool = False) -> Optional[Dict[str, Any]]:
    """生成完美FSM的入口函数"""
    debug_dir = os.path.join(output_dir, "llm_calls") if debug_llm_calls else None

    generator = FSMPerfectGenerator(
        model=model,
        target_score=100,
        concurrent_count=concurrent_count,
        complexity_profile=complexity_profile,
        debug_output_dir=debug_dir
    )

    return await generator.find_perfect_fsm(theme=theme, output_dir=output_dir)


if __name__ == "__main__":
    import argparse
    import json

    async def main():
        parser = argparse.ArgumentParser(description="FSM Perfect Generator")
        parser.add_argument("--theme", required=True, help="应用主题")
        parser.add_argument("--model", default="claude-4-sonnet", help="LLM模型")
        parser.add_argument("--concurrent_count", type=int, default=16, help="并发数量")
        parser.add_argument("--output_dir", default="fsm_perfect_outputs", help="输出目录")
        parser.add_argument("--profile_json", default=None, help="复杂度配置 JSON 路径，可选")
        parser.add_argument("--debug_llm_calls", action="store_true", help="保存所有LLM调用的输入输出")
        args = parser.parse_args()

        profile_data = None
        if args.profile_json:
            with open(args.profile_json, "r", encoding="utf-8") as pf:
                profile_data = json.load(pf)

        perfect_fsm = await generate_perfect_fsm(
            theme=args.theme,
            model=args.model,
            concurrent_count=args.concurrent_count,
            output_dir=args.output_dir,
            complexity_profile=profile_data,
            debug_llm_calls=args.debug_llm_calls
        )

        if perfect_fsm:
            print(f"\n🎉 成功找到完美FSM!")
            print(f"   页面数: {len(perfect_fsm.get('pages', []))}")
            print(f"   应用名: {perfect_fsm.get('meta', {}).get('app', 'Unknown')}")
        else:
            print(f"\n😞 未能找到完美FSM，请检查配置")

    asyncio.run(main())
