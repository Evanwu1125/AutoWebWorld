import os
import asyncio
import time
from typing import Dict, Any, List, Tuple, Optional
from .fsm_generator_agent import FSMGeneratorAgent
from .fsm_validator_agent import FSMValidatorAgent
from .fsm_improve_agent import FSMImproveAgent
from .utils import (
    save_json,
    save_cost_report,
    slugify_theme,
    normalize_complexity_profile,
)


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
        self.complexity_profile = normalize_complexity_profile(complexity_profile)
        self.debug_output_dir = debug_output_dir

        generator_debug_dir = (
            os.path.join(debug_output_dir, "generator") if debug_output_dir else None
        )
        validator_debug_dir = (
            os.path.join(debug_output_dir, "validator") if debug_output_dir else None
        )
        improver_debug_dir = (
            os.path.join(debug_output_dir, "improver") if debug_output_dir else None
        )

        self.generator = FSMGeneratorAgent(model=model, debug_output_dir=generator_debug_dir)
        self.validator = FSMValidatorAgent(model=model, debug_output_dir=validator_debug_dir)
        self.improver = FSMImproveAgent(model=model, debug_output_dir=improver_debug_dir)

    async def generate_initial_fsms(self, theme: str, output_dir: str) -> List[Dict[str, Any]]:
        print(f"🚀 Starting concurrent generation of {self.concurrent_count} initial FSMs...")
        print(f"   Theme: {theme}")

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
                print(f"❌ FSM {i+1} generation failed: {result}")
            else:
                valid_fsms.append(result)
                print(f"✅ FSM {i+1} generated successfully")

        print(f"🎉 Initial generation complete, duration {duration:.2f}s, succeeded {len(valid_fsms)}/{self.concurrent_count}")
        return valid_fsms

    async def validate_all_fsms(self,
                                fsms: List[Dict[str, Any]],
                                output_dir: str) -> Dict[int, Dict[str, Any]]:
        print(f"\n🧪 Starting validation of {len(fsms)} FSMs...")

        tasks = [
            self.validator.call(
                fsm_data=fsm,
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
                print(f"❌ FSM {i+1} validation failed: {rep}")
                continue

            validation_dir = f"{output_dir}/validation_{i+1}"
            bfs_summary = rep.pop("_bfs_summary", None) if isinstance(rep, dict) else None
            save_json(rep, f"{validation_dir}/validation_report.json")
            if isinstance(bfs_summary, dict):
                save_json(bfs_summary, f"{validation_dir}/bfs_summary.json")

            score = self.validator.get_score(rep)
            pages = len(fsms[i].get('pages', []))
            print(f"   FSM {i+1}: score {score}, pages {pages}")

            validation_reports[i] = rep

        print(f"✅ Validation complete, duration {duration:.2f}s")
        return validation_reports

    async def improve_fsm_once(self,
                            fsm_data: Dict[str, Any],
                            validation_report: Dict[str, Any],
                            fsm_id: int,
                            theme: str,
                            output_dir: str) -> Dict[str, Any]:
        print(f"🔧 Starting improvement of FSM {fsm_id}...")

        improved_fsm = await self.improver.call(
            original_fsm=fsm_data,
            validation_report=validation_report,
            theme=theme,
            process_id=fsm_id,
            output_dir=f"{output_dir}/fsm_{fsm_id}/improved",
            complexity_profile=self.complexity_profile
        )

        save_json(improved_fsm, f"{output_dir}/fsm_{fsm_id}/improved/fsm.json")
        print(f"✅ FSM {fsm_id} improvement complete")

        return improved_fsm

    def select_best_fsm(self,
                    fsms: List[Dict[str, Any]],
                    validation_reports: Dict[int, Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], int, int]:
        best_score = 0
        best_fsm = None
        best_fsm_id = 0

        print(f"\n📊 Final results summary:")
        for i, rep in validation_reports.items():
            score = self.validator.get_score(rep)
            pages = len(fsms[i].get('pages', []))
            print(f"   FSM {i+1}: score {score}, pages {pages}")

            current_best_pages = len(best_fsm.get('pages', [])) if isinstance(best_fsm, dict) else -1
            if score > best_score or (score == best_score and pages > current_best_pages):
                best_score = score
                best_fsm = fsms[i]
                best_fsm_id = i + 1

        print(f"\n   Best score: {best_score}")
        return best_fsm, best_score, best_fsm_id

    async def find_perfect_fsm(self,
                            theme: str,
                            output_dir: str = "fsm_perfect_outputs") -> Optional[Dict[str, Any]]:
        os.makedirs(output_dir, exist_ok=True)

        print("=" * 80)
        print(f"🎯 FSM Perfect Generator started")
        print(f"   Theme: {theme}")
        _cp = self.complexity_profile or {}
        _pp = (_cp.get('pages') or {}).get('min')
        _tt = (_cp.get('terminals') or {}).get('count')
        print(f"   Minimum pages: {_pp if _pp is not None else 'N/A'}")
        print(f"   Minimum terminal pages: {_tt if _tt is not None else 'N/A'}")
        print(f"   Concurrent count: {self.concurrent_count}")
        print(f"   Target score: {self.target_score}")

        theme_slug = slugify_theme(theme)
        themed_output_dir = os.path.join(output_dir, theme_slug)
        os.makedirs(themed_output_dir, exist_ok=True)
        print(f"   Output directory: {themed_output_dir}")
        print("=" * 80)

        initial_fsms = await self.generate_initial_fsms(theme, themed_output_dir)
        if not initial_fsms:
            print("❌ No initial FSM was generated successfully")
            return None

        initial_validation_reports = await self.validate_all_fsms(
            initial_fsms, f"{themed_output_dir}/initial"
        )
        if not initial_validation_reports:
            print("❌ All initial validations failed, cannot proceed with improvement")
            save_cost_report(
                themed_output_dir, theme,
                self.generator.get_usage_summary(),
                self.validator.get_usage_summary(),
                self.improver.get_usage_summary()
            )
            return None

        for i, rep in initial_validation_reports.items():
            score = self.validator.get_score(rep)
            if score >= self.target_score:
                pages = len(initial_fsms[i].get('pages', []))
                final_path = f"{themed_output_dir}/perfect_fsm_{theme_slug}_{score}.json"
                save_json(initial_fsms[i], final_path)
                print(f"🏆 Initial FSM meets target score, adopting directly")
                print(f"   Score: {score}, pages: {pages}")
                print(f"   Saved: {final_path}")
                save_cost_report(
                    themed_output_dir, theme,
                    self.generator.get_usage_summary(),
                    self.validator.get_usage_summary(),
                    self.improver.get_usage_summary()
                )
                return initial_fsms[i]


        print(f"\n🔧 Starting concurrent improvement of {len(initial_fsms)} FSMs...")
        improve_source_indices = list(initial_validation_reports.keys())
        improve_tasks = [
            self.improve_fsm_once(
                fsm_data=initial_fsms[i],
                validation_report=initial_validation_reports[i],
                fsm_id=i + 1,
                theme=theme,
                output_dir=themed_output_dir
            )
            for i in improve_source_indices
        ]

        improved_fsms_results = await asyncio.gather(*improve_tasks, return_exceptions=True)

        improved_fsms = []
        for pos, result in enumerate(improved_fsms_results):
            source_index = improve_source_indices[pos]
            if isinstance(result, Exception):
                print(f"❌ FSM {source_index+1} improvement failed: {result}")
                improved_fsms.append(initial_fsms[source_index])
            else:
                improved_fsms.append(result)

        improved_validation_reports = await self.validate_all_fsms(
            improved_fsms, f"{themed_output_dir}/improved"
        )

        best_fsm, best_score, best_fsm_id = self.select_best_fsm(improved_fsms, improved_validation_reports)

        if best_fsm:
            final_path = f"{themed_output_dir}/best_fsm_{theme_slug}_{best_score}.json"
            save_json(best_fsm, final_path)
            print(f"\n🏆 Best FSM saved: {final_path}")
            print(f"   Source: FSM {best_fsm_id}")
            print(f"   Score: {best_score}")
        else:
            print("❌ No valid FSM was found")

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
    """Entry function for generating a perfect FSM"""
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
        parser.add_argument("--theme", required=True, help="Application theme")
        parser.add_argument("--model", default="claude-4-sonnet", help="LLM model")
        parser.add_argument("--concurrent_count", type=int, default=16, help="Concurrent count")
        parser.add_argument("--output_dir", default="fsm_perfect_outputs", help="Output directory")
        parser.add_argument("--profile_json", default=None, help="Complexity profile JSON path, optional")
        parser.add_argument("--debug_llm_calls", action="store_true", help="Save input/output of all LLM calls")
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
            print(f"\n🎉 Successfully found a perfect FSM!")
            print(f"   Pages: {len(perfect_fsm.get('pages', []))}")
            print(f"   App name: {perfect_fsm.get('meta', {}).get('app', 'Unknown')}")
        else:
            print(f"\n😞 Failed to find a perfect FSM, please check configuration")

    asyncio.run(main())
