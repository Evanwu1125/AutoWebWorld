"""QA Generator - Main processing logic"""
import asyncio
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from .trajectory_processor import TrajectoryProcessor
from .question_generator import QuestionGenerator
from .vlm_verifier import VLMVerifier
from .utils import (
    load_json, save_json, get_item_by_id, 
    build_item_to_collection_map, count_by_type
)


class QAGenerator:
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        llm_model: str,
        vlm_model: str,
        max_questions: int = 9,
        concurrency: int = 10,
        skip_vlm: bool = False
    ):
        self.processor = TrajectoryProcessor()
        self.question_gen = QuestionGenerator(api_key, base_url, llm_model)
        self.vlm_verifier = VLMVerifier(api_key, base_url, vlm_model) if not skip_vlm else None

        self.max_questions = max_questions
        self.concurrency = concurrency
        self.skip_vlm = skip_vlm
    
    async def process_project(
        self,
        grounding_dir: Path,
        caption_file: Path,
        output_dir: Path
    ) -> Dict:
        """Process a single project to generate QA queries"""
        print(f"\n{'='*60}")
        print(f"Processing Project: {grounding_dir.parent.name}")
        print(f"{'='*60}\n")

        caption_data = load_json(caption_file)
        item_to_collection = build_item_to_collection_map(caption_data)

        successful_trajectories = self._find_successful_trajectories(
            grounding_dir,
            item_to_collection
        )

        print(f"Found {len(successful_trajectories)} successful trajectories")

        # Deduplicate by item_id: keep only the first trajectory for each item
        unique_trajectories = {}
        for traj in successful_trajectories:
            item_id = traj['item_id']
            if item_id not in unique_trajectories:
                unique_trajectories[item_id] = traj

        duplicates_count = len(successful_trajectories) - len(unique_trajectories)
        if duplicates_count > 0:
            print(f"Deduplicated: {duplicates_count} duplicate item_id(s) removed")
        print(f"Processing {len(unique_trajectories)} unique items\n")

        failed_items = []
        total_items = len(unique_trajectories)
        started_count = 0
        progress_lock = asyncio.Lock()

        semaphore = asyncio.Semaphore(self.concurrency)

        async def process_with_semaphore(traj_info):
            nonlocal started_count
            # Get current progress index before starting
            async with progress_lock:
                current_idx = started_count
                started_count += 1

            async with semaphore:
                result = await self._process_single_trajectory(
                    traj_info,
                    caption_data,
                    item_to_collection,
                    output_dir,
                    total_items,
                    current_idx
                )
                return result

        tasks = [process_with_semaphore(t) for t in unique_trajectories.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = 0
        for result in results:
            if isinstance(result, Exception):
                print(f"  ✗ Exception: {result}")
                continue

            if result:
                if result.get('success'):
                    success_count += 1
                else:
                    failed_items.append(result.get('item_id'))

        # Generate summary from saved files
        summary = self._generate_summary(output_dir, caption_data, failed_items)
        save_json(summary, output_dir / 'summary.json')

        save_json(self.question_gen.get_usage_stats(), output_dir / 'api_cost_llm.json')
        if not self.skip_vlm:
            save_json(self.vlm_verifier.get_usage_stats(), output_dir / 'api_cost_vlm.json')

        print(f"\n{'='*60}")
        print(f"✅ QA Generation Complete!")
        print(f"   Total items processed: {success_count}")
        print(f"   Total QA pairs: {summary.get('total_qa_pairs', 0)}")
        print(f"   Output directory: {output_dir}")
        print(f"{'='*60}\n")

        return summary
    
    def _find_successful_trajectories(
        self,
        grounding_dir: Path,
        item_to_collection: Dict[str, str]
    ) -> List[Dict]:
        """Find all successful trajectories"""
        successful = []
        
        for traj_dir in sorted(grounding_dir.iterdir()):
            if not traj_dir.is_dir():
                continue
            
            if not self.processor.is_trajectory_success(traj_dir):
                continue
            
            item_id = self.processor.extract_item_id_from_dirname(traj_dir.name)
            if not item_id or item_id not in item_to_collection:
                continue
            
            successful.append({
                'item_id': item_id,
                'traj_dir': traj_dir
            })
        
        return successful
    
    async def _process_single_trajectory(
        self,
        traj_info: Dict,
        caption_data: Dict,
        item_to_collection: Dict[str, str],
        output_dir: Path,
        total_items: int = 0,
        current_index: int = 0
    ) -> Dict:
        """Process a single trajectory"""
        item_id = traj_info['item_id']
        traj_dir = traj_info['traj_dir']

        # Display progress
        if total_items > 0:
            progress_str = f"[{current_index + 1}/{total_items}]"
            print(f"\n{progress_str} Processing item: {item_id}")
        else:
            print(f"\nProcessing item: {item_id}")

        try:
            result = load_json(traj_dir / 'result.json')
            bfs = load_json(traj_dir / 'bfs.json')

            detail_action_idx, detail_action = self.processor.find_detail_action(result)
            if not detail_action:
                print(f"  ✗ No DETAIL action found")
                return {"success": False, "item_id": item_id}

            print(f"  ✓ Found DETAIL action: {detail_action['action_id']}")

            truncated_trajectory = self.processor.truncate_trajectory(result, detail_action_idx)

            collection_name = item_to_collection[item_id]
            item = get_item_by_id(caption_data[collection_name], item_id)

            if not item:
                print(f"  ✗ Item not found in caption data")
                return {"success": False, "item_id": item_id}

            print(f"  Generating questions...")
            questions = await self.question_gen.generate_questions(item, self.max_questions)
            print(f"  ✓ Generated {len(questions)} questions")

            if not questions:
                print(f"  ✗ No questions generated")
                return {"success": False, "item_id": item_id}

            screenshot_path = self.processor.get_target_screenshot(detail_action, traj_dir)
            if not screenshot_path or not screenshot_path.exists():
                print(f"  ✗ Screenshot not found")
                return {"success": False, "item_id": item_id}

            if self.skip_vlm:
                verified_qa_pairs = [
                    {**q, 'vlm_verification': {'skipped': True}}
                    for q in questions
                ]
            else:
                print(f"  Verifying with VLM...")
                verified_qa_pairs = await self.vlm_verifier.batch_verify(
                    screenshot_path,
                    questions
                )
                print(f"  ✓ Passed VLM: {len(verified_qa_pairs)}/{len(questions)}")

            if not verified_qa_pairs:
                print(f"  ✗ No QA pairs passed VLM verification")
                return {"success": False, "item_id": item_id}

            # Create output directory for this trajectory
            traj_output_dir = output_dir / traj_dir.name
            traj_output_dir.mkdir(parents=True, exist_ok=True)

            # Save qa_pairs.json
            save_json(verified_qa_pairs, traj_output_dir / 'qa_pairs.json')

            # Save truncated bfs.json
            truncated_bfs = {
                'item_id': item_id,
                'actions': bfs['trajectory'][:detail_action_idx + 1]
            }
            save_json(truncated_bfs, traj_output_dir / 'bfs.json')

            # Save truncated result.json
            save_json(truncated_trajectory, traj_output_dir / 'result.json')

            # Save item.json
            save_json(item, traj_output_dir / 'item.json')

            # Save metadata.json
            metadata = {
                'item_id': item_id,
                'mockdata_key': collection_name,
                'trajectory_dir_name': traj_dir.name,
                'grounding_dir': str(traj_dir),
                'target_action': {
                    'action_id': detail_action['action_id'],
                    'from': detail_action['from'],
                    'to': detail_action['to'],
                    'screenshot_before': str(screenshot_path.relative_to(traj_dir))
                },
                'statistics': {
                    'total_questions_generated': len(questions),
                    'questions_passed_vlm': len(verified_qa_pairs),
                    'pass_rate': len(verified_qa_pairs) / len(questions) if questions else 0,
                    'by_type': count_by_type(verified_qa_pairs)
                },
                'generated_at': datetime.now().isoformat()
            }
            save_json(metadata, traj_output_dir / 'metadata.json')

            print(f"  ✓ Completed - saved to {traj_output_dir.name}")

            return {"success": True, "item_id": item_id}

        except Exception as e:
            print(f"  ✗ Error: {e}")
            return {"success": False, "item_id": item_id}
    
    def _generate_summary(
        self,
        output_dir: Path,
        caption_data: Dict,
        failed_items: List[str]
    ) -> Dict:
        """Generate summary statistics from saved trajectory files"""
        total_items_in_caption = sum(
            len(items) for items in caption_data.values()
            if isinstance(items, list)
        )

        breakdown_by_collection = {}
        breakdown_by_type = {
            'single_feature': {'generated': 0, 'passed': 0},
            'visual_description': {'generated': 0, 'passed': 0},
            'composite_answer': {'generated': 0, 'passed': 0}
        }

        total_generated = 0
        total_passed = 0
        total_qa_pairs = 0
        items_processed = 0

        # Read all trajectory folders
        for traj_dir in output_dir.iterdir():
            if not traj_dir.is_dir() or not traj_dir.name.startswith('macro_'):
                continue

            metadata_file = traj_dir / 'metadata.json'
            qa_pairs_file = traj_dir / 'qa_pairs.json'

            if not metadata_file.exists() or not qa_pairs_file.exists():
                continue

            metadata = load_json(metadata_file)
            qa_pairs = load_json(qa_pairs_file)

            items_processed += 1
            collection = metadata['mockdata_key']

            # Update collection breakdown
            if collection not in breakdown_by_collection:
                breakdown_by_collection[collection] = {
                    'items': 0,
                    'qa_pairs': 0
                }

            breakdown_by_collection[collection]['items'] += 1
            breakdown_by_collection[collection]['qa_pairs'] += len(qa_pairs)

            # Update type breakdown
            for qa in qa_pairs:
                qa_type = qa.get('type', 'unknown')
                if qa_type in breakdown_by_type:
                    breakdown_by_type[qa_type]['passed'] += 1

            # Update statistics
            stats = metadata.get('statistics', {})
            total_generated += stats.get('total_questions_generated', 0)
            total_passed += stats.get('questions_passed_vlm', 0)
            total_qa_pairs += len(qa_pairs)

            for qa_type in breakdown_by_type:
                breakdown_by_type[qa_type]['generated'] += stats.get('by_type', {}).get(qa_type, 0)

        # Calculate pass rates
        for qa_type in breakdown_by_type:
            gen = breakdown_by_type[qa_type]['generated']
            passed = breakdown_by_type[qa_type]['passed']
            breakdown_by_type[qa_type]['pass_rate'] = passed / gen if gen > 0 else 0

        return {
            "total_items_in_caption": total_items_in_caption,
            "items_processed": items_processed,
            "items_failed": len(failed_items),
            "total_qa_pairs": total_qa_pairs,
            "total_questions_generated": total_generated,
            "total_questions_passed_vlm": total_passed,
            "overall_pass_rate": total_passed / total_generated if total_generated > 0 else 0,
            "breakdown_by_collection": breakdown_by_collection,
            "breakdown_by_type": breakdown_by_type
        }

