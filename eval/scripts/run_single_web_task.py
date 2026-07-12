#!/usr/bin/env python3
"""
Single WebVoyager task runner script.
Reads tasks from a JSONL file and runs them one at a time.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, Optional
from io import StringIO

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import warnings
from typing import Type, cast

import yaml

from autoappworld.agent.gui_agent import GuiAgent
from autoappworld.evaluation import (
    GuiAgentEnvironment,
    GuiAgentEnvironmentConfig,
    GuiAgentRunner,
)
from autoappworld.evaluation.evaluators import GPTEvaluator
from easyagent.config.base import ModelConfig
from easyagent.memory import StepWindowMemory
from easyagent.model.litellm_model import LiteLLMModel


class TeeLogger:
    """Logger that writes to both the terminal and a StringIO buffer simultaneously."""
    def __init__(self, terminal, string_io):
        self.terminal = terminal
        self.string_io = string_io

    def write(self, message):
        self.terminal.write(message)
        self.string_io.write(message)

    def flush(self):
        self.terminal.flush()
        self.string_io.flush()


def load_task_from_jsonl(file_path: str, task_index: int) -> Dict[str, Any]:
    """Load a task at the given index from a JSONL file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Task file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == task_index:
                return json.loads(line.strip())

    raise IndexError(f"Task index {task_index} not found in {file_path}")


def _truncate_base64_in_logs(log_content: str, max_length: int = 100) -> str:
    """
    Truncate base64 image data in log content to keep log files small.

    Mirrors the _truncate_base64_in_messages method in gpt_evaluator.py.

    Args:
        log_content: Raw log content string
        max_length: Maximum number of base64 characters to retain (default: 100)

    Returns:
        Log content with base64 data truncated
    """
    import re

    # Regex to match base64 image data URLs
    # Pattern: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...
    pattern = r'(data:image/[^;]+;base64,)([A-Za-z0-9+/=]{100,})'

    def replace_base64(match):
        prefix = match.group(1)       # e.g., data:image/png;base64,
        base64_data = match.group(2)  # the actual base64 payload

        # Truncate the base64 payload
        truncated = base64_data[:max_length] + f"...[truncated {len(base64_data) - max_length} chars]"
        return prefix + truncated

    # Replace all matching base64 data
    truncated_content = re.sub(pattern, replace_base64, log_content)

    return truncated_content


def run_task(args: argparse.Namespace) -> Dict[str, Any]:
    """Run a single task and return the result dict."""
    # Capture all stdout/stderr output via StringIO so it can be saved to a log file
    log_buffer_stdout = StringIO()
    log_buffer_stderr = StringIO()
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Tee output to both terminal and buffer
    sys.stdout = TeeLogger(original_stdout, log_buffer_stdout)
    sys.stderr = TeeLogger(original_stderr, log_buffer_stderr)

    # Load the task
    task_data = load_task_from_jsonl(args.task_file, args.task_index)

    task_id = task_data.get("id", f"task_{args.task_index}")
    task_query = task_data.get("ques", "")
    task_url = task_data.get("web", "")
    web_name = task_data.get("web_name", "Unknown")

    print(f"\n{'='*80}")
    print(f"Running Task: {task_id}")
    print(f"Web: {web_name}")
    print(f"URL: {task_url}")
    print(f"Query: {task_query}")
    print(f"{'='*80}\n")
    
    # Configure model
    config = ModelConfig.load()
    model = LiteLLMModel(**config.get_model(args.model))
    memory = StepWindowMemory(
        text_steps=args.text_steps,
        image_steps=args.image_steps,
    )

    # Use GuiAgent uniformly (agent_type selects the system prompt, template, and parser)
    agent_type = getattr(args, "agent_type", "gui_agent")
    agent = GuiAgent(model=model, memory=memory, agent_type=agent_type)
    
    # Configure environment
    # Pass artifact_dir directly; GuiAgentEnvironment will create the task_{timestamp}/ sub-directory
    artifact_dir = args.artifact_dir if args.artifact_dir else None

    env_config = GuiAgentEnvironmentConfig(
        web_dir=None,  # WebVoyager uses direct URLs, no local web dir needed
        max_steps=args.max_steps,
        headless=args.headless,
        viewport=(args.width, args.height),
        capture=args.capture,
        annotate=True,
        artifact_dir=artifact_dir,
        wait_seconds=args.wait_seconds,
        executable_path=args.executable_path,
    )
    
    env = GuiAgentEnvironment("gui_agent", config=env_config)
    runner = GuiAgentRunner(env)
    env.register_runner(runner)
    
    ctx = {
        "agent": agent,
        "task": task_query,
        "url": task_url,
        "manager": None,
        "script": None,
        "start_timeout": args.timeout,
        "model_name": args.model,
        "agent_type": agent_type,
        "use_timestamp_dir": False,  # Use artifact_dir directly, no auto timestamp sub-dir
    }
    
    # Run the task
    start_time = datetime.now()
    try:
        result = env.run(sample={"task": task_query}, runner_id="gui_agent", ctx=ctx)
        stop_reason = result.meta.get("stop_reason", "unknown")
        finish_message = result.meta.get("finish_message", "")
        # Retrieve the actual artifact run directory (task_{timestamp}/)
        actual_artifact_dir = result.meta.get("artifact_run_dir", artifact_dir)
        # finish is True only when the agent explicitly called finish()
        # This does NOT assess answer correctness; that requires an external evaluator
        finish = stop_reason == "finish_action"
        error = None
    except Exception as e:
        # Catch any runtime exception and set stop_reason = "error"
        # Possible sources: Playwright failures, LLM API errors, network timeouts, etc.
        stop_reason = "error"
        finish_message = str(e)
        finish = False
        error = str(e)
        actual_artifact_dir = artifact_dir
        print(f"[ERROR] Task failed: {e}")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # LLM-based evaluation (if enabled)
    # Evaluation conditions:
    #   1. args.evaluate is True
    #   2. stop_reason is "finish_action" (agent completed) or "max_steps" (step limit reached)
    #   "error" cases are not evaluated
    gpt_score = None
    gpt_reasoning = None
    all_eval_results = []  # Stores results from all concurrent evaluation runs

    if args.evaluate and stop_reason in ["finish_action", "max_steps"]:
        eval_concurrent_count = args.eval_concurrent_count

        print(f"\n{'='*80}")
        if eval_concurrent_count > 1:
            print(f"Running GPT Evaluation ({eval_concurrent_count} concurrent evaluations)...")
        else:
            print(f"Running GPT Evaluation (single evaluation)...")
        print(f"{'='*80}")
        print(f"Screenshot directory: {actual_artifact_dir}")

        try:
            # Determine answer text based on stop reason
            if stop_reason == "finish_action":
                answer_text = finish_message or "<finish>"
            else:  # max_steps
                answer_text = "<finish>"

            def run_single_evaluation(eval_id: int) -> Dict[str, Any]:
                """Run a single evaluation pass in an isolated thread with its own event loop."""
                import asyncio

                print(f"  Starting evaluation #{eval_id}...")

                # Create a fresh event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                try:
                    evaluator = GPTEvaluator(
                        model=args.eval_model,
                        max_screenshots=args.eval_max_screenshots,
                    )

                    score, reasoning, request_info = loop.run_until_complete(
                        evaluator.evaluate_async(
                            task=task_query,
                            answer=answer_text,
                            screenshot_dir=actual_artifact_dir,
                            return_request=True,
                        )
                    )

                    print(f"  Evaluation #{eval_id} complete: score={score}")

                    return {
                        "eval_id": eval_id,
                        "score": score,
                        "reasoning": reasoning,
                        "request_info": request_info,
                        "is_success": score == 1.0,  # Score is binary: 0.0 or 1.0
                    }
                finally:
                    loop.close()

            # Run evaluations (concurrent or single)
            all_eval_results = []

            if eval_concurrent_count > 1:
                # Concurrent evaluation: submit all runs in parallel
                with ThreadPoolExecutor(max_workers=eval_concurrent_count) as executor:
                    futures = {executor.submit(run_single_evaluation, i): i for i in range(1, eval_concurrent_count + 1)}

                    for future in as_completed(futures):
                        result = future.result()
                        all_eval_results.append(result)

                # Sort by eval_id for consistent ordering
                all_eval_results.sort(key=lambda x: x["eval_id"])
            else:
                # Single evaluation run
                result = run_single_evaluation(1)
                all_eval_results = [result]

            # Select final result:
            # - If any evaluation succeeded, pick the first success
            # - Otherwise, fall back to the first result
            if eval_concurrent_count > 1:
                success_results = [r for r in all_eval_results if r["is_success"]]

                if success_results:
                    final_result = success_results[0]
                    print(f"\nFound SUCCESS result (Evaluation #{final_result['eval_id']})")
                else:
                    final_result = all_eval_results[0]
                    print(f"\nAll evaluations failed, using first result (Evaluation #1)")
            else:
                final_result = all_eval_results[0]
                print(f"\nEvaluation complete")

            gpt_score = final_result["score"]
            gpt_reasoning = final_result["reasoning"]

            print(f"\n{'='*80}")
            print(f"Final Evaluation Result:")
            print(f"  Score: {gpt_score}")
            print(f"  Reasoning: {gpt_reasoning[:200]}...")
            print(f"{'='*80}")

            # Save evaluation results to score.json alongside the task artifact directory
            score_file = os.path.join(artifact_dir, "score.json")

            all_evaluations_detail = []
            for eval_result in all_eval_results:
                all_evaluations_detail.append({
                    "eval_id": eval_result["eval_id"],
                    "score": eval_result["score"],
                    "is_success": eval_result["is_success"],
                    "reasoning": eval_result["reasoning"],
                    "request_info": eval_result["request_info"],
                })

            save_data = {
                # Final selected result
                "score": gpt_score,
                "gpt_response_text": gpt_reasoning,

                # Metadata
                "task": task_query,
                "answer": answer_text,
                "stop_reason": stop_reason,
                "evaluated_at": datetime.now().isoformat(),
                "eval_model": args.eval_model,
                "eval_concurrent_count": eval_concurrent_count,
            }

            if eval_concurrent_count > 1:
                success_results = [r for r in all_eval_results if r["is_success"]]
                save_data.update({
                    "final_eval_id": final_result["eval_id"],
                    "selection_reason": "first_success" if success_results else "all_failed_use_first",
                    "all_evaluations": all_evaluations_detail,
                    "total_evaluations": len(all_eval_results),
                    "success_count": len(success_results),
                })
            else:
                save_data.update({
                    "all_evaluations": all_evaluations_detail,
                    "total_evaluations": 1,
                })

            with open(score_file, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)

            print(f"Evaluation saved to: {score_file}")
            if eval_concurrent_count > 1:
                print(f"   - Total evaluations: {len(all_eval_results)}")
                print(f"   - Success count: {len(success_results)}")
                print(f"   - Final result from: Evaluation #{final_result['eval_id']}")

        except Exception as e:
            print(f"GPT Evaluation failed: {e}")
            import traceback
            traceback.print_exc()

    # Build result dict
    result_data = {
        "task_id": task_id,
        "web_name": web_name,
        "url": task_url,
        "query": task_query,
        "finish": finish,
        "stop_reason": stop_reason,
        "finish_message": finish_message,
        "error": error,
        "duration_seconds": duration,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "model": args.model,
        "max_steps": args.max_steps,
        "gpt_score": gpt_score,
        "gpt_reasoning": gpt_reasoning,
    }

    print(f"\n{'='*80}")
    print(f"Task Completed: {task_id}")
    print(f"Finish: {finish}")
    print(f"Stop Reason: {stop_reason}")
    print(f"Duration: {duration:.2f}s")
    if finish_message:
        print(f"Message: {finish_message}")
    print(f"{'='*80}\n")

    # Restore original stdout/stderr
    sys.stdout = original_stdout
    sys.stderr = original_stderr

    # Write captured log to task.log file (with base64 data truncated)
    if actual_artifact_dir:
        log_file = os.path.join(actual_artifact_dir, "task.log")
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                stdout_content = _truncate_base64_in_logs(log_buffer_stdout.getvalue())
                f.write(stdout_content)
                stderr_content = log_buffer_stderr.getvalue()
                if stderr_content:
                    f.write("\n" + "="*80 + "\n")
                    f.write("STDERR OUTPUT:\n")
                    f.write("="*80 + "\n")
                    stderr_content = _truncate_base64_in_logs(stderr_content)
                    f.write(stderr_content)
            print(f"Log saved to: {log_file}")
        except Exception as e:
            print(f"Failed to save log file: {e}")

    return result_data


def main() -> int:
    # Suppress irrelevant Pydantic serialization warnings
    try:
        from pydantic import PydanticSerializationUnexpectedValue
        warnings.filterwarnings("ignore", category=cast(Type[Warning], PydanticSerializationUnexpectedValue))
    except Exception:
        warnings.filterwarnings("ignore", message=r"Pydantic serializer warnings:.*", category=UserWarning)

    parser = argparse.ArgumentParser(description="Run a single WebVoyager task from JSONL file")
    parser.add_argument("--task-file", required=True, help="Path to JSONL task file")
    parser.add_argument("--task-index", type=int, required=True, help="Task index in JSONL file (0-based)")
    parser.add_argument("--output-file", help="Path to save result JSON")
    parser.add_argument("--agent-type", default="gui_agent", help="Agent type: gui_agent, ui_tars, tongui, ui_venus, open_cua")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name defined in model_config.yaml")
    parser.add_argument("--max-steps", type=int, default=60, help="Max action steps")
    parser.add_argument("--text-steps", type=int, default=10, help="Number of text history steps to keep")
    parser.add_argument("--image-steps", type=int, default=0, help="Number of image history steps to keep")
    parser.add_argument("--headless", action="store_true", default=True, help="Run browser in headless mode")
    parser.add_argument("--width", type=int, default=1280, help="Viewport width")
    parser.add_argument("--height", type=int, default=720, help="Viewport height")
    parser.add_argument("--artifact-dir", help="Directory to store screenshots and trajectory")
    parser.add_argument("--capture", action="store_true", default=False, help="Enable screenshot capture per action")
    parser.add_argument("--wait-seconds", type=float, default=0.1, help="Seconds to wait after a 'wait' action")
    parser.add_argument("--timeout", type=int, default=60, help="Web server startup timeout (seconds)")
    parser.add_argument("--executable-path", help="Path to Chrome/Chromium executable")

    # LLM-based evaluation arguments
    parser.add_argument("--evaluate", action="store_true", help="Enable LLM-based evaluation after task completion")
    parser.add_argument("--eval-model", default="gpt-4o", help="Model for evaluation (default: gpt-4o)")
    parser.add_argument("--eval-max-screenshots", type=int, default=5, help="Max screenshots to include in evaluation (default: 5)")
    parser.add_argument("--eval-concurrent-count", type=int, default=1, help="Number of concurrent evaluation runs (default: 1)")

    args = parser.parse_args()

    # Run the task
    result = run_task(args)

    # Save result to file if requested
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Result saved to: {args.output_file}")

    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
