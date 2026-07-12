#!/usr/bin/env python3
"""
Batch WebVoyager task runner.
Supports multiple websites, checkpoint-resume, and progress saving.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml

# Import accuracy calculation module
from calculate_accuracy import calculate_accuracy, save_metrics


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_web_files(data_dir: str, webs: List[str] = None) -> Dict[str, str]:
    """
    Collect the JSONL files for the specified websites.
    
    Args:
        data_dir: Directory containing WebVoyager JSONL files
        webs: List of website names to include; None means all
    
    Returns:
        Dict mapping web_name -> file_path
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    web_files = {}
    
    # Scan JSONL files matching the naming convention: {WebName}_only.jsonl
    for file_path in data_path.glob("*_only.jsonl"):
        # Extract web name: Apple_only.jsonl -> Apple
        web_name = file_path.stem.replace("_only", "")
        
        # Filter by requested websites if specified
        if webs and web_name not in webs:
            continue
        
        web_files[web_name] = str(file_path)
    
    return web_files


def count_tasks_in_file(file_path: str) -> int:
    """Count the number of tasks (lines) in a JSONL file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def load_progress(progress_file: str) -> Dict[str, Any]:
    """Load a progress file for checkpoint-resume."""
    if not os.path.exists(progress_file):
        return {"completed": set(), "failed": set(), "results": []}
    
    with open(progress_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Convert lists back to sets
    data["completed"] = set(data.get("completed", []))
    data["failed"] = set(data.get("failed", []))
    
    return data


def save_progress(progress_file: str, progress: Dict[str, Any]):
    """Persist the current progress to disk."""
    os.makedirs(os.path.dirname(progress_file), exist_ok=True)
    
    # Convert sets to lists for JSON serialization
    save_data = {
        "completed": list(progress["completed"]),
        "failed": list(progress["failed"]),
        "results": progress["results"],
        "last_updated": datetime.now().isoformat(),
    }
    
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)


def run_single_task(
    task_file: str,
    task_index: int,
    output_file: str,
    config: Dict[str, Any],
    task_artifact_dir: str = None,
) -> Dict[str, Any]:
    """Invoke the single-task runner as a subprocess and return the result."""
    import subprocess

    # Build command
    cmd = [
        sys.executable,
        "scripts/run_single_web_task.py",
        "--task-file", task_file,
        "--task-index", str(task_index),
        "--output-file", output_file,
        "--agent-type", config.get("agent_type", "gui_agent"),
        "--model", config["model"],
        "--max-steps", str(config["max_steps"]),
        "--text-steps", str(config["text_steps"]),
        "--image-steps", str(config["image_steps"]),
        "--width", str(config["width"]),
        "--height", str(config["height"]),
        "--wait-seconds", str(config["wait_seconds"]),
        "--timeout", str(config["timeout"]),
    ]

    if config.get("headless"):
        cmd.append("--headless")

    if config.get("capture"):
        cmd.append("--capture")

    # Pass task-specific artifact directory
    if task_artifact_dir:
        cmd.extend(["--artifact-dir", task_artifact_dir])
    elif config.get("artifact_dir"):
        cmd.extend(["--artifact-dir", config["artifact_dir"]])

    if config.get("executable_path"):
        cmd.extend(["--executable-path", config["executable_path"]])

    # LLM evaluation arguments
    if config.get("evaluate"):
        cmd.append("--evaluate")
    if config.get("eval_model"):
        cmd.extend(["--eval-model", config["eval_model"]])
    if config.get("eval_max_screenshots"):
        cmd.extend(["--eval-max-screenshots", str(config["eval_max_screenshots"])])
    if config.get("eval_concurrent_count"):
        cmd.extend(["--eval-concurrent-count", str(config["eval_concurrent_count"])])

    # Run subprocess
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(cmd, check=False, capture_output=False)
        
        # Read result from output file
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return {
                "finish": False,
                "error": "Output file not created",
                "return_code": result.returncode,
            }
    except Exception as e:
        return {
            "finish": False,
            "error": str(e),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch run WebVoyager tasks")
    parser.add_argument("--config", default="scripts/batch_config.yaml", help="Config file path")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be run")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f"Loaded config from: {args.config}")

    # Discover JSONL files for each website
    web_files = get_web_files(config["data_dir"], config.get("webs"))
    print(f"\nFound {len(web_files)} web(s) to process:")
    for web_name in sorted(web_files.keys()):
        task_count = count_tasks_in_file(web_files[web_name])
        print(f"  - {web_name}: {task_count} tasks")

    # Build timestamped output directory structure:
    # output/webvoyager_batch/{agent_type}/{model_name}/{web_name}/{web_name}_{timestamp}/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = config.get("output_base_dir", "output/webvoyager_batch")

    agent_type = config.get("agent_type", "gui_agent")
    model_name = config.get("model", "unknown_model")

    agent_type_dir = os.path.join(output_base_dir, agent_type)
    model_dir = os.path.join(agent_type_dir, model_name)

    print(f"\nAgent type directory: {agent_type_dir}")
    print(f"Model directory: {model_dir}")
    print(f"Agent type: {agent_type}")
    print(f"Inference model: {model_name}")
    print(f"Timestamp: {timestamp}")

    # Prepare per-website task lists and output directories
    web_tasks = {}

    for web_name, file_path in sorted(web_files.items()):
        web_dir = os.path.join(model_dir, web_name)
        web_run_dir = os.path.join(web_dir, f"{web_name}_{timestamp}")
        web_progress_file = os.path.join(web_run_dir, "progress.json")

        # Load or initialise progress for this website
        web_progress = None
        if config.get("resume"):
            web_progress = load_progress(web_progress_file)
            print(f"\n{web_name} - Progress loaded:")
            print(f"  - Completed: {len(web_progress['completed'])} tasks")
            print(f"  - Failed: {len(web_progress['failed'])} tasks")
        else:
            web_progress = {"completed": set(), "failed": set(), "results": []}

        # Build the list of tasks to run for this website
        tasks = []
        task_count = count_tasks_in_file(file_path)
        max_tasks = config.get("max_tasks_per_web")
        skip_tasks = config.get("skip_tasks", {}).get(web_name, [])

        for task_idx in range(task_count):
            if task_idx in skip_tasks:
                continue

            if max_tasks is not None and task_idx >= max_tasks:
                break

            task_id = f"{web_name}--{task_idx}"

            # Skip already-completed tasks when resuming
            if config.get("resume") and task_id in web_progress["completed"]:
                continue

            tasks.append({
                "web_name": web_name,
                "file_path": file_path,
                "task_index": task_idx,
                "task_id": task_id,
            })

        web_tasks[web_name] = {
            "tasks": tasks,
            "output_dir": web_run_dir,
            "progress_file": web_progress_file,
            "progress": web_progress,
        }

        print(f"\n{web_name}:")
        print(f"  - Output directory: {web_run_dir}")
        print(f"  - Tasks to run: {len(tasks)}")

    # Total task count across all websites
    total_tasks = sum(len(web_data["tasks"]) for web_data in web_tasks.values())
    print(f"\nTotal tasks to run across all webs: {total_tasks}")

    if args.dry_run:
        print("\n[DRY RUN] Would run the following tasks:")
        count = 0
        for web_name, web_data in web_tasks.items():
            for task in web_data["tasks"][:5]:  # Show first 5 per website
                print(f"  - {task['task_id']}")
                count += 1
                if count >= 10:
                    break
            if count >= 10:
                break
        if total_tasks > 10:
            print(f"  ... and {total_tasks - 10} more")
        return 0

    # Execute tasks for all websites
    overall_finish_count = 0
    overall_failed_count = 0
    task_counter = 0

    for web_name, web_data in web_tasks.items():
        tasks = web_data["tasks"]
        output_dir = web_data["output_dir"]
        progress_file = web_data["progress_file"]
        progress = web_data["progress"]

        if not tasks:
            print(f"\n{web_name}: No tasks to run (all completed or skipped)")
            continue

        print(f"\n{'='*80}")
        print(f"Starting Web: {web_name}")
        print(f"Output directory: {output_dir}")
        print(f"Tasks: {len(tasks)}")
        print(f"{'='*80}")

        os.makedirs(output_dir, exist_ok=True)

        finish_count = 0
        failed_count = 0

        for i, task in enumerate(tasks, 1):
            task_counter += 1
            print(f"\n{'='*80}")
            print(f"Overall Progress: {task_counter}/{total_tasks} ({task_counter*100//total_tasks}%)")
            print(f"{web_name} Progress: {i}/{len(tasks)}")
            print(f"Task: {task['task_id']}")
            print(f"{'='*80}")

            # Output directory structure:
            # {web_run_dir}/{task_id}/result.json
            task_dir = os.path.join(output_dir, task["task_id"])
            output_file = os.path.join(task_dir, "result.json")

            os.makedirs(task_dir, exist_ok=True)

            # Run the task; GuiAgentEnvironment will create task_{timestamp}/ inside task_dir
            result = run_single_task(
                task_file=task["file_path"],
                task_index=task["task_index"],
                output_file=output_file,
                config=config,
                task_artifact_dir=task_dir,
            )

            # Update progress
            if result.get("finish"):
                progress["completed"].add(task["task_id"])
                finish_count += 1
                overall_finish_count += 1
            else:
                progress["failed"].add(task["task_id"])
                failed_count += 1
                overall_failed_count += 1

            progress["results"].append(result)

            if config.get("resume"):
                save_progress(progress_file, progress)

            # Stop early if configured to abort on failure
            if not result.get("finish") and not config.get("continue_on_error"):
                print(f"\n[ERROR] Task failed and continue_on_error=False. Stopping.")
                break

        # Save per-website summary
        summary_file = os.path.join(output_dir, "summary.json")
        summary = {
            "web_name": web_name,
            "total": len(tasks),
            "completed": len(progress["completed"]),
            "failed": len(progress["failed"]),
            "finish_rate": len(progress["completed"]) / len(tasks) if len(tasks) > 0 else 0,
            "results": progress["results"],
            "timestamp": datetime.now().isoformat(),
        }

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*80}")
        print(f"{web_name} completed!")
        print(f"{'='*80}")
        print(f"Total: {len(tasks)}")
        print(f"Finish: {len(progress['completed'])}")
        print(f"Failed: {len(progress['failed'])}")
        print(f"Finish Rate: {summary['finish_rate']*100:.1f}%")
        print(f"\nSummary saved to: {summary_file}")

        # Compute accuracy metrics for this website if evaluation was enabled
        if config.get("evaluate", False):
            try:
                print(f"\n{'='*80}")
                print(f"Calculating accuracy for {web_name}...")
                print(f"{'='*80}")
                my_metrics = calculate_accuracy(str(output_dir), silent=False)
                my_metrics_path = Path(output_dir) / "my_metrics.json"
                save_metrics(my_metrics, my_metrics_path, silent=False)
                print(f"\nAccuracy calculation complete: {my_metrics['accuracy']:.2f}%")
            except Exception as e:
                print(f"Warning: Failed to calculate accuracy: {e}")
                import traceback
                traceback.print_exc()

    # Print overall statistics
    print(f"\n{'='*80}")
    print(f"All web batch runs complete!")
    print(f"{'='*80}")
    print(f"Total tasks: {total_tasks}")
    print(f"Finished: {overall_finish_count}")
    print(f"Failed: {overall_failed_count}")
    if total_tasks > 0:
        print(f"Overall finish rate: {overall_finish_count*100/total_tasks:.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
