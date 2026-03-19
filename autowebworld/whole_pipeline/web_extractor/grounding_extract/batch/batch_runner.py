"""Batch grounding extraction runner."""

import json
import asyncio
import shutil
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from tqdm.asyncio import tqdm

from core.extractor import BFSGroundingExtractor
from utils.logger import Logger
from utils.dev_server import start_dev_server


class BatchRunner:
    """Batch grounding extraction runner."""

    def __init__(
        self,
        input_folder: str,
        web_dir: str,
        base_url: Optional[str] = None,
        output_dir: str = "batch_outputs",
        max_workers: int = 3,
        scroll_step_size: int = 100,
        num_slider_steps: int = 3,
        headless: bool = False,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        verbose: bool = True,
        summary_path: Optional[str] = None,
        max_scroll_iterations: int = 50
    ):
        self.input_folder = Path(input_folder)
        self.web_dir = web_dir
        self.max_workers = max_workers
        self.scroll_step_size = scroll_step_size
        self.num_slider_steps = num_slider_steps
        self.headless = headless
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.verbose = verbose
        self.max_scroll_iterations = max_scroll_iterations

        if summary_path:
            self.summary_path = Path(summary_path)
        else:
            self.summary_path = self.input_folder.parent / "summary.json"

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = Logger("batch", verbose=verbose)
        if verbose:
            self.logger.info(f"Output directory: {self.output_dir}")

        # Auto-start Vue dev server if base_url not provided
        self.server = None
        if base_url is None:
            if verbose:
                self.logger.info(f"Starting Vue dev server for {web_dir}...")
            try:
                self.server, self.base_url = start_dev_server(web_dir, timeout=60)
                if verbose:
                    self.logger.info(f"✅ Dev server started: {self.base_url}")
            except Exception as e:
                self.logger.error(f"❌ Failed to start dev server: {e}")
                raise
        else:
            self.base_url = base_url

        self.total_files = 0
        self.completed_files = 0
        self.failed_files = 0
        self.results: List[Dict[str, Any]] = []

        # Error statistics: {error_key: count}
        self.error_stats: Dict[str, int] = {}

        # URL change warnings
        self.url_warnings: List[Dict[str, Any]] = []

        self.report_file = self.output_dir / "batch_report.json"
        self.error_stats_file = self.output_dir / "error_statistics.json"
        self.warnings_file = self.output_dir / "url_warnings.json"
        self.report_lock = threading.Lock()

        self._init_report()

    def _init_report(self):
        """Initialize report file."""
        initial_report = {
            "input_folder": str(self.input_folder),
            "output_folder": str(self.output_dir),
            "total_files": 0,
            "completed_files": 0,
            "failed_files": 0,
            "max_workers": self.max_workers,
            "results": [],
            "start_time": datetime.now().isoformat(),
            "last_update": datetime.now().isoformat()
        }

        with open(self.report_file, 'w', encoding='utf-8') as f:
            json.dump(initial_report, f, indent=2, ensure_ascii=False)

        if self.verbose:
            self.logger.info(f"Report initialized: {self.report_file}")

        # Initialize error statistics file
        initial_error_stats = {
            "total_errors": 0,
            "unique_errors": 0,
            "errors": [],
            "last_update": datetime.now().isoformat()
        }

        with open(self.error_stats_file, 'w', encoding='utf-8') as f:
            json.dump(initial_error_stats, f, indent=2, ensure_ascii=False)

        if self.verbose:
            self.logger.info(f"Error statistics initialized: {self.error_stats_file}")

    def _extract_error_key(self, error_message: str) -> str:
        """Extract error key from error message (remove URL part).

        Example:
            Input: "Page URL: http://localhost:36547/forward\n Grounding extraction failed: ..."
            Output: "Grounding extraction failed: ..."
        """
        # Split by newline and find the first line that doesn't start with "Page URL:"
        lines = error_message.split('\n')

        # Skip "Page URL:" line if present, but preserve original line formatting
        filtered_lines = []
        for line in lines:
            if not line.strip().startswith('Page URL:'):
                filtered_lines.append(line)

        # Join back and return
        error_key = '\n'.join(filtered_lines).strip()

        # If empty, return original
        return error_key if error_key else error_message

    def _update_report(self, new_result: Dict[str, Any]):
        """Update report file in real-time (thread-safe)."""
        with self.report_lock:
            with open(self.report_file, 'r', encoding='utf-8') as f:
                report = json.load(f)

            report["total_files"] = self.total_files
            report["completed_files"] = self.completed_files
            report["failed_files"] = self.failed_files
            report["last_update"] = datetime.now().isoformat()

            report["results"].append(new_result)

            with open(self.report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            # Update error statistics if this is a failed result
            if new_result.get("status") == "failed" and "error" in new_result:
                self._update_error_stats(new_result["error"])

    def _update_error_stats(self, error_message: str):
        """Update error statistics file (thread-safe)."""
        error_key = self._extract_error_key(error_message)

        # Update in-memory stats
        self.error_stats[error_key] = self.error_stats.get(error_key, 0) + 1

        # Sort by count (descending)
        sorted_errors = sorted(
            self.error_stats.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Calculate totals
        total_errors = sum(count for _, count in sorted_errors)
        unique_errors = len(sorted_errors)

        # Create error statistics data
        error_stats_data = {
            "total_errors": total_errors,
            "unique_errors": unique_errors,
            "errors": [
                {"error": error, "count": count}
                for error, count in sorted_errors
            ],
            "last_update": datetime.now().isoformat()
        }

        # Write to file
        with open(self.error_stats_file, 'w', encoding='utf-8') as f:
            json.dump(error_stats_data, f, indent=2, ensure_ascii=False)

    def find_json_files(self) -> List[Path]:
        """Find all trajectory JSON files recursively (exclude metadata files)."""
        all_json_files = list(self.input_folder.rglob("*.json"))

        # Filter out non-trajectory files
        exclude_names = {'entity_usage_history.json', 'result.json', 'batch_report.json', 'summary.json'}
        json_files = [
            f for f in all_json_files
            if f.name not in exclude_names
        ]

        if self.verbose:
            self.logger.info(f"Found {len(json_files)} trajectory files in {self.input_folder}")
            if len(all_json_files) > len(json_files):
                self.logger.info(f"  (Excluded {len(all_json_files) - len(json_files)} metadata files)")

        return json_files

    def find_summary_for_file(self, json_file: Path) -> Optional[Path]:
        try:
            relative_path = json_file.relative_to(self.input_folder)
            parts = relative_path.parts

            if len(parts) > 0 and parts[0] in ['home_initial', 'sub_initial']:
                subdir_summary = self.input_folder / parts[0] / 'summary.json'
                if subdir_summary.exists():
                    return subdir_summary

            if self.summary_path and self.summary_path.exists():
                return self.summary_path

        except ValueError:
            pass

        return None

    async def process_single_file(
        self,
        json_file: Path,
        semaphore: asyncio.Semaphore,
        file_index: int
    ) -> Dict[str, Any]:
        """Process a single JSON file."""
        async with semaphore:
            start_time = datetime.now()
            relative_path = json_file.relative_to(self.input_folder)

            file_output_dir = self.output_dir / json_file.stem
            file_output_dir.mkdir(parents=True, exist_ok=True)

            bfs_copy_path = file_output_dir / "bfs.json"
            shutil.copy2(json_file, bfs_copy_path)

            log_file = file_output_dir / "process.log"
            task_logger = Logger(
                name=f"task_{json_file.stem}",
                verbose=True,
                log_file=log_file,
                console_output=False
            )

            task_logger.info(f"{'='*80}")
            task_logger.info(f"Processing: {relative_path}")
            task_logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            task_logger.info(f"{'='*80}\n")

            try:
                actual_summary_path = self.find_summary_for_file(json_file)
                summary_path_str = str(actual_summary_path) if actual_summary_path else None

                if summary_path_str:
                    task_logger.info(f"Using summary file: {actual_summary_path}")
                else:
                    task_logger.warning("No summary.json found for this file")

                extractor = BFSGroundingExtractor(
                    web_dir=self.web_dir,
                    base_url=self.base_url,
                    output_dir=str(file_output_dir),
                    scroll_step_size=self.scroll_step_size,
                    num_slider_steps=self.num_slider_steps,
                    headless=self.headless,
                    viewport_width=self.viewport_width,
                    viewport_height=self.viewport_height,
                    verbose=True,
                    logger=task_logger,
                    summary_path=summary_path_str,
                    max_scroll_iterations=self.max_scroll_iterations
                )

                await extractor.start()

                try:
                    results = await extractor.extract_trajectory(str(json_file))

                    result_file = file_output_dir / "result.json"
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)

                    duration = (datetime.now() - start_time).total_seconds()
                    self.completed_files += 1

                    task_logger.info(f"\n{'='*80}")
                    task_logger.info(f"✅ Completed: {relative_path}")
                    task_logger.info(f"   Duration: {duration:.2f}s")
                    task_logger.info(f"   Output: {file_output_dir}")
                    task_logger.info(f"   Screenshots: {extractor.grounding.counter}")
                    task_logger.info(f"   End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    task_logger.info(f"{'='*80}")

                    # Collect URL warnings from this file
                    if extractor.url_warnings:
                        for warning in extractor.url_warnings:
                            warning["file"] = str(file_output_dir.resolve())
                            self.url_warnings.append(warning)

                    result = {
                        "file": str(relative_path),
                        "status": "success",
                        "duration": duration,
                        "output_dir": str(file_output_dir),
                        "screenshot_count": extractor.grounding.counter,
                        "completed_at": datetime.now().isoformat()
                    }

                    self._update_report(result)

                    return result

                finally:
                    await extractor.close()

            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                self.failed_files += 1

                task_logger.error(f"\n{'='*80}")
                task_logger.error(f"❌ Failed: {relative_path}")
                task_logger.error(f"   Error: {str(e)}")
                task_logger.error(f"   Duration: {duration:.2f}s")
                task_logger.error(f"   End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                task_logger.error(f"{'='*80}")

                self.logger.error(f"\n❌ Task failed: {relative_path}")
                self.logger.error(f"   Error: {str(e)}")
                self.logger.error(f"   Detailed log: {log_file}")

                error_file = file_output_dir / "error.txt"
                with open(error_file, 'w', encoding='utf-8') as f:
                    f.write(f"Error: {str(e)}\n")
                    f.write(f"File: {json_file}\n")
                    f.write(f"Time: {start_time}\n")

                result = {
                    "file": str(relative_path),
                    "status": "failed",
                    "error": str(e),
                    "duration": duration,
                    "output_dir": str(file_output_dir),
                    "completed_at": datetime.now().isoformat()
                }

                self._update_report(result)

                return result

    async def run(self):
        """Run batch extraction."""
        try:
            if self.verbose:
                self.logger.info("🚀 Batch Grounding Extraction")
                self.logger.info("=" * 80)
                self.logger.info(f"Input folder: {self.input_folder}")
                self.logger.info(f"Output folder: {self.output_dir}")
                self.logger.info(f"Web directory: {self.web_dir}")
                self.logger.info(f"Base URL: {self.base_url}")
                self.logger.info(f"Max workers: {self.max_workers}")
                self.logger.info(f"Scroll step: {self.scroll_step_size}px")
                self.logger.info("=" * 80)

            json_files = self.find_json_files()
            if not json_files:
                self.logger.warning("No JSON files found")
                return

            self.total_files = len(json_files)

            with self.report_lock:
                with open(self.report_file, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                report["total_files"] = self.total_files
                with open(self.report_file, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)

            if self.verbose:
                self.logger.info(f"\n📊 Starting to process {self.total_files} files...")
                self.logger.info(f"💾 The detailed log for each task will be saved as process.log in the corresponding output directory\n")

            semaphore = asyncio.Semaphore(self.max_workers)

            tasks = [
                self.process_single_file(json_file, semaphore, i + 1)
                for i, json_file in enumerate(json_files)
            ]

            start_time = datetime.now()
            self.results = await tqdm.gather(*tasks, desc="Processing progress", total=len(tasks))
            total_duration = (datetime.now() - start_time).total_seconds()

            self.print_summary(total_duration)
            self.save_report()

        finally:
            # Stop dev server if we started it
            if self.server:
                if self.verbose:
                    self.logger.info("Stopping dev server...")
                self.server.stop()
                if self.verbose:
                    self.logger.info("✅ Dev server stopped")

    def print_summary(self, total_duration: float):
        """Print summary statistics."""
        self.logger.always_info("\n" + "=" * 80)
        self.logger.always_info("📊 Batch Extraction Summary")
        self.logger.always_info("=" * 80)
        self.logger.always_info(f"Total files: {self.total_files}")
        self.logger.always_info(f"Completed: {self.completed_files}")
        self.logger.always_info(f"Failed: {self.failed_files}")
        self.logger.always_info(f"Total duration: {total_duration:.2f}s")
        self.logger.always_info(f"Average: {total_duration / self.total_files:.2f}s/file")
        self.logger.always_info("=" * 80)

        if self.failed_files > 0:
            self.logger.always_info("\n❌ Failed files:")
            for result in self.results:
                if result.get("status") == "failed":
                    output_path = result.get('output_dir', str(self.input_folder / result['file']))
                    self.logger.always_info(f"   - {output_path}: {result.get('error', 'Unknown error')}")

        # Print error statistics (simplified)
        if self.error_stats:
            # Sort by count (descending)
            sorted_errors = sorted(
                self.error_stats.items(),
                key=lambda x: x[1],
                reverse=True
            )

            total_errors = sum(count for _, count in sorted_errors)
            unique_errors = len(sorted_errors)

            self.logger.always_info(f"\n📈 Error statistics: {total_errors} errors, {unique_errors} unique types")
            self.logger.always_info(f"   Details: {self.error_stats_file}")

        # Print URL change warnings (simplified)
        if self.url_warnings:
            # Save warnings to file
            with open(self.warnings_file, 'w', encoding='utf-8') as f:
                json.dump(self.url_warnings, f, indent=2, ensure_ascii=False)

            self.logger.always_info(f"\n⚠️  URL change warnings: {len(self.url_warnings)} warnings")
            self.logger.always_info(f"   Details: {self.warnings_file}")

    def save_report(self):
        """Save final report."""
        with self.report_lock:
            with open(self.report_file, 'r', encoding='utf-8') as f:
                report = json.load(f)

            report["end_time"] = datetime.now().isoformat()
            report["status"] = "completed"

            with open(self.report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

        self.logger.always_info(f"\n📄 Final report saved: {self.report_file}")


async def run_batch(
    input_folder: str,
    web_dir: str,
    base_url: Optional[str] = None,
    output_dir: str = "batch_outputs",
    summary_path: Optional[str] = None,
    max_workers: int = 3,
    scroll_step: int = 100,
    num_slider_steps: int = 3,
    headless: bool = False,
    viewport_width: int = 1280,
    viewport_height: int = 720,
    verbose: bool = True,
    max_scroll_iterations: int = 50
):
    """
    Run batch grounding extraction.

    Args:
        input_folder: Input folder containing trajectory JSON files
        web_dir: Web project directory (Vue dev server will be started automatically)
        base_url: Base URL (if None, auto-start Vue dev server)
        output_dir: Output directory for results
        summary_path: Path to summary.json file containing slider metadata
        max_workers: Maximum concurrent workers
        scroll_step: Scroll step size in pixels
        num_slider_steps: Number of intermediate steps for slider drag
        headless: Run browser in headless mode
        viewport_width: Browser viewport width
        viewport_height: Browser viewport height
        verbose: Enable verbose logging
        max_scroll_iterations: Maximum scroll iterations
    """
    batch_runner = BatchRunner(
        input_folder=input_folder,
        web_dir=web_dir,
        base_url=base_url,
        output_dir=output_dir,
        summary_path=summary_path,
        max_workers=max_workers,
        scroll_step_size=scroll_step,
        num_slider_steps=num_slider_steps,
        headless=headless,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
        verbose=verbose,
        max_scroll_iterations=max_scroll_iterations
    )

    await batch_runner.run()

