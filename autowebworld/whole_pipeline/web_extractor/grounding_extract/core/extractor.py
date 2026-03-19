"""Main BFS grounding extractor."""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from playwright.async_api import async_playwright

from core.grounding import GroundingExtractor
from core.preprocessor import StepPreprocessor
from core.scroll_handler import ScrollHandler
from core.action_executor import ActionExecutor
from utils.logger import Logger
from utils.router import load_routes
from utils.dev_server import start_dev_server


class BFSGroundingExtractor:
    """BFS trajectory grounding extractor."""

    def __init__(
        self,
        web_dir: str,
        base_url: str,
        output_dir: str = "outputs",
        scroll_step_size: int = 100,
        num_slider_steps: int = 3,
        headless: bool = False,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        verbose: bool = True,
        logger: Optional[Logger] = None,
        summary_path: Optional[str] = None,
        max_scroll_iterations: int = 50
    ):
        self.web_dir = Path(web_dir)
        self.base_url = base_url.rstrip('/')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scroll_step_size = scroll_step_size
        self.num_slider_steps = num_slider_steps
        self.headless = headless
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.summary_path = summary_path
        self.max_scroll_iterations = max_scroll_iterations

        self.state_to_url = load_routes(self.web_dir)
        self.logger = logger if logger is not None else Logger("grounding", verbose=verbose)
        self.logger.info(f"Base URL: {self.base_url}")
        self.logger.info(f"Loaded {len(self.state_to_url)} routes")

        self.browser = None
        self.page = None
        self.grounding = GroundingExtractor(str(self.output_dir))
        self.preprocessor = None
        self.scroll_handler = None
        self.executor = None
        self.current_trajectory_file = None

        # Collect URL change warnings
        self.url_warnings: List[Dict[str, str]] = []

    async def start(self):
        """Start browser and initialize components."""
        playwright = await async_playwright().start()
        
        chromium_headless_shell_path = "/data2/wuyifan/MetaGPT/autoappworld/web_extractor/chromium_shell/chrome-linux/headless_shell"
        
        self.browser = await playwright.chromium.launch(
            headless=self.headless,
            executable_path=chromium_headless_shell_path if self.headless else None
        )

        context = await self.browser.new_context(
            viewport={'width': self.viewport_width, 'height': self.viewport_height},
            device_scale_factor=1
        )

        self.page = await context.new_page()

        self.preprocessor = StepPreprocessor(
            self.page,
            self.logger,
            summary_path=self.summary_path,
            current_bfs_file=self.current_trajectory_file,
            trajectory_data=None  # Will be set in extract_trajectory
        )
        self.scroll_handler = ScrollHandler(
            self.page,
            self.grounding,
            self.logger,
            self.scroll_step_size,
            self.viewport_width,
            self.viewport_height,
            preprocessor=self.preprocessor,
            max_scroll_iterations=self.max_scroll_iterations
        )
        self.executor = ActionExecutor(self.page, self.logger)

        # Initial navigation is now handled in extract_trajectory based on first action's from state

    async def close(self):
        """Close browser."""
        if self.browser:
            await self.browser.close()

    async def extract_trajectory(self, trajectory_file: str) -> List[Dict]:
        self.current_trajectory_file = Path(trajectory_file)

        with open(trajectory_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            trajectory = data
            trajectory_data = None
            self.logger.info(f"\nProcessing: {Path(trajectory_file).name} (old format)")
        elif isinstance(data, dict) and 'trajectory' in data:
            trajectory = data['trajectory']
            trajectory_data = data
            self.logger.info(f"\nProcessing: {Path(trajectory_file).name} (new format)")
            if 'mockdata_key' in data:
                self.logger.info(f"Mockdata key: {data['mockdata_key']}")
            if 'item_id' in data:
                self.logger.info(f"Item ID: {data['item_id']}")
            if 'trajectory_type' in data:
                self.logger.info(f"Trajectory type: {data['trajectory_type']}")
        else:
            raise ValueError(f"Unknown trajectory format in {trajectory_file}")

        self.logger.info(f"Actions: {len(trajectory)}\n")

        # Update preprocessor with current BFS file and trajectory data
        if self.preprocessor:
            self.preprocessor.current_bfs_file = self.current_trajectory_file
            self.preprocessor.trajectory_data = trajectory_data

        # Preprocess actions to merge FILTER + CLICK patterns
        trajectory = self.preprocessor.process_filter_actions(trajectory)
        self.logger.info(f"After preprocessing: {len(trajectory)} actions\n")

        # Get initial state from first action and navigate to it
        if trajectory:
            first_action = trajectory[0]
            initial_state = first_action['from']

            # Get initial URL from state_to_url mapping
            path = self.state_to_url.get(initial_state, '/')
            initial_url = f"{self.base_url}{path}" if path.startswith('/') else f"{self.base_url}/{path}"
            self.logger.info(f"Initial state: {initial_state}")
            self.logger.info(f"Initial URL: {initial_url}")

            await self.page.goto(initial_url, wait_until='domcontentloaded')
            await self.page.wait_for_timeout(300)

        results = []
        current_state = None

        for i, action in enumerate(trajectory):
            is_last_action = (i == len(trajectory) - 1)
            result = await self._extract_action(action, current_state, is_last_action)
            results.append(result)
            current_state = action['to']

        return results

    async def _extract_action(self, action: Dict, current_state: str, is_last_action: bool = False) -> Dict:
        """Extract grounding for a single action."""
        action_id = action['id']
        action_name = action.get('name', '')
        from_state = action['from']
        to_state = action['to']
        gui_steps = action.get('gui_procedure', [])
        params = action.get('params', {})

        self.logger.info(f"Action: {action_id} ({from_state} → {to_state}) [{action_name}]")

        if current_state != from_state:
            path = self.state_to_url.get(from_state, '/')
            url = f"{self.base_url}{path}" if path.startswith('/') else path
            self.logger.info(f"  Navigate to {from_state} -> {url}")
            try:
                await self.page.goto(url, wait_until='domcontentloaded', timeout=60000)
                await self.page.wait_for_timeout(300)
            except Exception as e:
                import traceback
                self.logger.error(f"  ❌ Navigation failed to {url}")
                self.logger.error(f"  Error: {e}")
                self.logger.error(f"  Traceback:\n{traceback.format_exc()}")
                raise
        else:
            self.logger.info(f"  Already at {from_state}")

        # Record URL before last action
        url_before = None
        if is_last_action:
            url_before = self.page.url

        processed_steps = self.preprocessor.process_click_drag_steps(gui_steps, params)
        processed_steps = await self.preprocessor.process_select_steps(
            processed_steps, action_name, params
        )
        processed_steps = self.preprocessor.process_datepicker_steps(processed_steps)

        steps_data = []
        current_step_index = 0
        i = 0
        while i < len(processed_steps):
            step = processed_steps[i]

            # Wait 100ms before processing each step (except the first one)
            if i > 0:
                await self.page.wait_for_timeout(100)

            # Detect hover+click combination and handle with fallback
            if (i + 1 < len(processed_steps) and
                step.get('op') == 'hover' and
                processed_steps[i + 1].get('op') == 'click'):

                hover_step = step
                click_step = processed_steps[i + 1]

                try:
                    # Try hover+click strategy
                    step_results = await self._extract_hover_click_with_fallback(
                        i, hover_step, click_step, current_step_index
                    )
                    steps_data.extend(step_results)
                    current_step_index += len(step_results)
                    i += 2  # Skip both hover and click steps
                    continue
                except Exception as e:
                    import traceback
                    self.logger.error(f"  hover+click failed: {e}")
                    self.logger.error(f"  Hover step: {hover_step}")
                    self.logger.error(f"  Click step: {click_step}")
                    self.logger.error(f"  Traceback:\n{traceback.format_exc()}")
                    raise

            # Normal step processing
            step_results = await self._extract_step(i, step, current_step_index)
            steps_data.extend(step_results)
            current_step_index += len(step_results)
            i += 1

        # Check URL change for last action
        if is_last_action:
            await self.page.wait_for_timeout(300)  # Wait for navigation to complete
            url_after = self.page.url

            if url_before == url_after:
                warning_msg = (
                    f"Last action did not change URL (still at: {url_after})\n"
                    f"  Action: {action_id} ({from_state} → {to_state})\n"
                    f"  This may indicate the action was ineffective or operates within the same page."
                )
                self.logger.warning(f"  ⚠ {warning_msg}")

                # Collect warning for final summary
                self.url_warnings.append({
                    "action_id": action_id,
                    "from_state": from_state,
                    "to_state": to_state,
                    "url": url_after,
                    "message": warning_msg
                })
            else:
                self.logger.info(f"  ✓ Last action changed URL: {url_before} → {url_after}")

        return {
            "action_id": action_id,
            "from": from_state,
            "to": to_state,
            "steps": steps_data
        }

    async def _extract_step(self, index: int, step: Dict, current_step_index: int) -> List[Dict]:
        """Extract grounding for a single step."""
        op = step.get('op', '')
        selector = step.get('selector', '')
        text = step.get('text', '')
        key = step.get('key', '')

        if op == 'scroll_to_child':
            return await self.scroll_handler.extract_scroll_to_child(step, current_step_index)

        if op == 'filter_and_click':
            return await self.scroll_handler.extract_filter_and_click(
                step,
                current_step_index,
                num_intermediate_steps=self.num_slider_steps
            )

        if op == 'slider':
            return await self.scroll_handler.extract_slider(
                step,
                current_step_index,
                num_intermediate_steps=self.num_slider_steps
            )

        if op == 'key_press' and not selector and self.scroll_handler.last_selector:
            selector = self.scroll_handler.last_selector
            self.logger.info(f"  Step {index+1}: {op} {key} (using last selector)")
        elif op == 'key_press':
            self.logger.info(f"  Step {index+1}: {op} {key}")
        else:
            self.logger.info(f"  Step {index+1}: {op} {selector}")

        steps_result = []

        scroll_actions = []
        if selector and op not in ['key_press']:
            selector = await self.preprocessor.normalize_selector(selector)
            refined_selector = await self.scroll_handler.find_element_in_same_parent(selector)
            if refined_selector != selector:
                self.logger.info(f"    Refined: {refined_selector}")
                selector = refined_selector

            scroll_actions = await self.scroll_handler.scroll_into_view(selector)
            self.scroll_handler.last_selector = selector

        if scroll_actions:
            for scroll_action in scroll_actions:
                scroll_container = scroll_action['container']
                scroll_selector = "html" if not scroll_container else scroll_container

                steps_result.append({
                    "index": current_step_index,
                    "operation": "scroll",
                    "selector": scroll_selector,
                    "text": "",
                    "key": "",
                    "grounding": scroll_action['grounding']
                })
                current_step_index += 1

        grounding = None
        if selector:
            image_before = await self.grounding.extract_before(self.page)

            # Try to extract annotation with page-level scroll retry if bbox is outside viewport
            annotation_data, page_scroll_steps = await self._extract_with_page_scroll_retry(
                selector, current_step_index
            )

            # Add page scroll steps to result
            if page_scroll_steps:
                steps_result.extend(page_scroll_steps)
                current_step_index += len(page_scroll_steps)

            if annotation_data.get('outside_viewport', False):
                bbox = annotation_data.get('bbox', [])
                raise Exception(
                    f"Element {selector} bbox {bbox} is outside viewport "
                    f"[{self.viewport_width}x{self.viewport_height}] and cannot be scrolled into view"
                )

            grounding = {
                **annotation_data,
                "image_before": image_before
            }

        step_to_perform = {**step, 'selector': selector}
        await self.executor.execute(step_to_perform)

        if op == 'key_press':
            await self.page.wait_for_timeout(300)
        else:
            await self.page.wait_for_timeout(200)

        if grounding:
            image_after = await self.grounding.extract_after(self.page)
            grounding["image_after"] = image_after
            self.grounding.counter += 1

            if op == 'type_text' and 'content' in grounding:
                self.logger.info(f"    Input: '{grounding['content']}'")

        steps_result.append({
            "index": current_step_index,
            "operation": op,
            "selector": selector,
            "text": text,
            "key": key,
            "grounding": grounding
        })

        return steps_result

    async def _extract_hover_click_with_fallback(
        self,
        step_index: int,
        hover_step: Dict,
        click_step: Dict,
        current_step_index: int
    ) -> List[Dict]:
        """Extract hover+click with fallback to click+click if hover+click fails.

        Strategy:
        1. Try hover+click first
        2. If hover+click fails, execute click+click

        Annotations saving:
        - If strategy 1 succeeds: save strategy 1's annotations
        - If strategy 1 fails: save only strategy 2's annotations (success or fail)
        """
        hover_selector = hover_step.get('selector', '')
        click_selector = click_step.get('selector', '')

        self.logger.info(f"  Step {step_index+1}-{step_index+2}: hover+click")
        self.logger.info(f"    Hover: {hover_selector}")
        self.logger.info(f"    Click: {click_selector}")

        # Strategy 1: Try hover+click (don't save grounding yet)
        hover_click_success = await self._try_hover_click_check_only(
            hover_selector, click_selector
        )

        if hover_click_success:
            self.logger.info("    ✓ hover+click succeeded")
            # Execute and save hover+click grounding
            return await self._execute_hover_click(
                hover_step, click_step, current_step_index
            )

        # Strategy 1 failed, use Strategy 2: click+click
        self.logger.warning("    ⚠ hover+click failed, option not visible")
        self.logger.info("    🔄 Falling back to click+click strategy")

        # Execute click+click and save its grounding (even if it fails)
        return await self._execute_click_click(
            hover_selector, click_selector, current_step_index
        )

    async def _try_hover_click_check_only(
        self,
        hover_selector: str,
        click_selector: str
    ) -> bool:
        """Check if hover+click would work (hover and check if option visible).

        Does NOT save grounding - just checks if strategy would work.
        """
        try:
            # Hover to trigger dropdown
            element = await self.page.query_selector(hover_selector)
            if not element:
                return False
            await element.hover()
            await self.page.wait_for_timeout(300)

            # Check if click target is now visible
            click_element = await self.page.query_selector(click_selector)
            if not click_element:
                return False

            is_visible = await click_element.is_visible()
            return is_visible
        except Exception:
            return False

    async def _execute_hover_click(
        self,
        hover_step: Dict,
        click_step: Dict,
        current_step_index: int
    ) -> List[Dict]:
        """Execute hover+click and save grounding."""
        results = []

        # Execute hover with grounding
        hover_results = await self._extract_step(0, hover_step, current_step_index)
        results.extend(hover_results)
        current_step_index += len(hover_results)

        await self.page.wait_for_timeout(300)

        # Execute click with grounding
        click_results = await self._extract_step(1, click_step, current_step_index)
        results.extend(click_results)

        return results

    async def _execute_click_click(
        self,
        trigger_selector: str,
        option_selector: str,
        current_step_index: int
    ) -> List[Dict]:
        """Execute click+click and save grounding (even if second click fails)."""
        results = []

        # First click: trigger dropdown (with grounding)
        click1_step = {'op': 'click', 'selector': trigger_selector}
        click1_results = await self._extract_step(0, click1_step, current_step_index)
        results.extend(click1_results)
        current_step_index += len(click1_results)

        await self.page.wait_for_timeout(300)

        # Check if option is visible
        element = await self.page.query_selector(option_selector)
        if not element:
            self.logger.error(f"    ✗ Option not found: {option_selector}")
            raise Exception(
                f"Both hover+click and click+click failed: "
                f"option not found {option_selector}"
            )

        is_visible = await element.is_visible()
        if not is_visible:
            self.logger.error(f"    ✗ Option not visible: {option_selector}")
            raise Exception(
                f"Both hover+click and click+click failed: "
                f"option not visible {option_selector}"
            )

        # Second click: select option (with grounding)
        click2_step = {'op': 'click', 'selector': option_selector}
        click2_results = await self._extract_step(1, click2_step, current_step_index)
        results.extend(click2_results)

        self.logger.info("    ✓ click+click succeeded")
        return results

    async def _extract_with_page_scroll_retry(
        self,
        selector: str,
        start_step_index: int,
        max_retries: int = 5
    ) -> Tuple[Dict[str, Any], List[Dict]]:
        """Extract annotation with page-level scroll retry if bbox is outside viewport.

        Note: Container scroll is already handled by scroll_into_view() in _extract_step().
        This method only handles page-level scroll for elements that are still outside viewport.
        Only records scroll steps that actually change the element's position.

        Args:
            selector: CSS selector
            start_step_index: Starting index for scroll steps
            max_retries: Maximum number of scroll retries

        Returns:
            Tuple of (annotation_data, scroll_steps_list)
        """
        scroll_steps = []
        last_bbox = None
        current_step_index = start_step_index

        for attempt in range(max_retries + 1):
            # Try to extract annotation without viewport check
            annotation_data = await self.grounding.extract_annotation(
                self.page,
                selector,
                self.viewport_width,
                self.viewport_height,
                check_viewport=False
            )

            # If element is within viewport, return immediately
            if not annotation_data.get('outside_viewport', False):
                if attempt > 0:
                    self.logger.info(f"    ✅ Element now in viewport after {attempt} scroll(s)")
                return annotation_data, scroll_steps

            bbox_before = annotation_data.get('bbox', [])

            # If this is the last attempt, raise exception
            if attempt >= max_retries:
                raise Exception(
                    f"Element bbox {bbox_before} is outside viewport "
                    f"[{self.viewport_width}x{self.viewport_height}] after {max_retries} scroll attempts"
                )

            # Check if bbox position hasn't changed (can't scroll anymore) - stop immediately
            if last_bbox and bbox_before == last_bbox:
                self.logger.warning(
                    f"    ⚠️  Bbox unchanged: {bbox_before}, scroll ineffective, stopping"
                )
                # Return what we have so far without raising exception
                return annotation_data, scroll_steps

            # Determine scroll direction and perform page scroll
            scroll_direction = self._determine_scroll_direction(bbox_before)
            self.logger.info(
                f"    📜 Bbox {bbox_before} outside viewport, "
                f"scrolling {scroll_direction} (attempt {attempt + 1}/{max_retries})..."
            )

            # Perform page scroll with effectiveness check
            # This only generates screenshots/annotations if scroll is effective
            scroll_grounding = await self.scroll_handler.perform_page_scroll_if_effective(
                scroll_direction,
                selector
            )

            if scroll_grounding is None:
                # Scroll had no effect, stop trying
                return annotation_data, scroll_steps

            # Scroll was effective, record it
            scroll_steps.append({
                "index": current_step_index,
                "operation": "scroll",
                "selector": "html",
                "text": "",
                "key": "",
                "grounding": scroll_grounding
            })
            current_step_index += 1
            last_bbox = annotation_data.get('bbox', [])

        # Should not reach here
        raise Exception(f"Failed to extract annotation for {selector}")

    def _determine_scroll_direction(self, bbox: List[int]) -> str:
        """Determine scroll direction based on bbox position.

        Args:
            bbox: [x1, y1, x2, y2]

        Returns:
            'up' or 'down'
        """
        _, y1, _, y2 = bbox

        # Check vertical position
        if y1 < 0:
            return 'up'
        elif y2 > self.viewport_height:
            return 'down'

        # Default to down
        return 'down'

    async def _extract_annotation_with_retry(self, selector: str, max_retries: int = 3) -> Dict:
        """Extract annotation with auto-scroll retry if element is outside viewport.

        Args:
            selector: CSS selector
            max_retries: Maximum number of scroll retries

        Returns:
            Annotation data dict

        Raises:
            Exception: If element cannot be brought into viewport after retries
        """
        last_bbox = None

        for attempt in range(max_retries):
            # Try to extract annotation without viewport check
            annotation_data = await self.grounding.extract_annotation(
                self.page,
                selector,
                self.viewport_width,
                self.viewport_height,
                check_viewport=False
            )

            # If element is within viewport, success!
            if not annotation_data.get('outside_viewport', False):
                return annotation_data

            current_bbox = annotation_data['bbox']

            # Check if bbox position changed from last attempt
            if last_bbox and current_bbox == last_bbox:
                # Bbox hasn't moved, can't scroll anymore
                self.logger.info(f"    Element bbox unchanged after scroll, cannot bring into viewport")
                raise Exception(
                    f"Element bbox {current_bbox} is outside viewport [{self.viewport_width}x{self.viewport_height}] "
                    f"and cannot be scrolled into view"
                )

            # Try to scroll the element into viewport
            self.logger.info(f"    Element outside viewport {current_bbox}, attempting scroll (attempt {attempt + 1}/{max_retries})...")

            try:
                # Scroll element into view
                element = await self.page.query_selector(selector)
                if element:
                    await element.scroll_into_view_if_needed()
                    await self.page.wait_for_timeout(300)
                    last_bbox = current_bbox
                else:
                    raise Exception(f"Element not found: {selector}")
            except Exception as e:
                self.logger.info(f"    Scroll failed: {e}")
                raise Exception(
                    f"Element bbox {current_bbox} is outside viewport [{self.viewport_width}x{self.viewport_height}] "
                    f"and scroll failed: {e}"
                )

        # Max retries reached
        raise Exception(
            f"Element bbox is outside viewport after {max_retries} scroll attempts"
        )


async def extract_single(
    trajectory_file: str,
    web_dir: str,
    base_url: Optional[str] = None,
    output_dir: str = None,
    scroll_step: int = 100,
    num_slider_steps: int = 3,
    headless: bool = False,
    viewport_width: int = 1280,
    viewport_height: int = 720,
    verbose: bool = True,
    max_scroll_iterations: int = 50,
    summary_path: Optional[str] = None
) -> List[Dict]:
    """
    Extract grounding from a single trajectory file.

    Args:
        trajectory_file: Path to trajectory JSON file
        web_dir: Web project directory (Vue dev server will be started automatically)
        base_url: Base URL (if None, auto-start Vue dev server)
        output_dir: Output directory (default: auto-generated)
        scroll_step: Scroll step size in pixels
        num_slider_steps: Number of intermediate steps for slider drag
        headless: Run browser in headless mode
        viewport_width: Browser viewport width
        viewport_height: Browser viewport height
        verbose: Enable verbose logging
        max_scroll_iterations: Maximum scroll iterations
        summary_path: Path to summary.json file containing slider metadata

    Returns:
        Extraction results
    """
    if output_dir is None:
        trajectory_path = Path(trajectory_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/{trajectory_path.stem}_{timestamp}"

    # Auto-start Vue dev server if base_url not provided
    server = None
    if base_url is None:
        logger = Logger("extractor", verbose=verbose)
        logger.info(f"Starting Vue dev server for {web_dir}...")
        try:
            server, base_url = start_dev_server(web_dir, timeout=60)
            logger.info(f"✅ Dev server started: {base_url}")
        except Exception as e:
            logger.error(f"❌ Failed to start dev server: {e}")
            raise

    try:
        extractor = BFSGroundingExtractor(
            web_dir=web_dir,
            base_url=base_url,
            output_dir=output_dir,
            scroll_step_size=scroll_step,
            num_slider_steps=num_slider_steps,
            headless=headless,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            verbose=verbose,
            max_scroll_iterations=max_scroll_iterations,
            summary_path=summary_path
        )

        await extractor.start()

        try:
            results = await extractor.extract_trajectory(trajectory_file)

            result_file = Path(output_dir) / "result.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            extractor.logger.info(f"\nResults saved to: {result_file}")
            extractor.logger.info(f"Screenshots: {extractor.grounding.counter}")

            return results

        finally:
            await extractor.close()

    finally:
        # Stop dev server if we started it
        if server:
            logger.info("Stopping dev server...")
            server.stop()
            logger.info("✅ Dev server stopped")

