"""Step preprocessing for GUI procedures."""

import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional


class StepPreprocessor:
    """Preprocess GUI steps before extraction."""

    def __init__(
        self,
        page,
        logger,
        summary_path: Optional[str] = None,
        current_bfs_file: Optional[str] = None,
        trajectory_data: Optional[Dict] = None
    ):
        self.page = page
        self.logger = logger
        self.summary_path = Path(summary_path) if summary_path else None
        self.current_bfs_file = Path(current_bfs_file) if current_bfs_file else None
        self.trajectory_data = trajectory_data
        self._slider_metadata_cache = None

    def process_filter_actions(self, actions: List[dict]) -> List[dict]:
        """Process actions: merge FILTER+CLICK patterns and mark slider actions.

        Handles:
        1. Merge FILTER + CLICK .row-filtered patterns into filter_and_click
        2. Mark slider actions (click+drag with 'slider' in selector) to prevent
           them from being converted to scroll_to_child later
        """
        processed_actions = []
        i = 0

        while i < len(actions):
            current_action = actions[i]

            # Check if this is a FILTER action followed by CLICK .row-filtered
            if i + 1 < len(actions):
                next_action = actions[i + 1]

                # Check if current action is a FILTER action
                is_filter_action = self._is_filter_slider_action(current_action)

                # Check if next action is CLICK .row-filtered
                is_click_filtered = self._is_click_filtered_action(next_action)

                if is_filter_action and is_click_filtered:
                    # Merge the two actions
                    merged_action = self._merge_filter_and_click(current_action, next_action)
                    processed_actions.append(merged_action)
                    self.logger.info(
                        f"🔄 Merged FILTER + CLICK: {current_action['id']} + {next_action['id']}"
                    )
                    i += 2
                    continue

            # Check if this is a slider action (click+drag with 'slider' in selector)
            # Mark it to prevent conversion to scroll_to_child
            if self._is_slider_action(current_action):
                marked_action = self._mark_as_slider_action(current_action)
                processed_actions.append(marked_action)
                self.logger.info(f"🎚️ Marked slider action: {current_action['id']}")
                i += 1
                continue

            # No special handling, keep original action
            processed_actions.append(current_action)
            i += 1

        return processed_actions

    def _is_slider_action(self, action: dict) -> bool:
        """Check if action contains click+drag on a slider element."""
        gui_procedure = action.get('gui_procedure', [])

        for i in range(len(gui_procedure) - 1):
            current_step = gui_procedure[i]
            next_step = gui_procedure[i + 1]

            if (current_step.get('op') == 'click' and
                next_step.get('op') == 'drag' and
                current_step.get('selector') == next_step.get('selector')):

                selector = current_step.get('selector', '')
                if 'slider' in selector.lower():
                    return True

        return False

    def _mark_as_slider_action(self, action: dict) -> dict:
        """Mark action's gui_procedure steps as slider operations."""
        marked_action = action.copy()
        marked_gui_procedure = []

        gui_procedure = action.get('gui_procedure', [])
        i = 0

        while i < len(gui_procedure):
            current_step = gui_procedure[i]

            # Check for click+drag on slider
            if (i + 1 < len(gui_procedure) and
                current_step.get('op') == 'click' and
                gui_procedure[i + 1].get('op') == 'drag' and
                'slider' in current_step.get('selector', '').lower()):

                drag_step = gui_procedure[i + 1]
                # Convert to slider operation
                marked_gui_procedure.append({
                    'op': 'slider',
                    'selector': current_step.get('selector'),
                    'to_value': drag_step.get('to_value', 'right')
                })
                i += 2
                continue

            marked_gui_procedure.append(current_step)
            i += 1

        marked_action['gui_procedure'] = marked_gui_procedure
        return marked_action

    def _is_filter_slider_action(self, action: dict) -> bool:
        """Check if action is a FILTER slider action."""
        action_id = action.get('id', '')
        gui_procedure = action.get('gui_procedure', [])

        # Check if action ID contains FILTER and SLIDER
        if 'FILTER' not in action_id or 'SLIDER' not in action_id:
            return False

        # Check if gui_procedure contains click + drag with to_value
        if len(gui_procedure) >= 2:
            for i in range(len(gui_procedure) - 1):
                if (gui_procedure[i].get('op') == 'click' and
                    gui_procedure[i + 1].get('op') == 'drag' and
                    'to_value' in gui_procedure[i + 1]):
                    return True

        return False

    def _is_click_filtered_action(self, action: dict) -> bool:
        """Check if action is CLICK .row-filtered."""
        gui_procedure = action.get('gui_procedure', [])

        if len(gui_procedure) == 1:
            step = gui_procedure[0]
            if (step.get('op') == 'click' and
                '.row-filtered' in step.get('selector', '')):
                return True

        return False

    def _merge_filter_and_click(self, filter_action: dict, click_action: dict) -> dict:
        """Merge FILTER and CLICK actions into a single filter_and_click action."""
        # Extract slider info from filter action
        slider_selector = None
        to_value = None

        for i, step in enumerate(filter_action['gui_procedure']):
            if step.get('op') == 'click' and i + 1 < len(filter_action['gui_procedure']):
                next_step = filter_action['gui_procedure'][i + 1]
                if next_step.get('op') == 'drag' and 'to_value' in next_step:
                    slider_selector = step.get('selector')
                    to_value = next_step.get('to_value')
                    break

        # Extract target item info from click action
        click_selector = click_action['gui_procedure'][0].get('selector', '')
        params = click_action.get('params', {})

        # Create merged action
        merged_action = {
            'id': f"{filter_action['id']}_MERGED_{click_action['id']}",
            'name': 'filter_and_click',
            'from': filter_action['from'],
            'to': click_action['to'],
            'params': params,
            'gui_procedure': [
                {
                    'op': 'filter_and_click',
                    'slider_selector': slider_selector,
                    'to_value': to_value,
                    'click_selector': click_selector,
                    'params': params
                }
            ]
        }

        return merged_action
    
    # Pattern: .xxx-matched, .row-visible, .draft-filtered, etc.
    ROW_SELECTOR_PATTERN = re.compile(r'\.[\w-]*(matched|visible|filtered)')

    def _extract_container_from_row_selector(self, selector: str) -> Optional[str]:
        """Extract container by removing row-matched/visible/filtered part.

        Example: '#drafts-list .draft-row-matched' → '#drafts-list'
        """
        match = self.ROW_SELECTOR_PATTERN.search(selector)
        if match:
            return selector[:match.start()].strip()
        return None

    def process_click_drag_steps(self, gui_steps: List[dict], params: dict) -> List[dict]:
        """Process click+drag combinations and parameterized clicks."""
        if not params:
            return gui_steps

        processed_steps = []
        i = 0

        while i < len(gui_steps):
            current_step = gui_steps[i]
            selector = current_step.get('selector', '')

            # Check for click + drag combination
            if (i + 1 < len(gui_steps) and
                current_step.get('op') == 'click' and
                gui_steps[i + 1].get('op') == 'drag'):

                drag_step = gui_steps[i + 1]

                if current_step.get('selector') == drag_step.get('selector'):
                    base_selector = selector

                    # Extract container by removing row-matched/visible/filtered
                    container = self._extract_container_from_row_selector(base_selector)
                    if container is not None:
                        base_selector = container

                    child_selector = self._build_child_selector(params)

                    self.logger.info(
                        f"    🔄 click+drag → scroll_to_child: {base_selector} → {child_selector}"
                    )

                    processed_steps.append({
                        'op': 'scroll_to_child',
                        'selector': base_selector,
                        'child_selector': child_selector,
                        'params': params
                    })

                    i += 2
                    continue

            # Check for parameterized click with row-matched/visible/filtered
            elif current_step.get('op') == 'click':
                container = self._extract_container_from_row_selector(selector)
                if container is not None:
                    child_selector = self._build_child_selector(params)
                    enhanced_selector = f"{container} {child_selector}"
                    self.logger.info(
                        f"    🔄 Parameterized click: {selector} → {enhanced_selector}"
                    )

                    processed_steps.append({
                        **current_step,
                        'selector': enhanced_selector
                    })

                    i += 1
                    continue

            processed_steps.append(current_step)
            i += 1

        return processed_steps
    
    async def process_select_steps(self, gui_steps: List[dict], action_name: str, params: dict) -> List[dict]:
        """Process select action with hover+click or click+click strategy."""
        if (action_name != 'select' or
            len(gui_steps) != 2 or
            len(params) != 2 or
            gui_steps[0].get('op') != 'hover' or
            gui_steps[1].get('op') != 'click'):
            return gui_steps

        second_selector = gui_steps[1].get('selector', '')
        pattern = r'^(#[\w-]+)\s+\.option$'
        match = re.match(pattern, second_selector)

        if not match:
            return gui_steps

        # Find the option value from params (could be 'option', 'dest_mailbox_id', etc.)
        option_value = None
        for key, value in params.items():
            if key != 'widget' and value:
                option_value = value
                break

        if not option_value:
            return gui_steps

        id_selector = match.group(1)
        option_selector = f".option-{option_value}"

        # Check if option is visible (not just exists)
        element = await self.page.query_selector(option_selector)
        is_visible = False

        if element:
            try:
                is_visible = await element.is_visible()
            except:
                is_visible = False

        if is_visible:
            self.logger.info(f"    🔄 select: hover+click → {id_selector}, {option_selector}")
            return [
                {'op': 'hover', 'selector': id_selector},
                {'op': 'click', 'selector': option_selector}
            ]
        else:
            self.logger.info(f"    🔄 select: click+click → {id_selector}, {option_selector}")
            return [
                {'op': 'click', 'selector': id_selector},
                {'op': 'click', 'selector': option_selector}
            ]
    
    def process_datepicker_steps(self, gui_steps: List[dict]) -> List[dict]:
        """Insert .ym-vertical click before year selection in datepicker."""
        processed_steps = []
        has_datepicker = False

        for step in gui_steps:
            selector = step.get('selector', '')
            op = step.get('op', '')

            if op == 'click' and 'date-picker' in selector:
                has_datepicker = True
                processed_steps.append(step)
                continue

            if has_datepicker and op == 'click' and re.match(r'\.year-\d+', selector):
                processed_steps.append({
                    'op': 'click',
                    'selector': '.ym-vertical'
                })
                processed_steps.append(step)
                self.logger.info(
                    f"    🔄 Datepicker: Inserted .ym-vertical before {selector}"
                )
                has_datepicker = False
                continue

            processed_steps.append(step)

        return processed_steps

    async def normalize_selector(self, selector: str) -> str:
        """Normalize selector by removing leading zeros (e.g., .minute-00 → .minute-0)."""
        if not selector:
            return selector

        try:
            element = await self.page.query_selector(selector)
            if element:
                return selector
        except:
            pass

        match = re.search(r'-0(\d)(?=\s|$|\.|\[)', selector)
        if match:
            digit = match.group(1)
            normalized = selector.replace(f'-0{digit}', f'-{digit}', 1)

            try:
                element = await self.page.query_selector(normalized)
                if element:
                    self.logger.info(f"    🔄 Normalized: {selector} → {normalized}")
                    return normalized
            except:
                pass

        return selector

    def _build_child_selector(self, params: dict) -> str:
        """Build child selector from params."""
        class_selectors = []
        for key, value in params.items():
            suffix = key.split('_', 1)[1] if '_' in key else key
            class_selectors.append(f".data-{suffix}-{value}")
        return ''.join(class_selectors)

    def _extract_macro_name(self, path: Path) -> Optional[str]:
        if path.name == 'bfs.json':
            return path.parent.name
        else:
            return path.stem

    def _load_slider_metadata(self) -> Optional[Dict]:
        if self._slider_metadata_cache is not None:
            return self._slider_metadata_cache

        if self.trajectory_data and 'slider' in self.trajectory_data:
            slider_data = self.trajectory_data['slider']
            if slider_data:
                self._slider_metadata_cache = slider_data
                self.logger.info("✅ Loaded slider metadata from trajectory file (new format)")
                return slider_data

        if not self.summary_path or not self.summary_path.exists():
            self.logger.warning("No slider metadata available (no trajectory data or summary file)")
            return None

        if not self.current_bfs_file:
            self.logger.warning("Current BFS file path not set")
            return None

        try:
            with open(self.summary_path, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)

            current_macro = self._extract_macro_name(self.current_bfs_file)
            if not current_macro:
                self.logger.warning(f"Could not extract macro name from {self.current_bfs_file}")
                return None

            for entry in summary_data:
                bfs_file = entry.get('bfs_file', '')
                entry_macro = self._extract_macro_name(Path(bfs_file))

                if entry_macro == current_macro:
                    slider_data = entry.get('slider', {})
                    self._slider_metadata_cache = slider_data
                    self.logger.info(f"✅ Loaded slider metadata for {current_macro} from summary.json")
                    return slider_data

            self.logger.warning(f"No slider metadata found for {current_macro}")
            return None

        except Exception as e:
            self.logger.error(f"Failed to load slider metadata: {e}")
            return None

    def get_slider_config(self, selector: str) -> Optional[Dict]:
        slider_metadata = self._load_slider_metadata()
        if not slider_metadata:
            return None

        if 'target_value' in slider_metadata or 'direction' in slider_metadata:
            return slider_metadata

        sliders = slider_metadata.get('sliders', [])
        for slider in sliders:
            if slider.get('selector') == selector:
                return slider

        return None
