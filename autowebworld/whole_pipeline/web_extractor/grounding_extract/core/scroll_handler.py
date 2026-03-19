"""Scroll handling for grounding extraction (Refactored)."""

from typing import List, Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw


class ScrollHandler:
    """Handle scrolling operations and grounding extraction.

    Two types of scroll operations:
    1. Page Scroll: Scroll the entire page (bbox = viewport size)
    2. Container Scroll: Scroll a container (bbox = container size)
    """

    def __init__(
        self,
        page,
        grounding_extractor,
        logger,
        scroll_step_size: int = 100,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        preprocessor = None,
        max_scroll_iterations: int = 50
    ):
        self.page = page
        self.grounding = grounding_extractor
        self.logger = logger
        self.scroll_step_size = scroll_step_size
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.last_selector: Optional[str] = None
        self.preprocessor = preprocessor
        self.max_scroll_iterations = max_scroll_iterations
    
    # ========== High-level Methods ==========
    
    async def scroll_into_view(self, selector: str) -> List[Dict]:
        """Scroll element into view if needed.

        Determines whether to use page scroll or container scroll.
        Also checks if element is within its scrollable parent container.

        Returns:
            List of scroll actions with format:
            [{"container": container_selector or None, "grounding": {...}}, ...]

        Raises:
            Exception: If element cannot be scrolled into view
        """
        # Check if element is already visible in viewport
        is_visible_in_viewport = await self.page.evaluate("""
            (selector) => {
                const element = document.querySelector(selector);
                if (!element) return false;

                const rect = element.getBoundingClientRect();
                return (
                    rect.top >= 0 &&
                    rect.left >= 0 &&
                    rect.bottom <= window.innerHeight &&
                    rect.right <= window.innerWidth
                );
            }
        """, selector)

        # Find scrollable parent container
        scroll_info = await self._find_scrollable_parent(selector)
        scroll_steps = []

        # If element is visible in viewport
        if is_visible_in_viewport:
            # Check if it has a scrollable parent
            if scroll_info:
                container_selector = scroll_info['containerSelector']
                # Check if element is within its scrollable parent's bbox
                is_within_parent = await self._check_target_bbox_within_parent_bbox(
                    container_selector, selector
                )

                if not is_within_parent:
                    # Element is in viewport but not within its scrollable parent
                    # Need to perform container scroll
                    self.logger.info(
                        "    Element in viewport but not within scrollable parent, "
                        "performing container scroll"
                    )
                    scroll_steps = await self._perform_container_scroll_until_visible(
                        container_selector, selector
                    )
                else:
                    # Element is in viewport and within scrollable parent
                    return []
            else:
                # Element is in viewport and has no scrollable parent
                return []
        else:
            # Element is not visible in viewport
            if not scroll_info:
                # No scrollable parent, use page scroll
                scroll_steps = await self._perform_page_scroll_until_visible(selector)
            else:
                # Has scrollable parent
                container_selector = scroll_info['containerSelector']

                container_in_viewport = await self._is_element_in_viewport(
                    container_selector
                )
                if not container_in_viewport:
                    self.logger.info(
                        f"    📜 Container {container_selector} not in viewport, "
                        "performing page scroll first"
                    )
                    page_scroll_steps = await self._perform_page_scroll_until_visible(
                        container_selector
                    )
                    scroll_steps.extend(page_scroll_steps)

                is_within = await self._check_target_bbox_within_parent_bbox(
                    container_selector, selector
                )
                if not is_within:
                    self.logger.info(
                        f"    📦 Target not within container, "
                        "performing container scroll"
                    )
                    container_scroll_steps = (
                        await self._perform_container_scroll_until_visible(
                            container_selector, selector
                        )
                    )
                    scroll_steps.extend(container_scroll_steps)

        await self._verify_element_visible_after_scroll(selector, scroll_info)

        return scroll_steps

    async def _is_element_in_viewport(self, selector: str) -> bool:
        """Check if element is within viewport."""
        return await self.page.evaluate("""
            (selector) => {
                const element = document.querySelector(selector);
                if (!element) return false;

                const rect = element.getBoundingClientRect();
                return (
                    rect.top >= 0 &&
                    rect.left >= 0 &&
                    rect.bottom <= window.innerHeight &&
                    rect.right <= window.innerWidth
                );
            }
        """, selector)

    async def _verify_element_visible_after_scroll(
        self,
        selector: str,
        scroll_info: Optional[Dict]
    ) -> None:
        visibility = await self.page.evaluate("""
            (selector) => {
                const element = document.querySelector(selector);
                if (!element) return { error: 'not_found' };

                const rect = element.getBoundingClientRect();
                return {
                    inViewport: (
                        rect.top >= 0 &&
                        rect.left >= 0 &&
                        rect.bottom <= window.innerHeight &&
                        rect.right <= window.innerWidth
                    ),
                    bbox: [
                        Math.round(rect.left),
                        Math.round(rect.top),
                        Math.round(rect.right),
                        Math.round(rect.bottom)
                    ]
                };
            }
        """, selector)

        if visibility.get('error') == 'not_found':
            raise Exception(f"Element {selector} not found after scroll")

        if not visibility['inViewport']:
            bbox = visibility['bbox']
            raise Exception(
                f"Element {selector} bbox {bbox} still outside viewport "
                f"[{self.viewport_width}x{self.viewport_height}] after scroll"
            )

        if scroll_info:
            container_selector = scroll_info.get('containerSelector')
            if container_selector:
                is_within = await self._check_target_bbox_within_parent_bbox(
                    container_selector, selector
                )
                if not is_within:
                    raise Exception(
                        f"Element {selector} still not within container "
                        f"{container_selector} bbox after scroll"
                    )
    
    # ========== Core Scroll Methods ==========
    
    async def perform_page_scroll(self, direction: str) -> Dict[str, Any]:
        """Perform a single page scroll step.

        Args:
            direction: 'up' or 'down'

        Returns:
            Grounding data dict with viewport bbox
        """
        # Extract image_before
        image_before = await self.grounding.extract_before(self.page)

        # Perform scroll
        delta = -self.scroll_step_size if direction == 'up' else self.scroll_step_size
        await self.page.evaluate(f"window.scrollBy(0, {delta})")

        # Wait for scroll animation
        await self.page.wait_for_timeout(100)

        # Extract image_after
        image_after = await self.grounding.extract_after(self.page)

        # Extract viewport grounding
        viewport_grounding = await self.extract_viewport_grounding()

        # Increment counter
        self.grounding.counter += 1

        return {
            "image_before": image_before,
            "image_after": image_after,
            "bbox": viewport_grounding["bbox"],
            "center_point": viewport_grounding.get("center_point", [0, 0]),
            "image_bbox": viewport_grounding["image_bbox"],
            "scroll": delta
        }

    async def perform_page_scroll_if_effective(
        self,
        direction: str,
        target_selector: str
    ) -> Optional[Dict[str, Any]]:
        """Perform page scroll only if it actually moves the target element.

        This method checks bbox before and after scroll. If bbox doesn't change,
        no screenshots or annotations are generated.

        Args:
            direction: 'up' or 'down'
            target_selector: CSS selector of element to check for movement

        Returns:
            Grounding data dict if scroll was effective, None if ineffective
        """
        # Get bbox before scroll
        bbox_before = await self._get_element_bbox(target_selector)

        # Execute scroll (without generating any screenshots yet)
        delta = -self.scroll_step_size if direction == 'up' else self.scroll_step_size
        await self.page.evaluate(f"window.scrollBy(0, {delta})")

        # Wait for scroll animation
        await self.page.wait_for_timeout(100)

        # Get bbox after scroll
        bbox_after = await self._get_element_bbox(target_selector)

        # Check if scroll was effective
        if bbox_after == bbox_before:
            self.logger.warning(f"    Page scroll had no effect, not recording")
            return None

        # Scroll was effective, now generate grounding data
        # Note: We need to take "before" screenshot from current state
        # but that's actually the "after" state. For scroll operations,
        # we capture the current viewport state as the result.
        image_before = await self.grounding.extract_before(self.page)
        image_after = await self.grounding.extract_after(self.page)
        viewport_grounding = await self.extract_viewport_grounding()
        self.grounding.counter += 1

        return {
            "image_before": image_before,
            "image_after": image_after,
            "bbox": viewport_grounding["bbox"],
            "center_point": viewport_grounding.get("center_point", [0, 0]),
            "image_bbox": viewport_grounding["image_bbox"],
            "scroll": delta
        }

    async def _get_element_bbox(self, selector: str) -> Optional[tuple]:
        """Get element bbox as tuple for comparison."""
        bbox_info = await self.page.evaluate("""
            (selector) => {
                const element = document.querySelector(selector);
                if (!element) return null;
                const rect = element.getBoundingClientRect();
                return {
                    x1: Math.round(rect.left),
                    y1: Math.round(rect.top),
                    x2: Math.round(rect.right),
                    y2: Math.round(rect.bottom)
                };
            }
        """, selector)

        if not bbox_info:
            return None
        return (
            bbox_info.get('x1'),
            bbox_info.get('y1'),
            bbox_info.get('x2'),
            bbox_info.get('y2')
        )
    
    async def perform_container_scroll(
        self,
        container_selector: str,
        direction: str,
        target_selector: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform a single container scroll step.

        Args:
            container_selector: CSS selector for the container
            direction: 'up' or 'down'
            target_selector: Optional CSS selector for the target element.
                           If provided, will check if target bbox is within container bbox
                           in viewport coordinates (current screenshot):
                           - If target bbox NOT within container bbox: extract container's bbox
                           - If target bbox within container bbox: extract target element's bbox

        Returns:
            Grounding data dict with container or target element bbox
        """
        # Extract image_before
        image_before = await self.grounding.extract_before(self.page)

        # Perform scroll (handles OverlayScrollbars by checking for viewport element)
        delta = -self.scroll_step_size if direction == 'up' else self.scroll_step_size
        await self.page.evaluate("""
            ({containerSelector, delta}) => {
                const container = document.querySelector(containerSelector);
                if (container) {
                    // Check if this is an OverlayScrollbars container
                    // Try both: .os-viewport class and [data-overlayscrollbars-viewport] attribute
                    const osViewport = container.querySelector(':scope > .os-viewport') ||
                                       container.querySelector(':scope > [data-overlayscrollbars-viewport]');
                    const scrollTarget = osViewport || container;
                    scrollTarget.scrollTop += delta;
                }
            }
        """, {'containerSelector': container_selector, 'delta': delta})

        # Wait for scroll animation
        await self.page.wait_for_timeout(100)

        # Extract image_after
        image_after = await self.grounding.extract_after(self.page)

        # Determine which element to extract grounding for
        grounding_selector = container_selector

        # If target_selector is provided, check if target bbox is within container bbox in viewport
        if target_selector:
            bbox_detail = await self.page.evaluate("""
                ({containerSelector, targetSelector}) => {
                    const container = document.querySelector(containerSelector);
                    const target = document.querySelector(targetSelector);
                    if (!container || !target) {
                        return {
                            error: !container ? 'container_not_found' : 'target_not_found'
                        };
                    }
                    const cRect = container.getBoundingClientRect();
                    const tRect = target.getBoundingClientRect();
                    return {
                        container: [
                            Math.round(cRect.left), Math.round(cRect.top),
                            Math.round(cRect.right), Math.round(cRect.bottom)
                        ],
                        target: [
                            Math.round(tRect.left), Math.round(tRect.top),
                            Math.round(tRect.right), Math.round(tRect.bottom)
                        ]
                    };
                }
            """, {'containerSelector': container_selector, 'targetSelector': target_selector})

            self.logger.info(
                f"    🔍 Container: {container_selector} bbox={bbox_detail.get('container')}"
            )
            self.logger.info(
                f"    🔍 Target: {target_selector} bbox={bbox_detail.get('target')}"
            )

            is_target_within = await self._check_target_bbox_within_parent_bbox(
                container_selector, target_selector
            )

            if is_target_within:
                # Target bbox is within container bbox, extract target's bbox
                grounding_selector = target_selector
                self.logger.info(
                    "    ✅ Target element bbox is within container bbox, "
                    "extracting target bbox"
                )
            else:
                # Target bbox is not within container bbox, extract container's bbox
                grounding_selector = container_selector
                self.logger.info(
                    "    📦 Target element bbox not within container bbox, "
                    "extracting container bbox"
                )

        # Extract grounding for the determined selector
        if grounding_selector == target_selector:
            # Extract target element's bbox (no clipping)
            element_grounding = await self.extract_element_grounding(grounding_selector)
            bbox = element_grounding["bbox"]
            center_point = element_grounding.get("center_point", [0, 0])
            image_bbox = element_grounding["image_bbox"]
        else:
            # Extract container's bbox
            container_grounding = await self.extract_container_grounding(grounding_selector)
            bbox = container_grounding["bbox"]
            center_point = container_grounding.get("center_point", [0, 0])
            image_bbox = container_grounding["image_bbox"]

        # Increment counter
        self.grounding.counter += 1

        return {
            "image_before": image_before,
            "image_after": image_after,
            "bbox": bbox,
            "center_point": center_point,
            "image_bbox": image_bbox,
            "scroll": delta
        }

    async def perform_container_scroll_if_effective(
        self,
        container_selector: str,
        direction: str,
        target_selector: str
    ) -> Optional[Dict[str, Any]]:
        """Perform container scroll only if it actually moves the target element.

        This method checks bbox before and after scroll. If bbox doesn't change,
        no screenshots or annotations are generated.

        Args:
            container_selector: CSS selector for the container
            direction: 'up' or 'down'
            target_selector: CSS selector of element to check for movement

        Returns:
            Grounding data dict if scroll was effective, None if ineffective
        """
        # Get bbox before scroll
        bbox_before = await self._get_element_bbox(target_selector)

        # Execute scroll (without generating any screenshots yet)
        delta = -self.scroll_step_size if direction == 'up' else self.scroll_step_size
        await self.page.evaluate("""
            ({containerSelector, delta}) => {
                const container = document.querySelector(containerSelector);
                if (container) {
                    const osViewport = container.querySelector(':scope > .os-viewport') ||
                                       container.querySelector(':scope > [data-overlayscrollbars-viewport]');
                    const scrollTarget = osViewport || container;
                    scrollTarget.scrollTop += delta;
                }
            }
        """, {'containerSelector': container_selector, 'delta': delta})

        # Wait for scroll animation
        await self.page.wait_for_timeout(100)

        # Get bbox after scroll
        bbox_after = await self._get_element_bbox(target_selector)

        # Check if scroll was effective
        if bbox_after == bbox_before:
            self.logger.warning(f"    Container scroll had no effect, not recording")
            return None

        # Scroll was effective, now generate grounding data
        image_before = await self.grounding.extract_before(self.page)
        image_after = await self.grounding.extract_after(self.page)

        # Determine which element to extract grounding for
        is_target_within = await self._check_target_bbox_within_parent_bbox(
            container_selector, target_selector
        )

        if is_target_within:
            element_grounding = await self.extract_element_grounding(target_selector)
            bbox = element_grounding["bbox"]
            center_point = element_grounding.get("center_point", [0, 0])
            image_bbox = element_grounding["image_bbox"]
        else:
            container_grounding = await self.extract_container_grounding(container_selector)
            bbox = container_grounding["bbox"]
            center_point = container_grounding.get("center_point", [0, 0])
            image_bbox = container_grounding["image_bbox"]

        self.grounding.counter += 1

        return {
            "image_before": image_before,
            "image_after": image_after,
            "bbox": bbox,
            "center_point": center_point,
            "image_bbox": image_bbox,
            "scroll": delta
        }

    async def get_slider_handle_position(
        self,
        slider_selector: str,
        target_value: float = None
    ) -> Dict[str, float]:
        """Get slider handle center position for a specific value.

        Args:
            slider_selector: CSS selector for the slider
            target_value: Optional value to calculate position for. If None, uses current slider value.

        Returns:
            Dict with x, y coordinates of handle center
        """
        handle_pos = await self.page.evaluate("""
            ({sliderSelector, targetValue}) => {
                const slider = document.querySelector(sliderSelector);
                if (!slider) return null;

                const rect = slider.getBoundingClientRect();
                const min = parseFloat(slider.getAttribute('min') || '0');
                const max = parseFloat(slider.getAttribute('max') || '100');

                // Use the specified targetValue; otherwise use the current value
                const value = targetValue !== null ? targetValue : parseFloat(slider.value || min);

                // Estimate the handle width (roughly 1.5x the slider height)
                const handleWidth = rect.height * 1.5;

                // Effective sliding range = total width - handle width
                const effectiveWidth = rect.width - handleWidth;

                // Compute the handle center position
                // When value=min, the handle center is at rect.left + handleWidth/2
                // When value=max, the handle center is at rect.right - handleWidth/2
                const ratio = (value - min) / (max - min);
                const handleX = rect.left + handleWidth / 2 + ratio * effectiveWidth;
                const handleY = rect.top + rect.height / 2;

                return {
                    x: Math.ceil(handleX),
                    y: Math.ceil(handleY)
                };
            }
        """, {'sliderSelector': slider_selector, 'targetValue': target_value})

        if not handle_pos:
            return {"x": 0, "y": 0}

        return handle_pos

    async def extract_filter_and_click(
        self,
        step: Dict,
        current_step_index: int,
        num_intermediate_steps: int = 3
    ) -> List[Dict]:
        slider_selector = step.get('slider_selector', '')
        to_value = step.get('to_value', '')
        click_selector = step.get('click_selector', '')
        params = step.get('params', {})

        self.logger.info(f"  🎯 Filter and Click:")
        self.logger.info(f"     Slider: {slider_selector}")
        self.logger.info(f"     Direction (BFS): {to_value}")
        self.logger.info(f"     Target: {click_selector}")
        self.logger.info(f"     Params: {params}")

        slider_config = None
        if self.preprocessor:
            slider_config = self.preprocessor.get_slider_config(slider_selector)
            if slider_config:
                self.logger.info(f"     ✅ Found slider metadata:")
                self.logger.info(f"        Target value: {slider_config.get('target_value')}")
                self.logger.info(f"        Direction: {slider_config.get('direction')}")
                if 'comparison' in slider_config:
                    self.logger.info(f"        Comparison: {slider_config.get('comparison')}")

        # Extract container and build target selector
        container_selector = click_selector.split('.row-filtered')[0].strip()
        item_id = params.get('item_id', '')
        target_selector = f"{container_selector} .data-id-{item_id}"

        self.logger.info(f"     Container: {container_selector}")
        self.logger.info(f"     Target item: {target_selector}")

        # Get slider properties
        slider_info = await self.page.evaluate("""
            (sliderSelector) => {
                const slider = document.querySelector(sliderSelector);
                if (!slider) return null;

                const rect = slider.getBoundingClientRect();
                const min = parseFloat(slider.getAttribute('min') || '0');
                const max = parseFloat(slider.getAttribute('max') || '100');
                const current = parseFloat(slider.value || min);

                return {
                    bbox: {
                        x: Math.round(rect.left),
                        y: Math.round(rect.top),
                        width: Math.round(rect.width),
                        height: Math.round(rect.height)
                    },
                    min: min,
                    max: max,
                    current: current,
                    range: max - min
                };
            }
        """, slider_selector)

        if not slider_info:
            self.logger.error(f"  ❌ Slider not found: {slider_selector}")
            return []

        self.logger.info(f"     Slider range: {slider_info['min']} - {slider_info['max']}")

        if not slider_config:
            error_msg = f"❌ Slider metadata not found for {slider_selector}. Please ensure trajectory data or summary.json contains slider configuration."
            self.logger.error(f"  {error_msg}")
            raise ValueError(error_msg)

        target_value = slider_config.get('target_value')
        direction = slider_config.get('direction', to_value)

        import math
        original_target = target_value
        target_value = math.floor(target_value)
        if original_target != target_value:
            self.logger.info(f"     Target value rounded down: {original_target} → {target_value}")

        slider_min = slider_info['min']
        slider_max = slider_info['max']
        if target_value < slider_min or target_value > slider_max:
            error_msg = (
                f"❌ Slider target value out of range!\n"
                f"   Slider: {slider_selector}\n"
                f"   Actual range: [{slider_min}, {slider_max}]\n"
                f"   Metadata target: {target_value}\n"
                f"   The target value from metadata ({target_value}) exceeds the slider's actual range.\n"
            )
            self.logger.error(f"  {error_msg}")
            raise ValueError(error_msg)

        if direction == 'left':
            start_value = slider_info['max']
            end_value = target_value
        else:  # 'right'
            start_value = slider_info['min']
            end_value = target_value

        self.logger.info(f"     Using metadata: {start_value} → {end_value} (direction: {direction})")

        # Set slider to start position
        await self.page.evaluate("""
            ({sliderSelector, value}) => {
                const slider = document.querySelector(sliderSelector);
                if (slider) {
                    slider.value = value;
                    slider.dispatchEvent(new Event('input', { bubbles: true }));
                    slider.dispatchEvent(new Event('change', { bubbles: true }));
                }
            }
        """, {'sliderSelector': slider_selector, 'value': start_value})

        await self.page.wait_for_timeout(100)

        self.logger.info(f"     Generating {num_intermediate_steps} positions from {start_value} to {end_value}")

        if num_intermediate_steps < 2:
            intermediate_positions = [start_value, end_value]
        else:
            intermediate_positions = []
            for i in range(num_intermediate_steps):
                ratio = i / (num_intermediate_steps - 1)
                position = start_value + (end_value - start_value) * ratio
                intermediate_positions.append(position)

        self.logger.info(f"     Positions: {[f'{p:.1f}' for p in intermediate_positions]}")
        self.logger.info(f"     Will generate {len(intermediate_positions) - 1} grounding steps")

        steps_result = []

        for i in range(len(intermediate_positions) - 1):
            current_position = intermediate_positions[i]
            next_position = intermediate_positions[i + 1]

            self.logger.info(f"     Capturing step {i+1}/{len(intermediate_positions)-1}: {current_position:.1f} → {next_position:.1f}")

            await self.page.evaluate("""
                ({sliderSelector, value}) => {
                    const slider = document.querySelector(sliderSelector);
                    if (slider) {
                        slider.value = value;
                        slider.dispatchEvent(new Event('input', { bubbles: true }));
                        slider.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                }
            """, {'sliderSelector': slider_selector, 'value': current_position})

            await self.page.wait_for_timeout(100)

            handle_before = await self.get_slider_handle_position(slider_selector, current_position)
            drag_x1 = handle_before['x']
            drag_y1 = handle_before['y']

            handle_after = await self.get_slider_handle_position(slider_selector, next_position)
            drag_x2 = handle_after['x']
            drag_y2 = handle_after['y']

            current_before = await self.grounding.extract_before(self.page)
            current_annotation = await self.grounding.extract_annotation(
                self.page,
                slider_selector,
                self.viewport_width,
                self.viewport_height,
                custom_center_point=(int(drag_x1), int(drag_y1))
            )

            await self.page.evaluate("""
                ({sliderSelector, value}) => {
                    const slider = document.querySelector(sliderSelector);
                    if (slider) {
                        slider.value = value;
                        slider.dispatchEvent(new Event('input', { bubbles: true }));
                        slider.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                }
            """, {'sliderSelector': slider_selector, 'value': next_position})

            await self.page.wait_for_timeout(100)

            current_after = await self.grounding.extract_after(self.page)

            # Calculate drag distance and direction
            drag_dx = drag_x2 - drag_x1
            drag_dy = drag_y2 - drag_y1
            drag_distance = (drag_dx ** 2 + drag_dy ** 2) ** 0.5

            # Determine drag direction
            if abs(drag_dx) > abs(drag_dy):
                drag_direction = 'right' if drag_dx > 0 else 'left'
            else:
                drag_direction = 'down' if drag_dy > 0 else 'up'

            # Increment counter
            self.grounding.counter += 1

            # Add slider drag step
            steps_result.append({
                "index": current_step_index + i,
                "operation": "drag",
                "selector": slider_selector,
                "text": "",
                "key": "",
                "grounding": {
                    "image_before": current_before,
                    "image_after": current_after,
                    "bbox": current_annotation.get('bbox', [0, 0, 0, 0]),
                    "center_point": [round(drag_x1, 2), round(drag_y1, 2)],
                    "image_bbox": current_annotation.get('image_bbox', ''),
                    "drag_start": [round(drag_x1, 2), round(drag_y1, 2)],
                    "drag_end": [round(drag_x2, 2), round(drag_y2, 2)],
                    "drag_distance": round(drag_distance, 2),
                    "drag_direction": drag_direction
                }
            })

        current_step_index += len(intermediate_positions) - 1

        # Now click the target item (ensure it's in viewport)
        # Use scroll_to_child if needed
        target_in_viewport = await self._check_element_in_viewport(target_selector)

        if not target_in_viewport:
            # Scroll target into view within container
            scroll_steps = await self.extract_scroll_to_child(
                {
                    'selector': container_selector,
                    'child_selector': target_selector,
                    'params': params
                },
                current_step_index
            )
            steps_result.extend(scroll_steps)
            current_step_index += len(scroll_steps)

        # Extract click grounding (before click to capture element bbox)
        image_before = await self.grounding.extract_before(self.page)

        # Get target annotation before clicking (element might disappear after click)
        target_annotation = await self.grounding.extract_annotation(
            self.page,
            target_selector,
            self.viewport_width,
            self.viewport_height
        )

        # Click the target
        target_element = await self.page.query_selector(target_selector)
        if target_element:
            self.logger.info(f"     ✅ Clicking target: {target_selector}")

            # Debug: log element info
            element_info = await target_element.evaluate("""
                (el) => {
                    return {
                        tagName: el.tagName,
                        className: el.className,
                        id: el.id,
                        textContent: el.textContent.substring(0, 50)
                    };
                }
            """)
            self.logger.info(f"     Element info: {element_info}")

            await target_element.click()
            await self.page.wait_for_timeout(100)

            # Log current URL after click
            current_url = self.page.url
            self.logger.info(f"     Current URL after click: {current_url}")
        else:
            self.logger.error(f"  ❌ Target element not found: {target_selector}")

        image_after = await self.grounding.extract_after(self.page)

        self.grounding.counter += 1

        steps_result.append({
            "index": current_step_index,
            "operation": "click",
            "selector": target_selector,
            "text": "",
            "key": "",
            "grounding": {
                "image_before": image_before,
                "image_after": image_after,
                "bbox": target_annotation.get('bbox', [0, 0, 0, 0]),
                "center_point": target_annotation.get('center_point', [0, 0]),
                "image_bbox": target_annotation.get('image_bbox', '')
            }
        })

        return steps_result

    async def _check_element_in_viewport(self, selector: str) -> bool:
        """Check if element is in viewport."""
        result = await self.page.evaluate("""
            (selector) => {
                const element = document.querySelector(selector);
                if (!element) return false;

                const rect = element.getBoundingClientRect();
                return (
                    rect.top >= 0 &&
                    rect.left >= 0 &&
                    rect.bottom <= window.innerHeight &&
                    rect.right <= window.innerWidth
                );
            }
        """, selector)
        return result

    async def extract_slider(
        self,
        step: Dict,
        current_step_index: int,
        num_intermediate_steps: int = 3
    ) -> List[Dict]:
        """Extract grounding for standalone slider operation.

        Args:
            step: Dict with:
                - selector: CSS selector for the slider
                - to_value: 'left', 'right', or numeric value (e.g., 3, 5)
            current_step_index: Starting index for steps
            num_intermediate_steps: Number of intermediate positions (default: 3)

        Returns:
            List of grounding steps for slider drags
        """
        slider_selector = step.get('selector', '')
        to_value = step.get('to_value', 'right')

        self.logger.info(f"  🎚️ Slider operation:")
        self.logger.info(f"     Selector: {slider_selector}")
        self.logger.info(f"     Direction: {to_value}")

        slider_config = None
        if self.preprocessor:
            slider_config = self.preprocessor.get_slider_config(slider_selector)
            if slider_config:
                self.logger.info(f"     ✅ Found slider metadata:")
                self.logger.info(f"        Target value: {slider_config.get('target_value')}")
                self.logger.info(f"        Direction: {slider_config.get('direction')}")

        # Get slider properties
        slider_info = await self.page.evaluate("""
            (sliderSelector) => {
                const slider = document.querySelector(sliderSelector);
                if (!slider) return null;

                const rect = slider.getBoundingClientRect();
                const min = parseFloat(slider.getAttribute('min') || '0');
                const max = parseFloat(slider.getAttribute('max') || '100');
                const current = parseFloat(slider.value || min);

                return {
                    bbox: {
                        x: Math.round(rect.left),
                        y: Math.round(rect.top),
                        width: Math.round(rect.width),
                        height: Math.round(rect.height)
                    },
                    min: min,
                    max: max,
                    current: current,
                    range: max - min
                };
            }
        """, slider_selector)

        if not slider_info:
            self.logger.error(f"  ❌ Slider not found: {slider_selector}")
            return []

        self.logger.info(f"     Slider range: {slider_info['min']} - {slider_info['max']}")

        # Determine start and end values
        is_numeric_to_value = False
        numeric_target = None
        
        if isinstance(to_value, (int, float)):
            is_numeric_to_value = True
            numeric_target = to_value
        elif isinstance(to_value, str):
            try:
                numeric_target = float(to_value)
                is_numeric_to_value = True
            except (ValueError, TypeError):
                is_numeric_to_value = False
        
        if is_numeric_to_value:
            target_value = numeric_target
            current_value = slider_info.get('current', slider_info['min'])
            direction = 'right' if target_value >= current_value else 'left'
            self.logger.info(f"     ✅ Using to_value as target: {target_value}")
            
            slider_min = slider_info['min']
            slider_max = slider_info['max']
            if target_value < slider_min or target_value > slider_max:
                error_msg = (
                    f"❌ Slider target value out of range!\n"
                    f"   Slider: {slider_selector}\n"
                    f"   Actual range: [{slider_min}, {slider_max}]\n"
                    f"   Provided to_value: {target_value}\n"
                )
                self.logger.error(f"  {error_msg}")
                raise ValueError(error_msg)
        elif slider_config:
            target_value = slider_config.get('target_value')
            direction = slider_config.get('direction', to_value)
            self.logger.info(f"     ✅ Using slider metadata target: {target_value}")

            import math
            original_target = target_value
            target_value = math.floor(target_value)
            if original_target != target_value:
                self.logger.info(f"     Target value rounded down: {original_target} → {target_value}")

            slider_min = slider_info['min']
            slider_max = slider_info['max']
            if target_value < slider_min or target_value > slider_max:
                error_msg = (
                    f"❌ Slider target value out of range!\n"
                    f"   Slider: {slider_selector}\n"
                    f"   Actual range: [{slider_min}, {slider_max}]\n"
                    f"   Metadata target: {target_value}\n"
                )
                self.logger.error(f"  {error_msg}")
                raise ValueError(error_msg)
        else:
            direction = to_value
            if direction == 'right':
                target_value = slider_info['max']
            else:
                target_value = slider_info['min']
            self.logger.warning(f"     ⚠️ No metadata, using fallback: {target_value}")

        if direction == 'left':
            start_value = slider_info['max']
            end_value = target_value
        else:  # 'right'
            start_value = slider_info['min']
            end_value = target_value

        self.logger.info(f"     Sliding: {start_value} → {end_value}")

        # Set slider to start position
        await self.page.evaluate("""
            ({sliderSelector, value}) => {
                const slider = document.querySelector(sliderSelector);
                if (slider) {
                    slider.value = value;
                    slider.dispatchEvent(new Event('input', { bubbles: true }));
                    slider.dispatchEvent(new Event('change', { bubbles: true }));
                }
            }
        """, {'sliderSelector': slider_selector, 'value': start_value})

        await self.page.wait_for_timeout(100)

        # Generate intermediate positions
        if num_intermediate_steps < 2:
            intermediate_positions = [start_value, end_value]
        else:
            intermediate_positions = []
            for i in range(num_intermediate_steps):
                ratio = i / (num_intermediate_steps - 1)
                position = start_value + (end_value - start_value) * ratio
                intermediate_positions.append(position)

        self.logger.info(f"     Positions: {[f'{p:.1f}' for p in intermediate_positions]}")
        self.logger.info(f"     Will generate {len(intermediate_positions) - 1} grounding steps")

        steps_result = []

        for i in range(len(intermediate_positions) - 1):
            current_position = intermediate_positions[i]
            next_position = intermediate_positions[i + 1]

            self.logger.info(f"     Step {i+1}/{len(intermediate_positions)-1}: {current_position:.1f} → {next_position:.1f}")

            # Set slider to current position
            await self.page.evaluate("""
                ({sliderSelector, value}) => {
                    const slider = document.querySelector(sliderSelector);
                    if (slider) {
                        slider.value = value;
                        slider.dispatchEvent(new Event('input', { bubbles: true }));
                        slider.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                }
            """, {'sliderSelector': slider_selector, 'value': current_position})

            await self.page.wait_for_timeout(100)

            handle_before = await self.get_slider_handle_position(slider_selector, current_position)
            drag_x1 = handle_before['x']
            drag_y1 = handle_before['y']

            handle_after = await self.get_slider_handle_position(slider_selector, next_position)
            drag_x2 = handle_after['x']
            drag_y2 = handle_after['y']

            current_before = await self.grounding.extract_before(self.page)
            current_annotation = await self.grounding.extract_annotation(
                self.page,
                slider_selector,
                self.viewport_width,
                self.viewport_height,
                custom_center_point=(int(drag_x1), int(drag_y1))
            )

            # Move to next position
            await self.page.evaluate("""
                ({sliderSelector, value}) => {
                    const slider = document.querySelector(sliderSelector);
                    if (slider) {
                        slider.value = value;
                        slider.dispatchEvent(new Event('input', { bubbles: true }));
                        slider.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                }
            """, {'sliderSelector': slider_selector, 'value': next_position})

            await self.page.wait_for_timeout(100)

            current_after = await self.grounding.extract_after(self.page)

            # Calculate drag info
            drag_dx = drag_x2 - drag_x1
            drag_dy = drag_y2 - drag_y1
            drag_distance = (drag_dx ** 2 + drag_dy ** 2) ** 0.5

            if abs(drag_dx) > abs(drag_dy):
                drag_direction = 'right' if drag_dx > 0 else 'left'
            else:
                drag_direction = 'down' if drag_dy > 0 else 'up'

            self.grounding.counter += 1

            steps_result.append({
                "index": current_step_index + i,
                "operation": "drag",
                "selector": slider_selector,
                "text": "",
                "key": "",
                "grounding": {
                    "image_before": current_before,
                    "image_after": current_after,
                    "bbox": current_annotation.get('bbox', [0, 0, 0, 0]),
                    "center_point": [round(drag_x1, 2), round(drag_y1, 2)],
                    "image_bbox": current_annotation.get('image_bbox', ''),
                    "drag_start": [round(drag_x1, 2), round(drag_y1, 2)],
                    "drag_end": [round(drag_x2, 2), round(drag_y2, 2)],
                    "drag_distance": round(drag_distance, 2),
                    "drag_direction": drag_direction
                }
            })

        return steps_result

    async def extract_scroll_to_child(self, step: Dict, current_step_index: int) -> List[Dict]:
        """Extract grounding for scrolling to a child element (hover + scroll).

        Finds the scrollable parent container of target element and scrolls until visible.

        Args:
            step: GUI step dict with 'selector' (container) and 'child_selector'
            current_step_index: Starting index for steps

        Returns:
            List of scroll step dicts
        """
        base_selector = step.get('selector', '')
        child_selector = step.get('child_selector', '')

        if child_selector.startswith(base_selector):
            target_selector = child_selector
        else:
            target_selector = f"{base_selector} {child_selector}"

        self.logger.info(f"  Scroll to child: {target_selector}")

        steps_result = []
        iteration = 0

        # Find the scrollable parent container for the target element
        scroll_container_info = await self._find_scroll_container_for_element(target_selector)

        if 'error' in scroll_container_info:
            self.logger.error(f"  Error finding scroll container: {scroll_container_info['error']}")
            self.logger.error(f"    Target selector: {target_selector}")
            return steps_result

        scroll_container_selector = scroll_container_info.get('containerSelector', '')
        if not scroll_container_selector:
            # Fallback to page scroll
            scroll_container_selector = 'body'
            self.logger.info(f"  No scrollable container found, using page scroll")

        self.logger.info(f"  Found scroll container: {scroll_container_selector}")

        # Get scroll direction from container info
        container_scroll_direction = scroll_container_info.get('scrollDirection', 'vertical')

        # Track scroll ineffectiveness
        last_child_bbox = None

        while iteration < self.max_scroll_iterations:
            visibility_info = await self._check_child_visibility_in_container(
                scroll_container_selector,
                target_selector,
                container_scroll_direction
            )

            if 'error' in visibility_info:
                self.logger.error(f"  Error: {visibility_info['error']}")
                break

            if visibility_info['isFullyVisible']:
                self.logger.info(f"  Child visible after {iteration} scroll(s)")
                break

            scroll_direction = visibility_info['scrollDirection']
            child_bbox = visibility_info.get('childBbox', {})

            # Check if child bbox hasn't changed (scroll ineffective)
            current_bbox_tuple = (
                child_bbox.get('x1', 0),
                child_bbox.get('y1', 0),
                child_bbox.get('x2', 0),
                child_bbox.get('y2', 0)
            )

            if last_child_bbox is not None and current_bbox_tuple == last_child_bbox:
                self.logger.warning(f"  Scroll ineffective (child bbox unchanged)")
                break

            # Extract container bbox for grounding
            container_bbox = await self._get_container_bbox(scroll_container_selector)

            # Extract image_before
            image_before = await self.grounding.extract_before(self.page)

            # Perform scroll on container
            # Note: perform_page_scroll and perform_container_scroll already increment counter
            if scroll_container_selector == 'body':
                grounding = await self.perform_page_scroll(scroll_direction)
            else:
                grounding = await self.perform_container_scroll(
                    scroll_container_selector,
                    scroll_direction,
                    target_selector=target_selector
                )

            # Build scroll step result
            steps_result.append({
                "index": current_step_index,
                "operation": "scroll",
                "selector": scroll_container_selector,
                "text": "",
                "key": "",
                "grounding": grounding
            })

            # Update tracking variables
            last_child_bbox = current_bbox_tuple
            current_step_index += 1
            iteration += 1

        if iteration >= self.max_scroll_iterations:
            self.logger.warning(f"  Max iterations reached ({self.max_scroll_iterations})")

        self.logger.info(f"  Scrolled {len(steps_result)} steps")
        return steps_result

    async def _get_container_bbox(self, container_selector: str) -> List[int]:
        """Get container bounding box."""
        if container_selector == 'body':
            return [0, 0, self.viewport_width, self.viewport_height]

        bbox = await self.page.evaluate("""
            (selector) => {
                const container = document.querySelector(selector);
                if (!container) return null;
                const rect = container.getBoundingClientRect();
                return [
                    Math.round(rect.left),
                    Math.round(rect.top),
                    Math.round(rect.right),
                    Math.round(rect.bottom)
                ];
            }
        """, container_selector)

        return bbox or [0, 0, self.viewport_width, self.viewport_height]
    
    # ========== Grounding Extraction Methods ==========

    async def _get_scrollable_containers(self) -> List[Dict]:
        return await self.page.evaluate("""
            () => {
                const containers = [];
                const elements = document.querySelectorAll('*');

                for (const el of elements) {
                    if (el === document.body || el === document.documentElement) continue;

                    const style = getComputedStyle(el);
                    const overflowY = style.overflowY;
                    const overflowX = style.overflowX;

                    const isScrollable = (
                        (overflowY === 'auto' || overflowY === 'scroll' ||
                         overflowX === 'auto' || overflowX === 'scroll') &&
                        (el.scrollHeight > el.clientHeight || el.scrollWidth > el.clientWidth)
                    );

                    // Also check OverlayScrollbars
                    const isOsHost = el.classList.contains('os-host') ||
                                     el.hasAttribute('data-overlayscrollbars');

                    if (isScrollable || isOsHost) {
                        const rect = el.getBoundingClientRect();
                        containers.push({
                            x: rect.left,
                            y: rect.top,
                            width: rect.width,
                            height: rect.height,
                            right: rect.right,
                            bottom: rect.bottom
                        });
                    }
                }

                return containers;
            }
        """)

    async def _find_safe_page_scroll_point(self) -> Tuple[int, int]:
        containers = await self._get_scrollable_containers()

        if not containers:
            return (self.viewport_width // 2, self.viewport_height // 2)

        center_y = self.viewport_height // 2

        containers_at_center_y = [
            c for c in containers
            if c['y'] <= center_y <= c['bottom']
        ]

        if not containers_at_center_y:
            return (self.viewport_width // 2, center_y)

        sorted_containers = sorted(containers_at_center_y, key=lambda c: c['x'])

        gaps = []

        first_x = sorted_containers[0]['x']
        if first_x > 0:
            gaps.append({'start': 0, 'end': first_x, 'width': first_x})

        for i in range(len(sorted_containers) - 1):
            gap_start = sorted_containers[i]['right']
            gap_end = sorted_containers[i + 1]['x']
            if gap_end > gap_start:
                gaps.append({'start': gap_start, 'end': gap_end, 'width': gap_end - gap_start})

        last_right = sorted_containers[-1]['right']
        if last_right < self.viewport_width:
            gaps.append({
                'start': last_right,
                'end': self.viewport_width,
                'width': self.viewport_width - last_right
            })

        if gaps:
            best_gap = max(gaps, key=lambda g: g['width'])
            x = int((best_gap['start'] + best_gap['end']) / 2)
            self.logger.info(f"    Found safe scroll point at x={x} (gap: {best_gap['start']:.0f}-{best_gap['end']:.0f})")
            return (x, center_y)

        for test_y in [100, self.viewport_height - 100, 50, self.viewport_height - 50]:
            containers_at_y = [
                c for c in containers
                if c['y'] <= test_y <= c['bottom']
            ]

            if not containers_at_y:
                self.logger.info(f"    Found safe scroll point at y={test_y} (no containers)")
                return (self.viewport_width // 2, test_y)

            sorted_at_y = sorted(containers_at_y, key=lambda c: c['x'])
            test_gaps = []

            first_x = sorted_at_y[0]['x']
            if first_x > 50:
                test_gaps.append({'start': 0, 'end': first_x})

            last_right = sorted_at_y[-1]['right']
            if last_right < self.viewport_width - 50:
                test_gaps.append({'start': last_right, 'end': self.viewport_width})

            if test_gaps:
                best = max(test_gaps, key=lambda g: g['end'] - g['start'])
                x = int((best['start'] + best['end']) / 2)
                self.logger.info(f"    Found safe scroll point at ({x}, {test_y})")
                return (x, test_y)

        self.logger.warning("    ⚠️ No safe scroll point found, using viewport center")
        return (self.viewport_width // 2, self.viewport_height // 2)

    async def extract_viewport_grounding(self) -> Dict[str, Any]:
        # Viewport bbox is always [0, 0, viewport_width, viewport_height]
        bbox = [0, 0, self.viewport_width, self.viewport_height]

        center_x, center_y = await self._find_safe_page_scroll_point()

        # Create annotated screenshot with red box around viewport
        annotated_name = f"{self.grounding.counter:03d}_bbox.png"
        annotated_path = self.grounding.annotations_dir / annotated_name

        temp_path = self.grounding.annotations_dir / f"{self.grounding.counter:03d}_temp.png"
        await self.page.screenshot(path=str(temp_path))

        img = Image.open(temp_path)
        draw = ImageDraw.Draw(img)

        # Draw red box around viewport
        for i in range(3):
            draw.rectangle([i, i, self.viewport_width - i - 1, self.viewport_height - i - 1], outline='red')

        # Draw center point at safe location
        radius = 5
        draw.ellipse(
            [center_x - radius, center_y - radius,
             center_x + radius, center_y + radius],
            fill='red',
            outline='white'
        )

        img.save(annotated_path)
        temp_path.unlink()

        center_point = [center_x, center_y]

        return {
            "bbox": bbox,
            "center_point": center_point,
            "image_bbox": f"annotations/{annotated_name}"
        }

    async def extract_element_grounding(
        self,
        element_selector: str
    ) -> Dict[str, Any]:
        """Extract element grounding (bbox = element's original bbox, no clipping).

        Args:
            element_selector: CSS selector for the target element

        Returns:
            Dict with bbox and image_bbox
        """
        # Get element bbox (no clipping)
        element_info = await self.page.evaluate("""
            (elementSelector) => {
                const element = document.querySelector(elementSelector);

                if (!element) return null;

                const elementRect = element.getBoundingClientRect();

                const bbox = {
                    x1: Math.round(elementRect.left),
                    y1: Math.round(elementRect.top),
                    x2: Math.round(elementRect.right),
                    y2: Math.round(elementRect.bottom)
                };

                return bbox;
            }
        """, element_selector)

        if not element_info:
            return {"bbox": [0, 0, 0, 0], "image_bbox": ""}

        bbox = [
            element_info['x1'],
            element_info['y1'],
            element_info['x2'],
            element_info['y2']
        ]

        # Create annotated screenshot with red box around element
        annotated_name = f"{self.grounding.counter:03d}_bbox.png"
        annotated_path = self.grounding.annotations_dir / annotated_name

        temp_path = self.grounding.annotations_dir / f"{self.grounding.counter:03d}_temp.png"
        await self.page.screenshot(path=str(temp_path))

        # Draw red box and center point
        img = Image.open(temp_path)
        draw = ImageDraw.Draw(img)

        x1, y1, x2, y2 = bbox

        # Ensure valid bbox for drawing
        if x2 > x1 and y2 > y1:
            for i in range(3):
                draw.rectangle([x1+i, y1+i, x2-i, y2-i], outline='red')

            # Draw center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            radius = 5
            draw.ellipse(
                [center_x - radius, center_y - radius,
                 center_x + radius, center_y + radius],
                fill='red',
                outline='white'
            )

        img.save(annotated_path)
        temp_path.unlink()

        # Calculate center point
        center_point = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]

        return {
            "bbox": bbox,
            "center_point": center_point,
            "image_bbox": f"annotations/{annotated_name}"
        }

    async def extract_container_grounding(self, container_selector: str) -> Dict[str, Any]:
        """Extract container grounding (bbox = container size).

        The bbox will be clipped to the first scrollable parent's visible area
        to ensure it doesn't exceed the parent component's bounds.

        Args:
            container_selector: CSS selector for the container

        Returns:
            Dict with bbox and image_bbox
        """
        # Get container bbox and clip it to scrollable parent's visible area
        container_info = await self.page.evaluate("""
            (containerSelector) => {
                const container = document.querySelector(containerSelector);
                if (!container) return null;

                const rect = container.getBoundingClientRect();
                let bbox = {
                    x1: Math.round(rect.left),
                    y1: Math.round(rect.top),
                    x2: Math.round(rect.right),
                    y2: Math.round(rect.bottom)
                };

                // Find first scrollable parent
                let parent = container.parentElement;
                let scrollableParent = null;

                while (parent && parent !== document.body) {
                    const hasVerticalScroll = parent.scrollHeight > parent.clientHeight;
                    const overflowY = window.getComputedStyle(parent).overflowY;
                    const isScrollable = hasVerticalScroll &&
                                       (overflowY === 'auto' || overflowY === 'scroll');

                    if (isScrollable) {
                        scrollableParent = parent;
                        break;
                    }

                    parent = parent.parentElement;
                }

                // Clip bbox to scrollable parent's visible area
                if (scrollableParent) {
                    const parentRect = scrollableParent.getBoundingClientRect();

                    bbox.x1 = Math.max(bbox.x1, Math.round(parentRect.left));
                    bbox.y1 = Math.max(bbox.y1, Math.round(parentRect.top));
                    bbox.x2 = Math.min(bbox.x2, Math.round(parentRect.right));
                    bbox.y2 = Math.min(bbox.y2, Math.round(parentRect.bottom));
                }

                return bbox;
            }
        """, container_selector)

        if not container_info:
            # Container not found, return empty bbox
            return {
                "bbox": [0, 0, 0, 0],
                "image_bbox": ""
            }

        bbox = [container_info['x1'], container_info['y1'], container_info['x2'], container_info['y2']]

        # Create annotated screenshot with red box around container
        annotated_name = f"{self.grounding.counter:03d}_bbox.png"
        annotated_path = self.grounding.annotations_dir / annotated_name

        temp_path = self.grounding.annotations_dir / f"{self.grounding.counter:03d}_temp.png"
        await self.page.screenshot(path=str(temp_path))

        img = Image.open(temp_path)
        draw = ImageDraw.Draw(img)

        # Draw red box around container
        x1, y1, x2, y2 = bbox
        if x1 < x2 and y1 < y2:  # Valid bbox
            for i in range(3):
                draw.rectangle([x1 + i, y1 + i, x2 - i - 1, y2 - i - 1], outline='red')

            # Draw center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            radius = 5
            draw.ellipse(
                [center_x - radius, center_y - radius,
                 center_x + radius, center_y + radius],
                fill='red',
                outline='white'
            )

        img.save(annotated_path)
        temp_path.unlink()

        # Calculate center point
        center_point = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]

        return {
            "bbox": bbox,
            "center_point": center_point,
            "image_bbox": f"annotations/{annotated_name}"
        }

    # ========== Loop Methods with Ineffectiveness Detection ==========

    async def _perform_page_scroll_until_visible(
        self,
        selector: str
    ) -> List[Dict]:
        """Perform page scroll until element is visible.

        Uses bbox change detection to determine if scroll is ineffective.
        Only records scroll actions that actually change the element's position.
        No screenshots or annotations are generated for ineffective scrolls.

        Args:
            selector: CSS selector for the target element

        Returns:
            List of scroll actions (only effective ones)
        """
        steps_result = []
        iteration = 0

        while iteration < self.max_scroll_iterations:
            # Check if element is visible in viewport
            visibility_info = await self.page.evaluate("""
                (selector) => {
                    const element = document.querySelector(selector);
                    if (!element) return {error: 'Element not found'};

                    const rect = element.getBoundingClientRect();
                    const isVisible = (
                        rect.top >= 0 &&
                        rect.left >= 0 &&
                        rect.bottom <= window.innerHeight &&
                        rect.right <= window.innerWidth
                    );

                    let scrollDirection = null;
                    if (!isVisible) {
                        if (rect.top < 0) {
                            scrollDirection = 'up';
                        } else if (rect.bottom > window.innerHeight) {
                            scrollDirection = 'down';
                        }
                    }

                    return {
                        isVisible: isVisible,
                        scrollDirection: scrollDirection
                    };
                }
            """, selector)

            if 'error' in visibility_info:
                self.logger.error(f"    Error: {visibility_info['error']}")
                break

            if visibility_info['isVisible']:
                if iteration > 0:
                    self.logger.info(f"    Element visible after {iteration} scroll(s)")
                break

            scroll_direction = visibility_info['scrollDirection']
            if not scroll_direction:
                break

            # Perform page scroll with effectiveness check
            # This will only generate screenshots/annotations if scroll is effective
            grounding = await self.perform_page_scroll_if_effective(
                scroll_direction,
                selector
            )

            if grounding is None:
                # Scroll was ineffective, stop trying
                break

            # Scroll was effective, record it
            steps_result.append({
                "container": None,
                "grounding": grounding
            })
            iteration += 1

        if iteration >= self.max_scroll_iterations:
            self.logger.warning(f"    Max iterations reached ({self.max_scroll_iterations})")

        return steps_result

    async def _perform_container_scroll_until_visible(
        self,
        container_selector: str,
        target_selector: str
    ) -> List[Dict]:
        """Perform container scroll until element is visible.

        Uses bbox change detection to determine if scroll is ineffective.
        Only records scroll actions that actually change the element's position.

        Args:
            container_selector: CSS selector for the container
            target_selector: CSS selector for the target element

        Returns:
            List of scroll actions (only effective ones)
        """
        steps_result = []
        iteration = 0
        last_bbox = None

        while iteration < self.max_scroll_iterations:
            # Check if target bbox is within container bbox in viewport
            is_within = await self._check_target_bbox_within_parent_bbox(
                container_selector, target_selector
            )

            if is_within:
                if iteration > 0:
                    self.logger.info(
                        f"    Target bbox within container bbox after {iteration} scroll(s)"
                    )
                break

            # Get current bbox before scroll
            bbox_info = await self.page.evaluate("""
                (selector) => {
                    const element = document.querySelector(selector);
                    if (!element) return null;
                    const rect = element.getBoundingClientRect();
                    return {
                        x1: Math.round(rect.left),
                        y1: Math.round(rect.top),
                        x2: Math.round(rect.right),
                        y2: Math.round(rect.bottom)
                    };
                }
            """, target_selector)

            if not bbox_info:
                self.logger.error(f"    Error: Target element not found")
                break

            bbox_before = (
                bbox_info.get('x1'),
                bbox_info.get('y1'),
                bbox_info.get('x2'),
                bbox_info.get('y2')
            )

            # Determine scroll direction
            scroll_direction = await self._determine_container_scroll_direction(
                container_selector, target_selector
            )

            if not scroll_direction:
                break

            # Perform container scroll with target_selector for grounding validation
            grounding = await self.perform_container_scroll(
                container_selector,
                scroll_direction,
                target_selector=target_selector
            )

            # Check bbox after scroll to verify effectiveness
            bbox_after_info = await self.page.evaluate("""
                (selector) => {
                    const element = document.querySelector(selector);
                    if (!element) return null;
                    const rect = element.getBoundingClientRect();
                    return {
                        x1: Math.round(rect.left),
                        y1: Math.round(rect.top),
                        x2: Math.round(rect.right),
                        y2: Math.round(rect.bottom)
                    };
                }
            """, target_selector)

            if bbox_after_info:
                bbox_after = (
                    bbox_after_info.get('x1'),
                    bbox_after_info.get('y1'),
                    bbox_after_info.get('x2'),
                    bbox_after_info.get('y2')
                )

                # Only record if scroll was effective (bbox changed)
                if bbox_after != bbox_before:
                    steps_result.append({
                        "container": container_selector,
                        "grounding": grounding
                    })
                    last_bbox = bbox_after
                else:
                    self.logger.warning(f"    Container scroll had no effect, not recording")
                    break
            else:
                # Element disappeared after scroll, still record the scroll
                steps_result.append({
                    "container": container_selector,
                    "grounding": grounding
                })

            iteration += 1

        if iteration >= self.max_scroll_iterations:
            self.logger.warning(f"    Max iterations reached ({self.max_scroll_iterations})")

        return steps_result

    # ========== Helper Methods ==========

    async def _find_grounding_container(
        self,
        container_selector: str,
        target_selector: str
    ) -> str:
        """Find the correct container for grounding extraction.

        Ensures that the target element is within the container's scrollable visible area.
        If not, finds the target's direct parent's first scrollable component.

        Args:
            container_selector: Original container selector
            target_selector: Target element selector

        Returns:
            Container selector to use for grounding extraction
        """
        result = await self.page.evaluate("""
            ({containerSelector, targetSelector}) => {
                const container = document.querySelector(containerSelector);
                const target = document.querySelector(targetSelector);

                if (!container || !target) {
                    return {containerSelector: containerSelector};
                }

                // Check if container is scrollable
                const hasVerticalScroll = container.scrollHeight > container.clientHeight;
                const overflowY = window.getComputedStyle(container).overflowY;
                const isScrollable = hasVerticalScroll &&
                                   (overflowY === 'auto' || overflowY === 'scroll');

                if (!isScrollable) {
                    // Container is not scrollable, use it as is
                    return {containerSelector: containerSelector};
                }

                // Check if target is in container's scrollable visible area
                // Need to calculate relative positions considering scroll offset
                const containerRect = container.getBoundingClientRect();
                const targetRect = target.getBoundingClientRect();

                // Calculate target's position relative to container's content area
                const targetRelativeTop = targetRect.top - containerRect.top + container.scrollTop;
                const targetRelativeBottom = targetRelativeTop + targetRect.height;

                // Container's visible area in content coordinates
                const containerVisibleTop = container.scrollTop;
                const containerVisibleBottom = container.scrollTop + container.clientHeight;

                // Check if target is fully visible in container's scrollable area
                const isInVisibleArea = (
                    targetRelativeTop >= containerVisibleTop &&
                    targetRelativeBottom <= containerVisibleBottom
                );

                if (isInVisibleArea) {
                    // Target is already in visible area, use original container
                    return {containerSelector: containerSelector};
                }

                // Target is not in visible area, find target's direct parent's first scrollable component
                let parent = target.parentElement;

                while (parent && parent !== document.body && parent !== container) {
                    const hasVerticalScroll = parent.scrollHeight > parent.clientHeight;
                    const overflowY = window.getComputedStyle(parent).overflowY;
                    const isScrollable = hasVerticalScroll &&
                                       (overflowY === 'auto' || overflowY === 'scroll');

                    if (isScrollable) {
                        // Found a scrollable parent, generate selector
                        let newContainerSelector = null;

                        if (parent.id) {
                            newContainerSelector = '#' + parent.id;
                        } else if (parent.className) {
                            const classes = parent.className.split(' ').filter(c => c.trim());
                            if (classes.length > 0) {
                                newContainerSelector = '.' + classes.map(c => CSS.escape(c)).join('.');
                            }
                        }

                        if (newContainerSelector) {
                            return {containerSelector: newContainerSelector};
                        }
                    }

                    parent = parent.parentElement;
                }

                // No scrollable parent found, use original container
                return {containerSelector: containerSelector};
            }
        """, {'containerSelector': container_selector, 'targetSelector': target_selector})

        new_container = result.get('containerSelector', container_selector)

        if new_container != container_selector:
            self.logger.info(
                f"    Target not in container's scrollable visible area, "
                f"using {new_container} for grounding extraction"
            )

        return new_container

    async def _find_scrollable_parent(self, selector: str) -> Optional[Dict[str, Any]]:
        """Find the scrollable parent container of an element.

        Args:
            selector: CSS selector for the target element

        Returns:
            Dict with containerSelector and other info, or None if no scrollable parent
        """
        scroll_info = await self.page.evaluate("""
            (selector) => {
                const element = document.querySelector(selector);
                if (!element) return null;

                // Helper to check if element is OverlayScrollbars viewport
                function isOsViewport(el) {
                    return (el.classList && el.classList.contains('os-viewport')) ||
                           el.hasAttribute('data-overlayscrollbars-viewport');
                }

                // Helper to get the host container for an os-viewport
                function getOsHost(viewport) {
                    const parent = viewport.parentElement;
                    if (parent && parent.id) {
                        return '#' + parent.id;
                    }
                    return null;
                }

                let parent = element.parentElement;
                while (parent && parent !== document.body) {
                    const hasVerticalScroll = parent.scrollHeight > parent.clientHeight;
                    const overflowY = window.getComputedStyle(parent).overflowY;
                    const isScrollable = hasVerticalScroll &&
                                       (overflowY === 'auto' || overflowY === 'scroll');

                    if (isScrollable) {
                        const rect = parent.getBoundingClientRect();

                        let containerSelector = null;

                        // If this is an OverlayScrollbars viewport, return the host container id
                        if (isOsViewport(parent)) {
                            const hostSelector = getOsHost(parent);
                            if (hostSelector) {
                                containerSelector = hostSelector;
                            }
                        } else if (parent.id) {
                            containerSelector = '#' + parent.id;
                        } else if (parent.className) {
                            const classes = parent.className.split(' ').filter(c => c.trim());
                            if (classes.length > 0) {
                                containerSelector = '.' + classes.map(c => CSS.escape(c)).join('.');
                            }
                        }

                        return {
                            containerSelector: containerSelector,
                            scrollHeight: parent.scrollHeight,
                            clientHeight: parent.clientHeight,
                            scrollTop: parent.scrollTop
                        };
                    }

                    parent = parent.parentElement;
                }

                return null;
            }
        """, selector)

        return scroll_info

    async def _determine_container_scroll_direction(
        self,
        container_selector: str,
        target_selector: str
    ) -> Optional[str]:
        """Determine scroll direction to make target visible in container.

        Args:
            container_selector: CSS selector for the scrollable container
            target_selector: CSS selector for the target element

        Returns:
            'up' or 'down', or None if target is already visible
        """
        result = await self.page.evaluate("""
            ({containerSelector, targetSelector}) => {
                const container = document.querySelector(containerSelector);
                const target = document.querySelector(targetSelector);

                if (!container || !target) {
                    return null;
                }

                // Handle OverlayScrollbars: use viewport element if present
                const osViewport = container.querySelector(':scope > .os-viewport') ||
                                   container.querySelector(':scope > [data-overlayscrollbars-viewport]');
                const scrollContainer = osViewport || container;

                const containerRect = container.getBoundingClientRect();
                const targetRect = target.getBoundingClientRect();

                const targetRelativeTop = targetRect.top - containerRect.top + scrollContainer.scrollTop;
                const targetRelativeBottom = targetRelativeTop + targetRect.height;

                const containerVisibleTop = scrollContainer.scrollTop;
                const containerVisibleBottom = scrollContainer.scrollTop + scrollContainer.clientHeight;

                // Check if target is above visible area
                if (targetRelativeTop < containerVisibleTop) {
                    return 'up';
                }

                // Check if target is below visible area
                if (targetRelativeBottom > containerVisibleBottom) {
                    return 'down';
                }

                // Target is already visible
                return null;
            }
        """, {'containerSelector': container_selector, 'targetSelector': target_selector})

        return result

    async def _check_target_bbox_within_parent_bbox(
        self,
        container_selector: str,
        target_selector: str
    ) -> bool:
        """Check if target element's bbox is within container's bbox in viewport coordinates.

        This checks the actual bbox positions in the current screenshot (viewport coordinates),
        not the logical positions considering scroll offset.

        Args:
            container_selector: CSS selector for the scrollable container
            target_selector: CSS selector for the target element

        Returns:
            True if target bbox is fully within container bbox in viewport coordinates
        """
        result = await self.page.evaluate("""
            ({containerSelector, targetSelector}) => {
                const container = document.querySelector(containerSelector);
                const target = document.querySelector(targetSelector);

                if (!container || !target) {
                    return false;
                }

                // Get bboxes in viewport coordinates
                const containerRect = container.getBoundingClientRect();
                const targetRect = target.getBoundingClientRect();

                // Check if target bbox is fully within container bbox
                const isWithin = (
                    targetRect.left >= containerRect.left &&
                    targetRect.top >= containerRect.top &&
                    targetRect.right <= containerRect.right &&
                    targetRect.bottom <= containerRect.bottom
                );

                return isWithin;
            }
        """, {'containerSelector': container_selector, 'targetSelector': target_selector})

        return result

    async def _find_scroll_container_for_element(self, element_selector: str) -> Dict[str, Any]:
        """Find the scrollable parent container for an element.

        Args:
            element_selector: CSS selector for the target element

        Returns:
            Dict with containerSelector, scrollDirection ('vertical' or 'horizontal'), or error
        """
        result = await self.page.evaluate("""
            (elementSelector) => {
                const element = document.querySelector(elementSelector);
                if (!element) {
                    return { error: 'Element not found' };
                }

                // Helper to check if element is OverlayScrollbars viewport
                function isOsViewport(el) {
                    return (el.classList && el.classList.contains('os-viewport')) ||
                           el.hasAttribute('data-overlayscrollbars-viewport');
                }

                // Helper to get the host container for an os-viewport
                function getOsHost(viewport) {
                    const parent = viewport.parentElement;
                    if (parent && parent.id) {
                        return '#' + parent.id;
                    }
                    return null;
                }

                // Find first scrollable parent
                let parent = element.parentElement;
                while (parent) {
                    const style = window.getComputedStyle(parent);
                    const overflowY = style.overflowY;
                    const overflowX = style.overflowX;

                    const hasVerticalScroll = parent.scrollHeight > parent.clientHeight;
                    const hasHorizontalScroll = parent.scrollWidth > parent.clientWidth;

                    let scrollDirection = null;
                    let scrollDimension = null;

                    // Check vertical scroll first
                    if ((overflowY === 'auto' || overflowY === 'scroll') && hasVerticalScroll) {
                        scrollDirection = 'vertical';
                        scrollDimension = {
                            scrollSize: parent.scrollHeight,
                            clientSize: parent.clientHeight,
                            scrollPos: parent.scrollTop
                        };
                    }
                    // Check horizontal scroll
                    else if ((overflowX === 'auto' || overflowX === 'scroll') && hasHorizontalScroll) {
                        scrollDirection = 'horizontal';
                        scrollDimension = {
                            scrollSize: parent.scrollWidth,
                            clientSize: parent.clientWidth,
                            scrollPos: parent.scrollLeft
                        };
                    }

                    if (scrollDirection) {
                        // Found scrollable container
                        let selector = '';
                        let isOsContainer = false;

                        // If this is an OverlayScrollbars viewport, return the host container id
                        if (isOsViewport(parent)) {
                            isOsContainer = true;
                            const hostSelector = getOsHost(parent);
                            if (hostSelector) {
                                selector = hostSelector;
                            }
                        } else if (parent.id) {
                            selector = '#' + parent.id;
                        } else if (parent.className) {
                            const classes = parent.className.split(' ').filter(c => c);
                            if (classes.length > 0) {
                                selector = '.' + classes.map(c => CSS.escape(c)).join('.');
                            }
                        }

                        return {
                            containerSelector: selector,
                            scrollDirection: scrollDirection,
                            scrollHeight: scrollDimension.scrollSize,
                            clientHeight: scrollDimension.clientSize,
                            isOverlayScrollbars: isOsContainer,
                            rawElementTag: parent.tagName,
                            rawElementId: parent.id || null,
                            rawElementClass: parent.className || null
                        };
                    }
                    parent = parent.parentElement;
                }

                const body = document.body;
                const html = document.documentElement;
                const hasVerticalScroll = html.scrollHeight > html.clientHeight;
                const hasHorizontalScroll = html.scrollWidth > html.clientWidth;

                if (hasVerticalScroll) {
                    return {
                        containerSelector: 'body',
                        scrollDirection: 'vertical',
                        scrollHeight: html.scrollHeight,
                        clientHeight: html.clientHeight
                    };
                } else if (hasHorizontalScroll) {
                    return {
                        containerSelector: 'body',
                        scrollDirection: 'horizontal',
                        scrollHeight: html.scrollWidth,
                        clientHeight: html.clientWidth
                    };
                }

                return { error: 'No scrollable container found' };
            }
        """, element_selector)

        return result

    async def _check_child_visibility_in_container(
        self,
        container_selector: str,
        child_selector: str,
        scroll_direction: str = 'vertical'
    ) -> Dict[str, Any]:
        """Check if child element is fully visible in container.

        Args:
            container_selector: CSS selector for the scrollable container
            child_selector: CSS selector for the child element (full selector)
            scroll_direction: 'vertical' or 'horizontal'

        Returns:
            Dict with visibility info and bbox
        """
        visibility_info = await self.page.evaluate("""
            ({containerSelector, childSelector, scrollDirection}) => {
                const container = document.querySelector(containerSelector);
                const child = document.querySelector(childSelector);

                if (!container || !child) {
                    return {
                        error: !container ? 'Container not found' : 'Child not found'
                    };
                }

                const isBodyContainer = containerSelector === 'body';

                // Handle OverlayScrollbars: use viewport element if present
                const osViewport = !isBodyContainer ? (
                    container.querySelector(':scope > .os-viewport') ||
                    container.querySelector(':scope > [data-overlayscrollbars-viewport]')
                ) : null;
                const scrollElement = isBodyContainer ? document.documentElement : (osViewport || container);

                const childRect = child.getBoundingClientRect();

                let isFullyVisible, scrollDir, currentScroll, maxScroll;

                if (scrollDirection === 'horizontal') {
                    const scrollLeft = isBodyContainer ? window.pageXOffset || scrollElement.scrollLeft : scrollElement.scrollLeft;
                    const clientWidth = isBodyContainer ? window.innerWidth : scrollElement.clientWidth;
                    const scrollWidth = isBodyContainer ? scrollElement.scrollWidth : scrollElement.scrollWidth;

                    let childAbsoluteLeft, childAbsoluteRight;
                    if (isBodyContainer) {
                        childAbsoluteLeft = childRect.left + scrollLeft;
                        childAbsoluteRight = childAbsoluteLeft + childRect.width;
                    } else {
                        const containerRect = container.getBoundingClientRect();
                        childAbsoluteLeft = childRect.left - containerRect.left + scrollLeft;
                        childAbsoluteRight = childAbsoluteLeft + childRect.width;
                    }

                    const containerVisibleLeft = scrollLeft;
                    const containerVisibleRight = scrollLeft + clientWidth;

                    isFullyVisible =
                        childAbsoluteLeft >= containerVisibleLeft &&
                        childAbsoluteRight <= containerVisibleRight;

                    scrollDir = null;
                    if (!isFullyVisible) {
                        if (childAbsoluteLeft < containerVisibleLeft) {
                            scrollDir = 'left';
                        } else if (childAbsoluteRight > containerVisibleRight) {
                            scrollDir = 'right';
                        }
                    }

                    currentScroll = scrollLeft;
                    maxScroll = scrollWidth - clientWidth;
                } else {
                    const scrollTop = isBodyContainer ? window.pageYOffset || scrollElement.scrollTop : scrollElement.scrollTop;
                    const clientHeight = isBodyContainer ? window.innerHeight : scrollElement.clientHeight;
                    const scrollHeight = isBodyContainer ? scrollElement.scrollHeight : scrollElement.scrollHeight;

                    let childAbsoluteTop, childAbsoluteBottom;
                    if (isBodyContainer) {
                        childAbsoluteTop = childRect.top + scrollTop;
                        childAbsoluteBottom = childAbsoluteTop + childRect.height;
                    } else {
                        const containerRect = container.getBoundingClientRect();
                        childAbsoluteTop = childRect.top - containerRect.top + scrollTop;
                        childAbsoluteBottom = childAbsoluteTop + childRect.height;
                    }

                    const containerVisibleTop = scrollTop;
                    const containerVisibleBottom = scrollTop + clientHeight;

                    isFullyVisible =
                        childAbsoluteTop >= containerVisibleTop &&
                        childAbsoluteBottom <= containerVisibleBottom;

                    scrollDir = null;
                    if (!isFullyVisible) {
                        if (childAbsoluteTop < containerVisibleTop) {
                            scrollDir = 'up';
                        } else if (childAbsoluteBottom > containerVisibleBottom) {
                            scrollDir = 'down';
                        }
                    }

                    currentScroll = scrollTop;
                    maxScroll = scrollHeight - clientHeight;
                }

                return {
                    isFullyVisible: isFullyVisible,
                    scrollDirection: scrollDir,
                    currentScroll: currentScroll,
                    maxScroll: maxScroll,
                    childBbox: {
                        x1: Math.round(childRect.left),
                        y1: Math.round(childRect.top),
                        x2: Math.round(childRect.right),
                        y2: Math.round(childRect.bottom)
                    }
                };
            }
        """, {
            'containerSelector': container_selector,
            'childSelector': child_selector,
            'scrollDirection': scroll_direction
        })

        return visibility_info

    async def find_element_in_same_parent(self, selector: str) -> str:
        """Find element under the last operated element.

        Args:
            selector: CSS selector for the target element

        Returns:
            Refined selector if found, otherwise original selector
        """
        if not self.last_selector:
            return selector

        # If selector already starts with last_selector, don't refine
        if selector.startswith(self.last_selector + ' '):
            return selector

        result = await self.page.evaluate("""
            ({lastSelector, targetSelector}) => {
                const lastElement = document.querySelector(lastSelector);
                if (!lastElement) return null;

                const element = lastElement.querySelector(targetSelector);
                if (!element) return null;

                if (element.id) {
                    return '#' + element.id;
                }

                return lastSelector + ' ' + targetSelector;
            }
        """, {
            'lastSelector': self.last_selector,
            'targetSelector': selector
        })

        return result if result else selector

