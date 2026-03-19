"""Grounding data extraction for UI elements."""

from pathlib import Path
from typing import Dict, Any
from PIL import Image, ImageDraw


class GroundingExtractor:
    """Extract grounding data (screenshots + bboxes) for UI elements."""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.before_dir = self.output_dir / "before"
        self.after_dir = self.output_dir / "after"
        self.annotations_dir = self.output_dir / "annotations"

        self.before_dir.mkdir(exist_ok=True)
        self.after_dir.mkdir(exist_ok=True)
        self.annotations_dir.mkdir(exist_ok=True)

        self.counter = 0
    
    async def extract_before(self, page) -> str:
        """Capture screenshot before action."""
        img_name = f"{self.counter:03d}.png"
        img_path = self.before_dir / img_name
        await page.screenshot(path=str(img_path))
        return f"before/{img_name}"

    async def extract_after(self, page) -> str:
        """Capture screenshot after action."""
        img_name = f"{self.counter:03d}.png"
        img_path = self.after_dir / img_name
        await page.screenshot(path=str(img_path))
        return f"after/{img_name}"

    async def extract_annotation(self, page, selector: str, viewport_width: int = 1280, viewport_height: int = 720, check_viewport: bool = True, custom_center_point: tuple = None) -> Dict[str, Any]:
        """Extract element annotation with bbox.

        Args:
            page: Playwright page object
            selector: CSS selector
            viewport_width: Viewport width
            viewport_height: Viewport height
            check_viewport: If True, raise exception when bbox is outside viewport
            custom_center_point: Optional (x, y) tuple to override center point for annotation drawing

        Returns:
            Dict with annotation data, or dict with 'outside_viewport' flag if check_viewport=False
        """
        try:
            # Wait for at least one matching element to have valid bbox
            # (instead of waiting for the first element to be visible)
            element = None
            bbox = None
            max_attempts = 30  # 30 * 100ms = 3000ms total

            for attempt in range(max_attempts):
                elements = await page.query_selector_all(selector)

                if not elements:
                    if attempt < max_attempts - 1:
                        await page.wait_for_timeout(100)
                        continue
                    else:
                        raise Exception(f"Element not found: {selector}")

                # Find the first element with valid bbox (width > 0, height > 0)
                for elem in elements:
                    temp_bbox = await elem.bounding_box()
                    if temp_bbox and temp_bbox['width'] > 0 and temp_bbox['height'] > 0:
                        element = elem
                        bbox = temp_bbox
                        break

                if element and bbox:
                    break

                # No visible element found yet, wait and retry
                if attempt < max_attempts - 1:
                    await page.wait_for_timeout(100)

            if not element or not bbox:
                elements_count = len(await page.query_selector_all(selector))
                raise Exception(f"No visible element found for: {selector} (found {elements_count} elements, but all have invalid bbox)")

            # Check if bbox is within viewport
            x1, y1 = int(bbox['x']), int(bbox['y'])
            x2, y2 = x1 + int(bbox['width']), y1 + int(bbox['height'])

            is_outside = x1 < 0 or y1 < 0 or x2 > viewport_width or y2 > viewport_height

            if is_outside and check_viewport:
                raise Exception(f"Element bbox [{x1}, {y1}, {x2}, {y2}] is outside viewport [{viewport_width}x{viewport_height}]")

            tag = await element.evaluate("el => el.tagName.toLowerCase()")
            if tag in ['input', 'textarea']:
                content = await element.input_value()
            else:
                content = (await element.inner_text()).strip()

            annotated_name = f"{self.counter:03d}_bbox.png"
            annotated_path = self.annotations_dir / annotated_name

            temp_path = self.annotations_dir / f"{self.counter:03d}_temp.png"
            await page.screenshot(path=str(temp_path))

            # Use custom center point if provided, otherwise calculate from bbox
            if custom_center_point:
                center_x, center_y = custom_center_point
            else:
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

            self._draw_box(temp_path, annotated_path, bbox, custom_center_point=(center_x, center_y))
            temp_path.unlink()

            return {
                "selector": selector,
                "bbox": [x1, y1, x2, y2],
                "center_point": [center_x, center_y],
                "content": content,
                "image_bbox": f"annotations/{annotated_name}",
                "outside_viewport": False
            }

        except Exception as e:
            current_url = page.url
            raise Exception(f"Page URL: {current_url}\n Grounding extraction failed: {str(e)}")

    async def extract(self, page, selector: str) -> Dict[str, Any]:
        """Complete extraction: before + annotation + after."""
        image_before = await self.extract_before(page)
        annotation_data = await self.extract_annotation(page, selector)
        image_after = await self.extract_after(page)

        result = {
            **annotation_data,
            "image_before": image_before,
            "image_after": image_after
        }

        self.counter += 1
        return result

    async def extract_bbox_image(self, page, bbox: list) -> str:
        """Extract bbox image (screenshot with bbox highlighted).

        Args:
            page: Playwright page object
            bbox: [x1, y1, x2, y2]

        Returns:
            Image path (relative to output_dir)
        """
        img_name = f"{self.counter:03d}_bbox.png"
        img_path = self.annotations_dir / img_name

        # Take screenshot
        temp_path = self.annotations_dir / f"{self.counter:03d}_temp.png"
        await page.screenshot(path=str(temp_path))

        # Draw bbox
        bbox_dict = {
            'x': bbox[0],
            'y': bbox[1],
            'width': bbox[2] - bbox[0],
            'height': bbox[3] - bbox[1]
        }
        self._draw_box(temp_path, img_path, bbox_dict)
        temp_path.unlink()

        return f"annotations/{img_name}"

    def _draw_box(self, src: Path, dst: Path, bbox: Dict, custom_center_point: tuple = None):
        """Draw bounding box and center point on image.

        Args:
            src: Source image path
            dst: Destination image path
            bbox: Bounding box dict with 'x', 'y', 'width', 'height'
            custom_center_point: Optional (x, y) tuple to override center point
        """
        try:
            img = Image.open(src)
            draw = ImageDraw.Draw(img)

            x1, y1 = int(bbox['x']), int(bbox['y'])
            x2, y2 = x1 + int(bbox['width']), y1 + int(bbox['height'])

            # Ensure x2 > x1 and y2 > y1 (PIL requirement)
            if x2 <= x1 or y2 <= y1:
                # Element has zero or negative size, skip drawing
                # Just save the original screenshot
                img.save(dst)
                return

            # Draw red bounding box
            for i in range(3):
                draw.rectangle([x1+i, y1+i, x2-i, y2-i], outline='red')

            # Draw center point (use custom if provided, otherwise calculate from bbox)
            if custom_center_point:
                center_x, center_y = custom_center_point
            else:
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

            radius = 5
            draw.ellipse(
                [center_x - radius, center_y - radius,
                 center_x + radius, center_y + radius],
                fill='red',
                outline='white'
            )

            img.save(dst)
        except Exception:
            pass
