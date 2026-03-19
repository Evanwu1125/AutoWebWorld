"""Action execution for GUI procedures."""

from typing import Dict


class ActionExecutor:
    """Execute GUI actions on the page."""
    
    def __init__(self, page, logger):
        self.page = page
        self.logger = logger
    
    async def execute(self, step: Dict):
        """Execute a single GUI step."""
        op = step.get('op')
        selector = step.get('selector')
        value = step.get('value')
        text = step.get('text', '')
        key = step.get('key', '')

        if op == 'click':
            await self._click(selector)
        elif op == 'type':
            await self._type(selector, value)
        elif op == 'type_text':
            await self._type_text(selector, text)
        elif op == 'key_press':
            await self._key_press(key)
        elif op == 'hover':
            await self._hover(selector)
        elif op == 'select':
            await self._select(selector, value)
        elif op == 'scroll':
            await self._scroll(selector, step.get('direction'), step.get('amount'))
        else:
            self.logger.warning(f"Unknown operation: {op}")
    
    async def _click(self, selector: str):
        """Click an element.

        For .option-{value} selectors (custom dropdown options), just click directly.
        The Vue component handles the selection logic.

        If multiple elements match, try clicking each one until success.
        """
        elements = await self.page.query_selector_all(selector)

        if not elements:
            self.logger.warning(f"No elements found for selector: {selector}")
            return

        # Try clicking each element until one succeeds
        last_error = None
        for i, element in enumerate(elements):
            try:
                # Check if element has valid bbox
                bbox = await element.bounding_box()
                if not bbox or bbox['width'] == 0 or bbox['height'] == 0:
                    continue

                await element.click(timeout=300)
                await self.page.wait_for_timeout(100)
                self.logger.info(f"✅ Successfully clicked element {i+1}/{len(elements)} for selector: {selector}")
                return
            except Exception as e:
                last_error = e
                self.logger.debug(f"Failed to click element {i+1}/{len(elements)}: {str(e)}")
                continue

        # All elements failed
        if last_error:
            self.logger.warning(f"Failed to click any of {len(elements)} elements for selector: {selector}. Last error: {str(last_error)}")
    
    async def _type(self, selector: str, value: str):
        """Type text into an element.

        If multiple elements match, try typing into each one until success.
        """
        elements = await self.page.query_selector_all(selector)

        if not elements:
            self.logger.warning(f"No elements found for selector: {selector}")
            return

        last_error = None
        for i, element in enumerate(elements):
            try:
                bbox = await element.bounding_box()
                if not bbox or bbox['width'] == 0 or bbox['height'] == 0:
                    continue

                await element.fill(value)
                await self.page.wait_for_timeout(300)
                self.logger.info(f"✅ Successfully typed into element {i+1}/{len(elements)} for selector: {selector}")
                return
            except Exception as e:
                last_error = e
                self.logger.debug(f"Failed to type into element {i+1}/{len(elements)}: {str(e)}")
                continue

        if last_error:
            self.logger.warning(f"Failed to type into any of {len(elements)} elements for selector: {selector}. Last error: {str(last_error)}")

    async def _type_text(self, selector: str, text: str):
        """Type text character by character (for realistic input).

        If multiple elements match, try typing into each one until success.
        """
        elements = await self.page.query_selector_all(selector)

        if not elements:
            self.logger.warning(f"No elements found for selector: {selector}")
            return

        last_error = None
        for i, element in enumerate(elements):
            try:
                bbox = await element.bounding_box()
                if not bbox or bbox['width'] == 0 or bbox['height'] == 0:
                    continue

                # Click to focus
                await element.click(timeout=100)
                await self.page.wait_for_timeout(50)

                # Clear existing content
                await self.page.keyboard.press('Control+A')
                await self.page.keyboard.press('Backspace')

                # Type character by character
                for char in text:
                    await self.page.keyboard.type(char)
                    await self.page.wait_for_timeout(20)

                self.logger.info(f"✅ Successfully typed text into element {i+1}/{len(elements)} for selector: {selector}")
                return
            except Exception as e:
                last_error = e
                self.logger.debug(f"Failed to type text into element {i+1}/{len(elements)}: {str(e)}")
                continue

        if last_error:
            self.logger.warning(f"Failed to type text into any of {len(elements)} elements for selector: {selector}. Last error: {str(last_error)}")

    async def _key_press(self, key: str):
        """Press a keyboard key."""
        await self.page.keyboard.press(key)
        await self.page.wait_for_timeout(300)

    async def _hover(self, selector: str):
        """Hover over an element.

        If multiple elements match, try hovering over each one until success.
        """
        elements = await self.page.query_selector_all(selector)

        if not elements:
            self.logger.warning(f"No elements found for selector: {selector}")
            return

        last_error = None
        for i, element in enumerate(elements):
            try:
                bbox = await element.bounding_box()
                if not bbox or bbox['width'] == 0 or bbox['height'] == 0:
                    continue

                await element.hover()
                await self.page.wait_for_timeout(300)
                self.logger.info(f"✅ Successfully hovered over element {i+1}/{len(elements)} for selector: {selector}")
                return
            except Exception as e:
                last_error = e
                self.logger.debug(f"Failed to hover over element {i+1}/{len(elements)}: {str(e)}")
                continue

        if last_error:
            self.logger.warning(f"Failed to hover over any of {len(elements)} elements for selector: {selector}. Last error: {str(last_error)}")

    async def _select(self, selector: str, value: str):
        """Select an option from a dropdown.

        If multiple elements match, try selecting from each one until success.
        """
        elements = await self.page.query_selector_all(selector)

        if not elements:
            self.logger.warning(f"No elements found for selector: {selector}")
            return

        last_error = None
        for i, element in enumerate(elements):
            try:
                bbox = await element.bounding_box()
                if not bbox or bbox['width'] == 0 or bbox['height'] == 0:
                    continue

                await element.select_option(value)
                await self.page.wait_for_timeout(300)
                self.logger.info(f"✅ Successfully selected option in element {i+1}/{len(elements)} for selector: {selector}")
                return
            except Exception as e:
                last_error = e
                self.logger.debug(f"Failed to select option in element {i+1}/{len(elements)}: {str(e)}")
                continue

        if last_error:
            self.logger.warning(f"Failed to select option in any of {len(elements)} elements for selector: {selector}. Last error: {str(last_error)}")

    async def _scroll(self, selector: str, direction: str, amount: int):
        """Scroll an element.

        If multiple elements match, try scrolling each one until success.
        """
        elements = await self.page.query_selector_all(selector)

        if not elements:
            self.logger.warning(f"No elements found for selector: {selector}")
            return

        last_error = None
        for i, element in enumerate(elements):
            try:
                bbox = await element.bounding_box()
                if not bbox or bbox['width'] == 0 or bbox['height'] == 0:
                    continue

                if direction == 'down':
                    await element.evaluate(f"el => el.scrollTop += {amount}")
                elif direction == 'up':
                    await element.evaluate(f"el => el.scrollTop -= {amount}")
                await self.page.wait_for_timeout(300)
                self.logger.info(f"✅ Successfully scrolled element {i+1}/{len(elements)} for selector: {selector}")
                return
            except Exception as e:
                last_error = e
                self.logger.debug(f"Failed to scroll element {i+1}/{len(elements)}: {str(e)}")
                continue

        if last_error:
            self.logger.warning(f"Failed to scroll any of {len(elements)} elements for selector: {selector}. Last error: {str(last_error)}")
    
    async def _drag(self, selector: str, offset_x: int, offset_y: int):
        """Drag an element."""
        element = await self.page.query_selector(selector)
        if element:
            box = await element.bounding_box()
            if box:
                start_x = box['x'] + box['width'] / 2
                start_y = box['y'] + box['height'] / 2
                await self.page.mouse.move(start_x, start_y)
                await self.page.mouse.down()
                await self.page.mouse.move(start_x + offset_x, start_y + offset_y)
                await self.page.mouse.up()
                await self.page.wait_for_timeout(300)

