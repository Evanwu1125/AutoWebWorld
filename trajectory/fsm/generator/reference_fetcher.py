from __future__ import annotations

import asyncio
import base64
import os
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI


@dataclass
class ReferenceInfo:
    theme: str
    url: str
    screenshot_path: str
    image_b64: str


class ReferenceFetcher:
    """Discovers the official website for a given theme, screenshots its homepage,
    and returns the image encoded as base64 for use in LLM prompts."""

    def __init__(self, model: str = "gpt-5", base_url: str = ""):
        self.model = model
        self.base_url = base_url
        self._client: Optional[AsyncOpenAI] = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("Missing OPENAI_API_KEY environment variable")
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=self.base_url if self.base_url else None,
            )
        return self._client

    async def discover_url(self, theme: str) -> str:
        """Ask the LLM for the official homepage URL of the given theme/product name."""
        client = self._get_client()
        prompt = (
            f"What is the official homepage URL of the web product or service named '{theme}'?\n"
            "Reply with ONLY the URL, nothing else. Example: https://www.example.com"
        )
        resp = await client.chat.completions.create(
            model=self.model,
            max_tokens=64,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        url = resp.choices[0].message.content.strip().strip('"').strip("'")
        if not url.startswith("http"):
            url = "https://" + url
        return url

    async def screenshot_homepage(self, url: str, output_path: str) -> str:
        """Navigate to the URL with Playwright, wait for full render, and save a screenshot."""
        from playwright.async_api import async_playwright

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": 1440, "height": 900},
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            )
            page = await context.new_page()
            try:
                await page.goto(url, wait_until="load", timeout=30000)

                # Dismiss common cookie/consent banners
                await page.evaluate("""() => {
                    const keywords = ['accept', 'agree', 'allow', 'ok', 'got it', 'continue', 'close', 'dismiss'];
                    const els = Array.from(document.querySelectorAll('button, a[role="button"], [class*="cookie"] button, [id*="cookie"] button'));
                    for (const el of els) {
                        const text = (el.innerText || el.textContent || '').toLowerCase().trim();
                        if (keywords.some(k => text.includes(k)) && text.length < 40) {
                            el.click();
                            break;
                        }
                    }
                }""")

                # Wait for network to settle
                try:
                    await page.wait_for_load_state("networkidle", timeout=10000)
                except Exception:
                    pass  # networkidle timeout is non-fatal

                # Extra wait for animations and lazy-loaded images
                await asyncio.sleep(2)

                await page.screenshot(path=output_path, full_page=False)
            finally:
                await browser.close()

        return output_path

    async def fetch(self, theme: str, output_dir: str) -> Optional[ReferenceInfo]:
        """Full pipeline: discover URL → confirm with user → screenshot → return ReferenceInfo.

        Returns None if the user skips or any step fails (graceful degradation).
        """
        print(f"\n🔍 Discovering reference URL for theme: '{theme}'...")
        try:
            suggested_url = await self.discover_url(theme)
        except Exception as e:
            print(f"⚠️  URL discovery failed: {e}")
            return None

        print(f"   Suggested URL: {suggested_url}")
        try:
            answer = input("   Confirm? [Y/n/enter custom URL]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n   Skipping reference fetch.")
            return None

        if answer.lower() == "n":
            print("   Skipping reference fetch.")
            return None
        elif answer and answer.lower() not in ("y", "yes"):
            # User typed a custom URL
            url = answer
            if not url.startswith("http"):
                url = "https://" + url
        else:
            url = suggested_url

        screenshot_path = os.path.join(output_dir, "reference", "homepage.png")
        print(f"   Screenshotting {url} ...")
        try:
            await self.screenshot_homepage(url, screenshot_path)
        except Exception as e:
            print(f"⚠️  Screenshot failed: {e}")
            return None

        with open(screenshot_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

        print(f"✅ Reference screenshot saved: {screenshot_path}")
        return ReferenceInfo(
            theme=theme,
            url=url,
            screenshot_path=screenshot_path,
            image_b64=image_b64,
        )
