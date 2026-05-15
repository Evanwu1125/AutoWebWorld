from __future__ import annotations

import asyncio
import base64
import mimetypes
import os
from pathlib import Path


# Re-usable viewport / UA defaults (mirrors reference_fetcher.py)
DEFAULT_VIEWPORT_WIDTH = 1440
DEFAULT_VIEWPORT_HEIGHT = 900
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# JS snippet to dismiss common cookie / consent banners
_DISMISS_BANNERS_JS = """() => {
    const keywords = ['accept', 'agree', 'allow', 'ok', 'got it', 'continue', 'close', 'dismiss'];
    const els = Array.from(document.querySelectorAll(
        'button, a[role="button"], [class*="cookie"] button, [id*="cookie"] button'
    ));
    for (const el of els) {
        const text = (el.innerText || el.textContent || '').toLowerCase().trim();
        if (keywords.some(k => text.includes(k)) && text.length < 40) {
            el.click();
            break;
        }
    }
}"""


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

async def _async_screenshot(
    url: str,
    output_path: str,
    *,
    viewport_width: int = DEFAULT_VIEWPORT_WIDTH,
    viewport_height: int = DEFAULT_VIEWPORT_HEIGHT,
) -> str:
    """Take a screenshot of *url* using Playwright + stealth (async implementation)."""
    from playwright.async_api import async_playwright
    from playwright_stealth import Stealth

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    proxy_url = (
        os.environ.get("HTTPS_PROXY")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("ALL_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("http_proxy")
        or os.environ.get("all_proxy")
    )
    launch_kw: dict = {
        "headless": True,
        "args": [
            "--disable-blink-features=AutomationControlled",
        ],
    }
    if proxy_url:
        launch_kw["proxy"] = {"server": proxy_url}

    stealth = Stealth()

    async with async_playwright() as p:
        browser = await p.chromium.launch(**launch_kw)
        context = await browser.new_context(
            viewport={"width": viewport_width, "height": viewport_height},
            user_agent=DEFAULT_USER_AGENT,
        )
        await stealth.apply_stealth_async(context)
        page = await context.new_page()
        try:
            await page.goto(url, wait_until="load", timeout=30000)

            # Dismiss cookie / consent banners
            await page.evaluate(_DISMISS_BANNERS_JS)

            # Wait for network to settle
            try:
                await page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                pass  # non-fatal

            # Wait for Cloudflare challenge to resolve (if any)
            for _ in range(3):
                title = await page.title()
                if "just a moment" in title.lower() or "checking" in title.lower() or "验证" in title:
                    await asyncio.sleep(5)
                else:
                    break

            # Extra wait for animations / lazy-loaded images
            await asyncio.sleep(2)

            await page.screenshot(path=output_path, full_page=False)
        finally:
            await browser.close()

    return output_path


def screenshot_url(
    url: str,
    output_path: str,
    *,
    viewport_width: int = DEFAULT_VIEWPORT_WIDTH,
    viewport_height: int = DEFAULT_VIEWPORT_HEIGHT,
) -> str:
    """Synchronous wrapper: screenshot a URL and return the saved path."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already inside an event loop — run in a new thread to avoid deadlock
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                asyncio.run,
                _async_screenshot(url, output_path,
                                  viewport_width=viewport_width,
                                  viewport_height=viewport_height),
            ).result()
    else:
        return asyncio.run(
            _async_screenshot(url, output_path,
                              viewport_width=viewport_width,
                              viewport_height=viewport_height),
        )


def load_image_as_base64(image_path: str | Path) -> str:
    """Read a local image file and return its base64-encoded content."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ---------------------------------------------------------------------------
# browser-use based screenshot
# ---------------------------------------------------------------------------

async def _async_browser_use_screenshot(url: str, output_path: str) -> str:
    """Take a screenshot using browser-use. Returns base64 image string."""
    from browser_use import BrowserSession, BrowserProfile

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    proxy_url = (
        os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
        or os.environ.get("https_proxy") or os.environ.get("http_proxy")
    )
    profile_kw: dict = {"headless": True}
    if proxy_url:
        from browser_use.browser.profile import ProxySettings
        profile_kw["proxy"] = ProxySettings(server=proxy_url)

    profile = BrowserProfile(**profile_kw)
    session = BrowserSession(browser_profile=profile)
    await session.start()
    try:
        await session.navigate_to(url)
        await asyncio.sleep(3)
        screenshot_bytes = await session.take_screenshot(path=output_path)
        return base64.b64encode(screenshot_bytes).decode()
    finally:
        await session.stop()


def screenshot_url_browser_use(url: str, output_path: str) -> str:
    """Synchronous wrapper: screenshot a URL via browser-use. Returns base64 string."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, _async_browser_use_screenshot(url, output_path)).result()
    else:
        return asyncio.run(_async_browser_use_screenshot(url, output_path))


def _guess_media_type(image_path: str | Path) -> str:
    """Return MIME type for the image, defaulting to image/png."""
    mt, _ = mimetypes.guess_type(str(image_path))
    if mt and mt.startswith("image/"):
        return mt
    return "image/png"


def create_image_content_block(image_path: str | Path) -> dict:
    """Build an Anthropic API image content block from a local file.

    Returns::

        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "<base64 string>"
            }
        }
    """
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": _guess_media_type(image_path),
            "data": load_image_as_base64(image_path),
        },
    }


# ---------------------------------------------------------------------------
# CLI demo — run with: python -m src.agent.browser [url] [output_path]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Browser tool demo: screenshot a URL or view a local image.",
    )
    sub = parser.add_subparsers(dest="command")

    # --- screenshot (playwright) ---
    p_shot = sub.add_parser("screenshot", help="Screenshot a URL via Playwright")
    p_shot.add_argument("url", help="URL to screenshot, e.g. https://github.com")
    p_shot.add_argument(
        "-o", "--output", default="/tmp/browser_demo.png",
        help="Output path (default: /tmp/browser_demo.png)",
    )
    p_shot.add_argument("--width", type=int, default=DEFAULT_VIEWPORT_WIDTH)
    p_shot.add_argument("--height", type=int, default=DEFAULT_VIEWPORT_HEIGHT)

    # --- screenshot-bu (browser-use) ---
    p_bu = sub.add_parser("screenshot-bu", help="Screenshot a URL via browser-use")
    p_bu.add_argument("url", help="URL to screenshot, e.g. https://github.com")
    p_bu.add_argument(
        "-o", "--output", default="/tmp/browser_use_demo.png",
        help="Output path (default: /tmp/browser_use_demo.png)",
    )

    # --- view ---
    p_view = sub.add_parser("view", help="Load a local image and show its info")
    p_view.add_argument("path", help="Path to a local image file")

    args = parser.parse_args()

    if args.command == "screenshot":
        print(f"[Playwright] Launching headless Chromium ...")
        print(f"  URL:      {args.url}")
        print(f"  Viewport: {args.width}x{args.height}")
        print(f"  Output:   {args.output}")

        saved = screenshot_url(
            args.url, args.output,
            viewport_width=args.width, viewport_height=args.height,
        )
        size_kb = os.path.getsize(saved) / 1024
        print(f"  Screenshot saved: {saved} ({size_kb:.1f} KB)")

    elif args.command == "screenshot-bu":
        print(f"[browser-use] Launching headless browser ...")
        print(f"  URL:    {args.url}")
        print(f"  Output: {args.output}")

        b64 = screenshot_url_browser_use(args.url, args.output)
        size_kb = os.path.getsize(args.output) / 1024
        print(f"  Screenshot saved: {args.output} ({size_kb:.1f} KB)")
        print(f"  base64 length: {len(b64)} chars")

    elif args.command == "view":
        p = Path(args.path)
        if not p.exists():
            print(f"Error: File not found: {args.path}")
            raise SystemExit(1)

        size_kb = p.stat().st_size / 1024
        print(f"[1/2] Loading image: {p}  ({size_kb:.1f} KB)")

        block = create_image_content_block(p)
        b64_len = len(block["source"]["data"])
        print(f"[2/2] Image content block ready:")
        print(f"       media_type: {block['source']['media_type']}")
        print(f"       base64 length: {b64_len} chars")
        print(f"\n  This block would be sent inside a tool_result to Claude,")
        print(f"  so the model can 'see' the image as visual context.")

    else:
        parser.print_help()
        print("\nExamples:")
        print("  python -m src.agent.browser screenshot https://example.com -o /tmp/pw.png")
        print("  python -m src.agent.browser screenshot-bu https://example.com -o /tmp/bu.png")
        print("  python -m src.agent.browser view /tmp/pw.png")
