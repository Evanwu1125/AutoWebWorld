#!/usr/bin/env python3
"""Test Playwright proxy and screenshot functionality."""
import asyncio
import os

os.environ["https_proxy"] = "http://127.0.0.1:1087"
os.environ["http_proxy"] = "http://127.0.0.1:1087"

URL = "https://slack.com"
OUTPUT = "/tmp/test_playwright.png"


async def test_no_proxy():
    """Test without proxy — expect timeout if network is blocked."""
    from playwright.async_api import async_playwright
    print("[1] Testing WITHOUT proxy...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.goto(URL, wait_until="load", timeout=15000)
            print(f"    OK — title: {await page.title()}")
        except Exception as e:
            print(f"    FAILED — {type(e).__name__}: {e}")
        finally:
            await browser.close()


async def test_with_proxy():
    """Test with explicit proxy in Playwright launch."""
    from playwright.async_api import async_playwright
    proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
    print(f"[2] Testing WITH proxy: {proxy}")
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            proxy={"server": proxy},
        )
        page = await browser.new_page()
        try:
            await page.goto(URL, wait_until="load", timeout=15000)
            title = await page.title()
            print(f"    OK — title: {title}")
            await page.screenshot(path=OUTPUT)
            print(f"    Screenshot saved: {OUTPUT}")
        except Exception as e:
            print(f"    FAILED — {type(e).__name__}: {e}")
        finally:
            await browser.close()


async def test_with_socks5():
    """Test with SOCKS5 proxy (common for ClashX/V2Ray)."""
    from playwright.async_api import async_playwright
    socks_proxy = "socks5://127.0.0.1:1080"
    print(f"[3] Testing WITH SOCKS5 proxy: {socks_proxy}")
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            proxy={"server": socks_proxy},
        )
        page = await browser.new_page()
        try:
            await page.goto(URL, wait_until="load", timeout=15000)
            title = await page.title()
            print(f"    OK — title: {title}")
        except Exception as e:
            print(f"    FAILED — {type(e).__name__}: {e}")
        finally:
            await browser.close()


async def main():
    print(f"Target URL: {URL}")
    print(f"ENV https_proxy: {os.environ.get('https_proxy')}")
    print(f"ENV HTTP_PROXY: {os.environ.get('HTTP_PROXY')}")
    print()

    await test_no_proxy()
    print()
    await test_with_proxy()
    print()
    await test_with_socks5()


if __name__ == "__main__":
    asyncio.run(main())
