#!/usr/bin/env python3
"""Minimal Playwright test — is Chromium working at all?"""
import asyncio
import os

os.environ["https_proxy"] = "http://127.0.0.1:1087"
os.environ["http_proxy"] = "http://127.0.0.1:1087"


async def main():
    from playwright.async_api import async_playwright

    # Test 1: simple site (no proxy needed usually)
    print("[1] Testing http://example.com (no proxy)...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.goto("http://example.com", timeout=10000)
            print(f"    OK — title: {await page.title()}")
        except Exception as e:
            print(f"    FAILED — {e}")
        await browser.close()

    # Test 2: same with proxy
    print("[2] Testing https://www.google.com (with proxy)...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            proxy={"server": "http://127.0.0.1:1087"},
        )
        page = await browser.new_page()
        try:
            await page.goto("https://www.google.com", timeout=10000)
            print(f"    OK — title: {await page.title()}")
        except Exception as e:
            print(f"    FAILED — {e}")
        await browser.close()

    # Test 3: check if Chromium binary exists
    print("[3] Checking Playwright install...")
    async with async_playwright() as p:
        path = p.chromium.executable_path if hasattr(p.chromium, 'executable_path') else 'unknown'
        print(f"    Chromium path: {path}")


if __name__ == "__main__":
    asyncio.run(main())
