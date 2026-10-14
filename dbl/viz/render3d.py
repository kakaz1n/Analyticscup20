"""Static PNG of the 3D viewer for the README (needs `playwright` + Chromium; dev-only)."""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path


def snapshot(fig, path: Path, width: int = 900, height: int = 620) -> Path:
    from playwright.async_api import async_playwright

    fig = fig.update_layout(sliders=[], updatemenus=[], width=width, height=height)
    with tempfile.TemporaryDirectory() as d:
        html = Path(d) / "scene.html"
        fig.write_html(html, include_plotlyjs=True, config={"displayModeBar": False})

        async def run():
            async with async_playwright() as p:
                b = await p.chromium.launch(args=["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"])
                pg = await b.new_page(viewport={"width": width, "height": height}, device_scale_factor=2)
                await pg.goto(html.as_uri())
                await pg.wait_for_timeout(4000)
                await pg.screenshot(path=str(path))
                await b.close()

        asyncio.run(run())
    return path
