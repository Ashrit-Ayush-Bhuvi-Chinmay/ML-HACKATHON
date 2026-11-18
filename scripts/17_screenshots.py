"""Screenshots for the report: the Folium maps and the running UI (brief sections 6 and 7).

Starts its own API (port 8010, student-test trips) and UI dev server (port 5180) so it does not
disturb a running demo, drives the UI with Playwright using the installed Microsoft Edge, and
writes PNGs to results/figures/. The API console screenshot on the RECEIVING laptop must still be
taken by hand on the real two-laptop setup.
Usage: python scripts/17_screenshots.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
API, UI = 8010, 5180
API_URL = f"http://127.0.0.1:{API}"
LOG = ROOT / "results" / "report" / "screenshot_run_api_console.txt"


def wait(url: str, seconds: int = 120) -> None:
    for _ in range(seconds):
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except Exception:  # noqa: BLE001
            time.sleep(1)
    raise RuntimeError(f"{url} did not come up")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    py = ROOT / ".venv" / "Scripts" / "python.exe"
    student_test = next(p / "Provided Material" / "Student folder" / "student_test" for p in ROOT.parents
                        if (p / "Provided Material").is_dir())
    api_log = open(LOG, "w", encoding="utf-8")
    api = subprocess.Popen([str(py), "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(API)],
                           cwd=ROOT / "backend", env={**os.environ, "PATHSNAP_EVAL_DIR": str(student_test)},
                           stdout=api_log, stderr=subprocess.STDOUT)
    ui = subprocess.Popen([shutil.which("npm"), "run", "dev", "--", "--port", str(UI), "--strictPort"],
                          cwd=ROOT / "ui", env={**os.environ, "VITE_API_BASE_URL": API_URL},
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait(f"{API_URL}/health")
        wait(f"http://127.0.0.1:{UI}")
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="msedge", headless=True)
            page = browser.new_page(viewport={"width": 1500, "height": 950}, device_scale_factor=1.5)

            for t in ("00000009", "00000052"):  # Folium maps (OpenStreetMap tiles need internet)
                page.goto((ROOT / "results" / "maps" / f"trip_{t}.html").as_uri())
                page.wait_for_timeout(6000)
                page.screenshot(path=str(FIG / f"folium_trip_{t}.png"))

            base = f"http://localhost:{UI}"
            page.goto(base + "/")
            page.evaluate("localStorage.clear()")
            page.goto(base + "/")
            page.get_by_text("Online", exact=False).first.wait_for(timeout=60000)
            page.wait_for_timeout(1500)
            page.screenshot(path=str(FIG / "ui_overview.png"))

            page.goto(base + "/trips")
            page.get_by_text("00000009").first.click()
            page.wait_for_timeout(500)
            run = page.get_by_role("button", name="Run prediction")
            (run.first if run.count() else page.get_by_role("button", name="Run").first).click()
            page.get_by_text("Sequence F1", exact=False).first.wait_for(timeout=120000)
            page.wait_for_timeout(6000)
            page.screenshot(path=str(FIG / "ui_trip_prediction.png"))

            page.get_by_role("button", name="Quick comparison").first.click()
            page.get_by_text("Done", exact=False).first.wait_for(timeout=120000)
            page.wait_for_timeout(4000)
            page.screenshot(path=str(FIG / "ui_quick_comparison.png"))

            page.goto(base + "/analytics")
            page.get_by_text("Model leaderboard").first.wait_for(timeout=60000)
            page.wait_for_timeout(2000)
            page.screenshot(path=str(FIG / "ui_model_analytics.png"), full_page=True)

            # client-side navigation keeps the session, then run every trip through the API
            page.get_by_role("link", name="Evaluation").first.click()
            page.get_by_role("button", name="Run all").first.click()
            page.get_by_text("Not run").first.wait_for(state="detached", timeout=300000)
            page.wait_for_timeout(2000)
            page.screenshot(path=str(FIG / "ui_evaluation.png"))
            browser.close()
    finally:
        ui.terminate()
        api.terminate()
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(ui.pid)], capture_output=True)
        api_log.close()
    print("screenshots written to", FIG)


if __name__ == "__main__":
    main()
