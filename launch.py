"""PathSnap launcher - one command for everything:   python launch.py

  1) Server  (laptop 1): runs the model + API, shows the address laptop 2 must use
  2) UI      (laptop 2): asks for that address, checks it, starts the UI, opens the browser
  3) Both    (one laptop)
  4) Setup / repair only

Standard library only. The first run sets up whatever the chosen role needs. Choices are
remembered in .launcher.json (git-ignored).
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE = ROOT / ".launcher.json"
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
UI_DIR = ROOT / "ui"
API_PORT = 8000
UI_PORT = 5173
IS_WINDOWS = os.name == "nt"


# ----------------------------------------------------------------------------- small helpers
def say(msg: str = "") -> None:
    print(msg, flush=True)


def ask(prompt: str, default: str = "") -> str:
    shown = f" [{default}]" if default else ""
    answer = input(f"{prompt}{shown}: ").replace("﻿", "").strip()  # piped input may carry a BOM
    return answer or default


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def run(cmd: list[str], cwd: Path | None = None) -> None:
    say("  > " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


_POPUP = (
    "import sys, tkinter; from tkinter import messagebox; r = tkinter.Tk(); r.withdraw(); "
    "r.attributes('-topmost', True); messagebox.showinfo(sys.argv[1], sys.argv[2], parent=r); r.destroy()"
)


def popup(title: str, text: str) -> None:
    """A normal dialog in its own small Python process, so it never blocks the server."""
    try:
        subprocess.Popen([sys.executable, "-c", _POPUP, title, text])
    except OSError:
        pass  # the console shows the same text


def copy_to_clipboard(text: str) -> bool:
    if not IS_WINDOWS:
        return False
    try:
        subprocess.run(["clip"], input=text.encode("utf-16-le"), check=True)  # Windows' own clipboard tool
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def lan_ip() -> str | None:
    """IPv4 of the interface that carries normal traffic (Wi-Fi/Ethernet, not VPN/VM adapters).
    A UDP 'connect' sends no packet; it only asks the OS which interface it would use."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        return None if ip.startswith(("127.", "169.254.")) else ip
    except OSError:
        return None
    finally:
        s.close()


def api_health(url: str, timeout: float = 4.0) -> dict | None:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - any failure means "not reachable yet"
        return None


# ----------------------------------------------------------------------------- setup
def find_python312() -> list[str]:
    if sys.version_info[:2] == (3, 12):
        return [sys.executable]
    if shutil.which("py"):
        probe = subprocess.run(["py", "-3.12", "-c", "print(1)"], capture_output=True)
        if probe.returncode == 0:
            return ["py", "-3.12"]
    raise SystemExit("Python 3.12 is needed for the server. Install it from python.org, then re-run.")


def setup_api() -> None:
    say("\n=== Setting up the server (first run, a few minutes) ===")
    if not VENV_PY.exists():
        uv = shutil.which("uv")
        if uv:
            run([uv, "venv", "--python", "3.12", ROOT / ".venv"])
        else:
            run(find_python312() + ["-m", "venv", ROOT / ".venv"])
    uv = shutil.which("uv")
    pip = [uv, "pip", "install", "--python", VENV_PY] if uv else [VENV_PY, "-m", "pip", "install"]
    run(pip + ["-r", ROOT / "requirements.txt"])
    run(pip + ["-e", ROOT])
    run(pip + ["-r", ROOT / "backend" / "requirements.txt"])
    if ask("Install PyTorch (CPU) so the two neural models in the leaderboard can run? (y/n)", "y").lower().startswith("y"):
        run(pip + ["torch", "--index-url", "https://download.pytorch.org/whl/cpu"])
    run([VENV_PY, "-c", "from pathsnap.pipeline import load_bundle; print('model OK:', load_bundle()['model_name'])"])


def npm_cmd() -> str:
    npm = shutil.which("npm")
    if not npm:
        raise SystemExit("Node.js is needed for the UI. Install the LTS from nodejs.org, then re-run.")
    return npm


def setup_ui() -> None:
    say("\n=== Setting up the UI (first run, a few minutes) ===")
    npm = npm_cmd()
    run([npm, "install"], cwd=UI_DIR)
    build = subprocess.run([npm, "run", "build"], cwd=UI_DIR, capture_output=True, text=True)
    if build.returncode != 0:
        # npm sometimes skips the optional Windows native binaries of rolldown / oxlint
        say("  build failed once; adding the Windows native packages and retrying")
        def version(pkg: str) -> str:
            return json.loads((UI_DIR / "node_modules" / pkg / "package.json").read_text())["version"]
        run([npm, "install", "--no-save", f"@rolldown/binding-win32-x64-msvc@{version('rolldown')}",
             f"@oxlint/binding-win32-x64-msvc@{version('oxlint')}"], cwd=UI_DIR)
        run([npm, "run", "build"], cwd=UI_DIR)
    say("  UI build OK")


def ensure(role: str) -> None:
    if role == "api" and not VENV_PY.exists():
        setup_api()
    if role == "ui" and not (UI_DIR / "node_modules").exists():
        setup_ui()


# ----------------------------------------------------------------------------- trip folders
def is_trip_set(d: Path) -> bool:
    """A folder of trips: sub-folders <id>/<id>.track, or <id>.track files directly - and not a
    single trip folder itself."""
    if not d.is_dir() or (d / f"{d.name}.track").exists():
        return False
    if any((s / f"{s.name}.track").exists() for s in d.iterdir() if s.is_dir()):
        return True
    return any(d.glob("*.track"))


def known_trip_sets() -> list[Path]:
    found = []
    for parent in [ROOT, *ROOT.parents][:5]:
        pm = parent / "Provided Material"
        if pm.is_dir():
            for d in sorted(pm.rglob("*")):
                if d.is_dir() and "train" not in d.parts[-2:] and len(d.relative_to(pm).parts) <= 3 and is_trip_set(d):
                    found.append(d)
    return list(dict.fromkeys(found))


def choose_trips(state: dict) -> Path:
    options = []
    if state.get("eval_dir") and is_trip_set(Path(state["eval_dir"])):
        options.append(Path(state["eval_dir"]))
    options += [p for p in known_trip_sets() if p not in options]
    say("\nWhich trips should the server use?")
    for i, p in enumerate(options, 1):
        say(f"  {i}) {p}")
    say("  b) Browse for a folder...")
    choice = ask("Choose", "1" if options else "b")
    if choice.lower() == "b":
        try:
            import tkinter
            from tkinter import filedialog
            root = tkinter.Tk()
            root.withdraw()
            choice = filedialog.askdirectory(title="Folder with the trip sub-folders") or ""
            root.destroy()
        except Exception:  # noqa: BLE001
            choice = ask("Folder path")
    path = options[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(options) else Path(choice)
    if not is_trip_set(path):
        raise SystemExit(f"No trips found in: {path}")
    state["eval_dir"] = str(path)
    save_state(state)
    return path


# ----------------------------------------------------------------------------- roles
def firewall_hint() -> str:
    if not IS_WINDOWS:
        return ""
    out = subprocess.run(["netsh", "advfirewall", "firewall", "show", "rule", "name=PathSnap API"],
                         capture_output=True, text=True)
    if "PathSnap API" in out.stdout:
        return ""
    return ("Windows may block laptop 2. Once, in a PowerShell opened with 'Run as administrator':\n"
            f"  New-NetFirewallRule -DisplayName \"PathSnap API\" -Direction Inbound -Protocol TCP "
            f"-LocalPort {API_PORT} -Action Allow")


def start_server(state: dict, new_window: bool = False) -> subprocess.Popen:
    ensure("api")
    trips = choose_trips(state)
    env = {**os.environ, "PATHSNAP_EVAL_DIR": str(trips)}
    cmd = [str(VENV_PY), "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", str(API_PORT)]
    flags = subprocess.CREATE_NEW_CONSOLE if (new_window and IS_WINDOWS) else 0
    proc = subprocess.Popen(cmd, cwd=ROOT / "backend", env=env, creationflags=flags)
    say(f"\nStarting the API with trips from {trips} ...")
    for _ in range(90):
        if api_health(f"http://127.0.0.1:{API_PORT}", timeout=2):
            break
        if proc.poll() is not None:
            raise SystemExit("The API stopped during startup - see the messages above.")
        time.sleep(1)
    return proc


def run_server(state: dict) -> None:
    proc = start_server(state)
    ip = lan_ip()
    url = f"http://{ip}:{API_PORT}" if ip else None
    fw = firewall_hint()
    say("\n" + "=" * 64)
    if url:
        copied = copy_to_clipboard(url)
        say(f" Laptop 2: choose 2) UI and enter   {url}")
        say("   (copied to the clipboard)" if copied else "")
    else:
        say(" No network address found: connect to Wi-Fi or a phone hotspot.")
    say(f" On this laptop: http://127.0.0.1:{API_PORT}")
    if fw:
        say(" " + fw.replace("\n", "\n "))
    say(" Leave this window open: the evaluator watches this console. Ctrl+C stops the server.")
    say("=" * 64 + "\n")
    if url:
        text = (f"On laptop 2, run  python launch.py, choose 2) UI and enter:\n\n    {url}\n\n"
                "(already copied to the clipboard - paste it into your chat)\n\n"
                "Both laptops must be on the same network (a phone hotspot works best).")
        if fw:
            text += "\n\n" + fw
        popup("PathSnap server address", text)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()


def open_browser_when_ready() -> None:
    for _ in range(90):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{UI_PORT}", timeout=2)
            webbrowser.open(f"http://localhost:{UI_PORT}")
            return
        except Exception:  # noqa: BLE001
            time.sleep(1)


def run_ui(state: dict, api_url: str | None = None) -> None:
    ensure("ui")
    if not api_url:
        say("\nLaptop 1 shows the address in a popup when its server starts.")
        api_url = ask("API address (just the IP is fine)", state.get("api_url", f"http://127.0.0.1:{API_PORT}"))
        if not api_url.startswith(("http://", "https://")):
            api_url = "http://" + api_url
        if api_url.count(":") < 2:
            api_url = f"{api_url}:{API_PORT}"
    api_url = api_url.rstrip("/")
    state["api_url"] = api_url
    save_state(state)
    say(f"Checking {api_url} ...")
    health = None
    for _ in range(15):
        health = api_health(api_url)
        if health:
            break
        time.sleep(2)
    if health:
        say(f"API online: model {health.get('model')}, {health.get('n_trips')} trips")
    else:
        say("The API does not answer yet; the UI starts anyway and keeps retrying. Check that:\n"
            "  - the server is running on laptop 1 and this is the address it shows\n"
            "  - both laptops are on the same network (a phone hotspot is the most reliable)\n"
            "  - laptop 1 allowed port 8000 in its firewall (its server window shows how)")
    threading.Thread(target=open_browser_when_ready, daemon=True).start()
    say(f"Starting the UI - the browser opens by itself (http://localhost:{UI_PORT}). Ctrl+C stops it.\n")
    env = {**os.environ, "VITE_API_BASE_URL": api_url}
    try:
        subprocess.run([npm_cmd(), "run", "dev", "--", "--port", str(UI_PORT)], cwd=UI_DIR, env=env)
    except KeyboardInterrupt:
        pass


def main() -> None:
    state = load_state()
    say("\n  PathSnap")
    say("  1) Server  - laptop 1: runs the model and the API")
    say("  2) UI      - laptop 2: the evaluator interface")
    say("  3) Both    - everything on this laptop")
    say("  4) Setup   - install / repair only")
    choice = ask("\nChoose", "3")
    if choice == "1":
        run_server(state)
    elif choice == "2":
        run_ui(state)
    elif choice == "3":
        proc = start_server(state, new_window=True)
        say("API running in a separate window (leave it open).")
        try:
            run_ui(state, f"http://127.0.0.1:{API_PORT}")
        finally:
            proc.terminate()
    elif choice == "4":
        setup_api()
        setup_ui()
    else:
        raise SystemExit(f"Unknown choice: {choice}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"\nA setup step failed: {exc}")
