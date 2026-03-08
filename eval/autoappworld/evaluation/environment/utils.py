from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

PORT_PATTERNS = [
    re.compile(r"http://[^\s:]+:(\d{2,5})", re.IGNORECASE),
    re.compile(r"\blocalhost:(\d{2,5})\b", re.IGNORECASE),
    re.compile(r"\bport\s*[:=]?\s*(\d{2,5})\b", re.IGNORECASE),
    re.compile(r"\blistening on\b.*?(\d{2,5})\b", re.IGNORECASE),
]


def find_port_in_line(line: str) -> Optional[str]:
    for pattern in PORT_PATTERNS:
        match = pattern.search(line)
        if match:
            return match.group(1)
    return None


def find_url_in_line(line: str) -> Optional[str]:
    match = re.search(r"(https?://[^\s]+)", line)
    if match:
        return match.group(1)
    return None


def extract_port_from_url(url: str) -> Optional[int]:
    match = re.search(r":(\d{2,5})(?:/|$)", url)
    if match:
        return int(match.group(1))
    return None


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _extract_port_hint(script_value: Optional[str]) -> Optional[str]:
    if not script_value:
        return None
    match = re.search(r"--port(?:=|\s+)(\d{2,5})", script_value)
    if match:
        return match.group(1)
    match = re.search(r"\s-p(?:=|\s+)(\d{2,5})", script_value)
    if match:
        return match.group(1)
    return None


def sample_id(sample: Any) -> str:
    if isinstance(sample, dict):
        if "id" in sample:
            return str(sample["id"])
        if "item_id" in sample:
            return str(sample["item_id"])
    if hasattr(sample, "id"):
        return str(getattr(sample, "id"))
    return "run"


def get_action_type(action: Dict[str, Any]) -> str:
    action_type = action.get("action_type") or action.get("type") or ""
    return str(action_type).strip().lower()


def resolve_action_source(
    runner: Any,
    sample: Any,
    ctx: Dict[str, Any],
) -> Any:
    if "actions" in ctx:
        actions = ctx["actions"]
        if callable(actions):
            return actions
        if isinstance(actions, dict):
            return actions
        if isinstance(actions, Iterable):
            return iter(actions)
        return actions
    if hasattr(runner, "actions"):
        actions = getattr(runner, "actions")
        if callable(actions):
            return actions
        if isinstance(actions, dict):
            return actions
        if isinstance(actions, Iterable):
            return iter(actions)
        return actions
    if hasattr(runner, "iter_actions"):
        return runner.iter_actions(sample, ctx)
    if hasattr(runner, "next_action"):
        return runner
    if hasattr(runner, "act"):
        return runner
    raise ValueError("No action source available for WebEnvironment")


def next_action(
    action_source: Any,
    sample: Any,
    ctx: Dict[str, Any],
    last_result: Any,
) -> Optional[Dict[str, Any]]:
    if callable(action_source):
        return action_source(sample, last_result, ctx)
    if isinstance(action_source, dict):
        return action_source
    if isinstance(action_source, Iterable):
        if not isinstance(action_source, Iterator):
            action_source = iter(action_source)
        try:
            return next(action_source)
        except StopIteration:
            return None
    if hasattr(action_source, "next_action"):
        return action_source.next_action(sample, last_result, ctx)
    if hasattr(action_source, "act"):
        return action_source.act(sample, last_result, ctx)
    return None


def start_web_server(
    web_dir: str,
    timeout: int,
    manager: Optional[str],
    script: Optional[str],
) -> Tuple[subprocess.Popen[str], str]:
    package_json = load_package_json(web_dir)
    scripts = package_json.get("scripts", {})
    if not isinstance(scripts, dict):
        raise ValueError("package.json scripts field is invalid")

    script_name = pick_script(scripts, script)
    script_value = scripts.get(script_name, "")
    port_hint = _extract_port_hint(script_value)
    manager = detect_package_manager(web_dir, manager)
    command = build_command(manager, script_name)

    process = subprocess.Popen(
        command,
        cwd=web_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        creationflags=(
            subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        ),
    )
    port, url = wait_for_port(process, timeout=timeout)
    if port is None and port_hint:
        port = port_hint
    if port is None:
        stop_process(process)
        raise RuntimeError("Failed to detect port from web server output")
    if url is None:
        url = f"http://localhost:{port}/"
    return process, url


def wait_for_port(
    process: subprocess.Popen[str],
    *,
    timeout: int,
) -> Tuple[Optional[str], Optional[str]]:
    output_queue: "queue.Queue[Tuple[str, str]]" = queue.Queue()
    stdout_thread = threading.Thread(
        target=enqueue_output,
        args=(process.stdout, output_queue, "stdout"),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=enqueue_output,
        args=(process.stderr, output_queue, "stderr"),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    start_time = time.time()
    detected_port = None
    detected_url = None

    while time.time() - start_time < timeout:
        if process.poll() is not None and output_queue.empty():
            break
        try:
            _, line = output_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if not line:
            continue
        clean_line = _strip_ansi(line)
        if detected_port is None:
            detected_port = find_port_in_line(clean_line)
        if detected_url is None:
            detected_url = find_url_in_line(clean_line)
        if detected_url and detected_port is None:
            url_port = extract_port_from_url(detected_url)
            if url_port is not None:
                detected_port = str(url_port)
        if detected_port or detected_url:
            return detected_port, detected_url

    return detected_port, detected_url


def enqueue_output(stream, output_queue: queue.Queue, label: str) -> None:
    if stream is None:
        return
    for line in iter(stream.readline, ""):
        output_queue.put((label, line))
    stream.close()


def load_package_json(web_dir: str) -> Dict[str, Any]:
    package_path = os.path.join(web_dir, "package.json")
    if not os.path.exists(package_path):
        raise FileNotFoundError(f"package.json not found in: {web_dir}")
    with open(package_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def pick_script(scripts: Dict[str, Any], script_name: Optional[str]) -> str:
    if script_name:
        if script_name not in scripts:
            raise ValueError(f"script '{script_name}' not found in package.json")
        return script_name
    for candidate in ("dev", "start", "serve"):
        if candidate in scripts:
            return candidate
    raise ValueError("no runnable script found (tried dev/start/serve)")


def detect_package_manager(web_dir: str, manager: Optional[str]) -> str:
    if manager:
        if shutil.which(manager):
            return manager
        raise FileNotFoundError(f"Package manager '{manager}' not found in PATH")

    candidates = []
    if os.path.exists(os.path.join(web_dir, "pnpm-lock.yaml")):
        candidates.append("pnpm")
    if os.path.exists(os.path.join(web_dir, "yarn.lock")):
        candidates.append("yarn")
    if os.path.exists(os.path.join(web_dir, "bun.lockb")) or os.path.exists(
        os.path.join(web_dir, "bun.lock")
    ):
        candidates.append("bun")
    candidates.append("npm")

    for candidate in candidates:
        if shutil.which(candidate):
            return candidate

    raise FileNotFoundError("No package manager found in PATH (pnpm/yarn/bun/npm)")


def _resolve_executable(name: str) -> str:
    executable = name
    if os.name == "nt":
        if not name.lower().endswith(".cmd"):
            candidate = f"{name}.cmd"
            if shutil.which(candidate):
                executable = candidate
    found = shutil.which(executable)
    if found:
        return found
    return executable


def build_command(manager: str, script_name: str) -> list[str]:
    executable = _resolve_executable(manager)
    if manager == "npm":
        return [executable, "run", script_name]
    if manager == "pnpm":
        return [executable, "run", script_name]
    if manager == "yarn":
        return [executable, script_name]
    if manager == "bun":
        return [executable, "run", script_name]
    raise ValueError(f"unknown package manager: {manager}")


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
