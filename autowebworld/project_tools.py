from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class ProjectTools:
    def __init__(self, project_root: Path, fsm_path: Path, npm_client: str = "npm"):
        self.project_root = project_root.resolve()
        self.fsm_path = fsm_path.resolve()
        self.npm_client = npm_client

    def _resolve(self, rel_path: str) -> Path:
        target = (self.project_root / rel_path).resolve()
        if target != self.project_root and self.project_root not in target.parents:
            raise ValueError(f"Path escapes project root: {rel_path}")
        return target

    def list_files(self, rel_dir: str = ".", max_depth: int = 6, max_items: int = 500) -> str:
        root = self._resolve(rel_dir)
        if not root.exists():
            return f"Path not found: {rel_dir}"
        if root.is_file():
            return str(root.relative_to(self.project_root))

        lines: list[str] = []
        for path in sorted(root.rglob("*")):
            rel = path.relative_to(self.project_root)
            if len(rel.parts) > max_depth:
                continue
            suffix = "/" if path.is_dir() else ""
            lines.append(f"{rel}{suffix}")
            if len(lines) >= max_items:
                lines.append("... truncated ...")
                break
        return "\n".join(lines) if lines else "(empty)"

    def read_file(self, rel_path: str, max_chars: int = 200_000) -> str:
        path = self._resolve(rel_path)
        if not path.exists() or not path.is_file():
            return f"File not found: {rel_path}"
        data = path.read_text(encoding="utf-8")
        if len(data) > max_chars:
            return data[:max_chars] + "\n... truncated ..."
        return data

    def write_file(self, rel_path: str, content: str) -> str:
        path = self._resolve(rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"ok: wrote {rel_path} ({len(content)} chars)"

    def run_command(self, cmd: str, timeout_sec: int = 300) -> str:
        blocked_tokens = ["git reset --hard", "git checkout --", "rm -rf /", "sudo "]
        if any(token in cmd for token in blocked_tokens):
            return f"blocked command: {cmd}"

        completed = subprocess.run(
            cmd,
            shell=True,
            cwd=self.project_root,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        output = (
            f"$ {cmd}\n"
            f"[exit_code] {completed.returncode}\n"
            f"[stdout]\n{completed.stdout}\n"
            f"[stderr]\n{completed.stderr}"
        )
        if len(output) > 30_000:
            output = output[:30_000] + "\n... truncated ..."
        return output

    def run_build(self, timeout_sec: int = 480) -> tuple[bool, str]:
        output = self.run_command(f"{self.npm_client} run build", timeout_sec=timeout_sec)
        return ("[exit_code] 0" in output, output)

    def read_fsm_json(self) -> str:
        data = self.fsm_path.read_text(encoding="utf-8")
        if len(data) > 350_000:
            return data[:350_000] + "\n... truncated ..."
        return data

    def copy_fsm_into_project(self, target_name: str = "fsm.json") -> Path:
        dst = self._resolve(target_name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.fsm_path, dst)
        self.fsm_path = dst
        return dst


def ensure_project_from_template_or_vite(
    project_root: Path, npm_client: str, template_dir: Path | None
) -> None:
    package_json = project_root / "package.json"
    if package_json.exists():
        return

    project_root.parent.mkdir(parents=True, exist_ok=True)
    if project_root.exists() and any(project_root.iterdir()):
        raise RuntimeError(
            f"Project dir {project_root} exists and is not empty, but no package.json found."
        )

    if template_dir is not None and template_dir.exists():
        shutil.copytree(template_dir, project_root, dirs_exist_ok=False)
    else:
        subprocess.run(
            [npm_client, "create", "vite@latest", str(project_root), "--", "--template", "react-ts"],
            check=True,
        )
        subprocess.run([npm_client, "install"], cwd=project_root, check=True)
