"""Tool functions available to the autonomous research agent."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_project_path(path: str) -> Path:
    """Resolve a path inside the project root and reject escapes."""
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Path escapes project root: {path}") from exc
    return resolved


def run_python(code: str, timeout: int = 300) -> str:
    """Run Python code from ``experiments/temp_script.py`` and return output."""
    try:
        experiments_dir = PROJECT_ROOT / "experiments"
        experiments_dir.mkdir(parents=True, exist_ok=True)
        script_path = experiments_dir / "temp_script.py"
        script_path.write_text(code, encoding="utf-8")

        completed = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        marker = "[ERROR]\n" if completed.returncode != 0 else ""
        return (
            f"{marker}Script: {script_path}\n"
            f"Exit code: {completed.returncode}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )
    except subprocess.TimeoutExpired:
        return f"[TIMEOUT after {timeout}s]"
    except Exception as exc:
        return f"[ERROR] run_python failed: {exc}"


def file_io(operation: str, path: str, content: str | None = None) -> str:
    """Read, write, or list project files."""
    try:
        target = _resolve_project_path(path)
        if operation == "read":
            if not target.is_file():
                return f"[ERROR] File does not exist: {target}"
            return target.read_text(encoding="utf-8", errors="replace")

        if operation == "write":
            if content is None:
                return "[ERROR] content is required for write"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"[OK] Wrote {target}"

        if operation == "list_dir":
            if not target.exists():
                return f"[ERROR] Directory does not exist: {target}"
            if not target.is_dir():
                return f"[ERROR] Not a directory: {target}"
            entries = []
            for item in sorted(target.iterdir(), key=lambda value: value.name.lower()):
                suffix = "/" if item.is_dir() else ""
                entries.append(f"{item.name}{suffix}")
            return "\n".join(entries)

        return f"[ERROR] Unsupported operation: {operation}"
    except Exception as exc:
        return f"[ERROR] file_io failed: {exc}"


def log_experiment(
    task_id: int,
    version: str,
    hypothesis: str,
    code_changes: str,
    metrics: dict[str, Any],
    conclusion: str,
) -> str:
    """Append an experiment log and update the experiment index."""
    try:
        timestamp = datetime.now(timezone.utc).isoformat()
        submission_dir = PROJECT_ROOT / "submission"
        experiments_dir = PROJECT_ROOT / "experiments"
        submission_dir.mkdir(parents=True, exist_ok=True)
        experiments_dir.mkdir(parents=True, exist_ok=True)

        log_path = submission_dir / f"task{task_id}_logs.log"
        record = (
            f"\n## {timestamp} - {version}\n"
            f"Task: {task_id}\n"
            f"Hypothesis: {hypothesis}\n"
            f"Code changes: {code_changes}\n"
            f"Metrics: {json.dumps(metrics, ensure_ascii=False, sort_keys=True)}\n"
            f"Conclusion: {conclusion}\n"
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(record)

        index_path = experiments_dir / "experiment_index.json"
        if index_path.exists():
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                index = {}
        else:
            index = {}
        index[version] = {
            "timestamp": timestamp,
            "task_id": task_id,
            "hypothesis": hypothesis,
            "code_changes": code_changes,
            "metrics": metrics,
            "conclusion": conclusion,
        }
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        return f"[OK] Logged experiment to {log_path} and {index_path}"
    except Exception as exc:
        return f"[ERROR] log_experiment failed: {exc}"


def _find_latest(patterns: list[str]) -> Path | None:
    """Find the newest file matching any pattern below the project root."""
    matches: list[Path] = []
    ignored_parts = {"__pycache__", ".git", ".idea", ".claude"}
    for pattern in patterns:
        for path in PROJECT_ROOT.rglob(pattern):
            if any(part in ignored_parts for part in path.parts):
                continue
            if path.is_file():
                matches.append(path)
    if not matches:
        return None
    return max(matches, key=lambda value: value.stat().st_mtime)


def submit_final(task_id: int) -> str:
    """Package final deliverables into ``submission.zip``."""
    try:
        submission_dir = PROJECT_ROOT / "submission"
        submission_dir.mkdir(parents=True, exist_ok=True)
        zip_path = PROJECT_ROOT / "submission.zip"

        pred_file = _find_latest(
            [
                f"task{task_id}_pred.hdf5",
                f"task{task_id}*pred*.hdf5",
                "*pred*.hdf5",
            ]
        )
        if pred_file is None:
            return "[ERROR] No prediction HDF5 file found to package."

        time_file = _find_latest([f"task{task_id}_time.csv", f"task{task_id}*time*.csv", "*time*.csv"])
        log_file = submission_dir / f"task{task_id}_logs.log"
        team_name = os.environ.get("TEAM_NAME", "autonomous_agent")
        warnings: list[str] = []

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "submission.json",
                json.dumps({"team_name": team_name, "task_id": task_id}, ensure_ascii=False, indent=2),
            )
            archive.write(pred_file, f"task{task_id}_pred.hdf5")
            if time_file is not None:
                archive.write(time_file, f"task{task_id}_time.csv")
            else:
                warnings.append("No time CSV file found.")
            if log_file.exists():
                archive.write(log_file, f"task{task_id}_logs.log")
            else:
                archive.writestr(f"task{task_id}_logs.log", "")
                warnings.append("No experiment log found; packaged an empty log.")

            baselines_dir = PROJECT_ROOT / "baselines"
            if baselines_dir.exists():
                for file_path in baselines_dir.rglob("*"):
                    if file_path.is_file():
                        archive.write(file_path, Path("code") / file_path.relative_to(PROJECT_ROOT))
            else:
                warnings.append("No baselines directory found.")

        warning_text = f" Warnings: {'; '.join(warnings)}" if warnings else ""
        return f"[OK] Created {zip_path}.{warning_text}"
    except Exception as exc:
        return f"[ERROR] submit_final failed: {exc}"
