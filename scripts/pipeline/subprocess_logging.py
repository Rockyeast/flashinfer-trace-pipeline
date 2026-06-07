"""Helpers for teeing subprocess output to both terminal and run-local logs."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


def command_label(command: list[str]) -> str:
    """Return a compact filesystem-safe label for a command."""
    if not command:
        return "command"
    executable = Path(command[0]).name
    target = Path(command[1] if len(command) > 1 and executable.startswith("python") else command[0])
    stem = target.with_suffix("").as_posix().strip("./")
    label = stem.replace("/", "_").replace("\\", "_")
    return label or "command"


def _write_header(log_file: TextIO, command: list[str], cwd: str | Path | None) -> None:
    log_file.write(f"$ {' '.join(command)}\n")
    if cwd is not None:
        log_file.write(f"cwd: {cwd}\n")
    log_file.write("\n")
    log_file.flush()


@dataclass
class StreamedProcess:
    """Popen wrapper whose stdout/stderr is being teed by a reader thread."""

    process: subprocess.Popen
    log_file: TextIO
    reader: threading.Thread

    @property
    def returncode(self) -> int | None:
        return self.process.returncode


def _pipe_output(
    stream,
    log_file: TextIO,
    *,
    prefix: str | None,
) -> None:
    for line in stream:
        rendered = f"[{prefix}] {line}" if prefix else line
        sys.stdout.write(rendered)
        sys.stdout.flush()
        log_file.write(line)
        log_file.flush()


def popen_streamed(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
    log_path: Path,
    prefix: str | None = None,
) -> StreamedProcess:
    """Start command and tee stdout/stderr to terminal and log in a background thread."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    _write_header(log_file, command, cwd)
    try:
        proc = subprocess.Popen(
            command,
            env=env,
            cwd=str(cwd) if cwd is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        message = f"failed to launch command: {type(exc).__name__}: {exc}\n"
        sys.stderr.write(message)
        log_file.write(message)
        log_file.write("\nexit_code: 127\n")
        log_file.close()
        raise
    assert proc.stdout is not None
    reader = threading.Thread(
        target=_pipe_output,
        args=(proc.stdout, log_file),
        kwargs={"prefix": prefix},
        daemon=True,
    )
    reader.start()
    return StreamedProcess(process=proc, log_file=log_file, reader=reader)


def wait_streamed(streamed: StreamedProcess) -> int:
    """Wait for a streamed process, finish log bookkeeping, and close the log."""
    returncode = streamed.process.wait()
    streamed.reader.join()
    streamed.log_file.write(f"\nexit_code: {returncode}\n")
    streamed.log_file.flush()
    streamed.log_file.close()
    return returncode


def run_streamed(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
    log_path: Path | None = None,
    prefix: str | None = None,
) -> int:
    """Run command while streaming stdout/stderr to terminal and optional log file."""
    if log_path is None:
        return subprocess.run(command, env=env, cwd=str(cwd) if cwd is not None else None).returncode
    try:
        streamed = popen_streamed(
            command,
            env=env,
            cwd=cwd,
            log_path=log_path,
            prefix=prefix,
        )
    except OSError:
        return 127
    return wait_streamed(streamed)


def prepend_pythonpath(env: dict[str, str], path: Path) -> dict[str, str]:
    """Return env with path prepended to PYTHONPATH."""
    updated = dict(env)
    path_str = str(path)
    updated["PYTHONPATH"] = path_str + (
        os.pathsep + updated["PYTHONPATH"] if updated.get("PYTHONPATH") else ""
    )
    return updated
