"""FlashInfer fi_trace integration helpers for probe scheduler.

This module owns the official fi_trace plumbing:

- attach/copy local FlashInfer trace templates into the remote runtime
- preflight whether fi_trace can dump definitions
- collect dumped fi_trace definition JSONs for transport back to the local entrypoint
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import PurePosixPath
from pathlib import Path
from typing import Any, Callable


LogFn = Callable[[str], None]


def _github_archive_url(repo: str, ref: str) -> str:
    """把 GitHub repo/ref 转成 tar.gz archive 下载地址。

    支持两种输入：
      https://github.com/flashinfer-ai/flashinfer.git
      git@github.com:flashinfer-ai/flashinfer.git
    """
    repo = repo.strip().removesuffix(".git")
    if repo.startswith("git@github.com:"):
        repo = "https://github.com/" + repo[len("git@github.com:") :]
    if not repo.startswith("https://github.com/"):
        raise ValueError(
            "PROBE_FLASHINFER_PATCH_REPO must be a GitHub HTTPS repo URL "
            f"or git@github.com URL, got {repo!r}"
        )
    return f"{repo}/archive/{ref}.tar.gz"


def _extract_flashinfer_package_from_archive(
    archive_path: Path,
    dst: Path,
) -> int:
    """从 GitHub archive 里只抽取 flashinfer/ Python 包目录。

    GitHub archive 顶层通常是 flashinfer-<commit>/flashinfer/...。
    这里丢掉第一层 repo 目录，只把其中的 flashinfer/ 内容复制到 dst。
    返回复制的文件数，用于确认 archive 内容有效。
    """
    shutil.rmtree(dst, ignore_errors=True)
    copied = 0
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar:
            parts = PurePosixPath(member.name).parts
            if len(parts) < 2 or parts[1] != "flashinfer":
                continue
            rel_parts = parts[2:]
            if not rel_parts:
                dst.mkdir(parents=True, exist_ok=True)
                continue
            target = dst.joinpath(*rel_parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(member)
            if src is None:
                continue
            with src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
            copied += 1
    return copied


def ensure_flashinfer_patch_source(
    *,
    remote_flashinfer_patch: Path,
    patch_repo: str,
    patch_ref: str,
    log_phase: LogFn,
) -> None:
    """Ensure a FlashInfer Python patch package exists in the remote runtime.

    The probe keeps the image-installed FlashInfer native/runtime package, but
    overlays Python fi_trace support from a pinned source checkout. If Modal
    already mounted a local checkout at *remote_flashinfer_patch*, that source
    wins. Otherwise this downloads a GitHub archive at a pinned ref and extracts
    only the ``flashinfer/`` Python package directory.
    """
    # 优先使用 Modal 已经挂载/上传好的 source。例如用户显式设置了
    # PROBE_LOCAL_FLASHINFER_PACKAGE，本地目录会被 scheduler.py 上传到这里。
    if remote_flashinfer_patch.is_dir():
        log_phase(f"fi_trace_patch:source local_mount path={remote_flashinfer_patch}")
        return

    patch_repo = patch_repo.strip()
    patch_ref = patch_ref.strip()
    if not patch_repo or not patch_ref:
        log_phase("fi_trace_patch:source image_only")
        return

    # 没有本地 source 时，下载 pinned GitHub ref。这里固定 ref 是为了复现，
    # 不默认追 main/latest。
    url = _github_archive_url(patch_repo, patch_ref)
    with tempfile.TemporaryDirectory(prefix="flashinfer-trace-patch-") as tmp:
        archive_path = Path(tmp) / "flashinfer.tar.gz"
        log_phase(f"fi_trace_patch:download repo={patch_repo} ref={patch_ref}")
        with urllib.request.urlopen(url, timeout=120) as response:
            with archive_path.open("wb") as out:
                shutil.copyfileobj(response, out)
        copied = _extract_flashinfer_package_from_archive(
            archive_path,
            remote_flashinfer_patch,
        )

    if copied == 0:
        raise RuntimeError(
            f"GitHub archive {url} did not contain a flashinfer/ package"
        )
    log_phase(
        f"fi_trace_patch:source github repo={patch_repo} ref={patch_ref} "
        f"path={remote_flashinfer_patch} files={copied}"
    )


def merge_local_flashinfer_patch(
    *,
    remote_flashinfer_patch: Path,
    log_phase: LogFn,
) -> None:
    """把 patch source 里的 FlashInfer trace Python 文件合进已安装包。

    这一步不替换 FlashInfer native/JIT 库，只覆盖 Python 层：
      api_logging.py
      fi_trace.py
      trace/

    目的：让当前镜像里的 flashinfer 包拥有 official fi_trace Python 支持。
    """
    if not remote_flashinfer_patch.is_dir():
        log_phase("fi_trace_patch:skip no_local_flashinfer_patch")
        return

    import site

    # 找当前 Python 环境里真正安装的 flashinfer package 目录。
    # 不能写死 site-packages 路径，因为不同镜像 Python 路径可能不同。
    search_roots = []
    for getter in (site.getsitepackages,):
        try:
            search_roots.extend(Path(p) for p in getter())
        except Exception:
            pass
    try:
        search_roots.append(Path(site.getusersitepackages()))
    except Exception:
        pass
    search_roots.extend(Path(p) for p in sys.path if p)

    candidates = []
    for root in search_roots:
        candidate = root / "flashinfer"
        if candidate.is_dir() and candidate != remote_flashinfer_patch:
            candidates.append(candidate)

    if not candidates:
        msg = "installed flashinfer package not found; cannot apply local fi_trace patch"
        if os.environ.get("PROBE_REQUIRE_OFFICIAL_FI_TRACE", "1") not in ("0", ""):
            raise RuntimeError(msg)
        log_phase(f"fi_trace_patch:skip {msg}")
        return

    dst = candidates[0]
    copied: list[str] = []

    # 复制 official fi_trace 所需的两个顶层 Python 支持文件。
    for rel in ("api_logging.py", "fi_trace.py"):
        src_file = remote_flashinfer_patch / rel
        if not src_file.is_file():
            continue
        dst_file = dst / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        copied.append(rel)

    # 复制 official trace templates，例如 flashinfer/trace/templates/*.py。
    src_trace = remote_flashinfer_patch / "trace"
    if src_trace.is_dir():
        dst_trace = dst / "trace"
        shutil.copytree(
            src_trace,
            dst_trace,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", ".pytest_cache", ".mypy_cache"
            ),
        )
        copied.append("trace/")

    # 清掉 __pycache__ 和已 import 的 flashinfer module，确保后续 import 看到新文件。
    for pycache in dst.rglob("__pycache__"):
        shutil.rmtree(pycache, ignore_errors=True)
    for name in list(sys.modules):
        if name == "flashinfer" or name.startswith("flashinfer."):
            del sys.modules[name]

    log_phase(
        f"fi_trace_patch:merged src={remote_flashinfer_patch} dst={dst} "
        f"files={copied}"
    )


def _preflight_local_fi_trace_templates(log_phase: LogFn) -> dict[str, Any] | None:
    """调用注入目录里的 fi_trace_runtime_patch 做 target preflight。

    返回的是 fi_trace_runtime_patch.preflight_fi_trace_targets() 的结构化结果。
    这里不直接扫描/attach；具体逻辑在 fi_trace_runtime_patch.py。
    """
    if os.environ.get("PROBE_DISABLE_RUNTIME_FI_TRACE_ATTACH", "") in (
        "1",
        "true",
        "True",
    ):
        log_phase("fi_trace_preflight_targets:skip disabled_by_env")
        return None
    import importlib

    attach_mod = importlib.import_module("fi_trace_runtime_patch")
    preflight = getattr(attach_mod, "preflight_fi_trace_targets", None)
    if not callable(preflight):
        return None
    summary = preflight()
    if isinstance(summary, dict):
        return summary
    return None


def _preview_values(values: list[Any], *, limit: int = 8) -> str:
    """把 list/tuple 列表压成一行日志 preview。"""
    if not values:
        return ""
    rendered = []
    for item in values[:limit]:
        if isinstance(item, tuple) and len(item) >= 2:
            rendered.append(f"{item[0]} -> {item[1]}")
        else:
            rendered.append(str(item))
    suffix = "" if len(values) <= limit else f", ... +{len(values) - limit}"
    return ", ".join(rendered) + suffix


def _preview_list(values: list[Any], *, limit: int = 20) -> list[str]:
    """把长列表截断为 JSON 友好的字符串 preview。"""
    return [str(value) for value in values[:limit]]


def _preview_pairs(values: list[Any], *, limit: int = 20) -> list[dict[str, str]]:
    """把 (target, error) 列表截断为 JSON 友好的 preview。"""
    rendered = []
    for item in values[:limit]:
        if isinstance(item, tuple) and len(item) >= 2:
            rendered.append({"target": str(item[0]), "error": str(item[1])})
        else:
            rendered.append({"target": str(item), "error": ""})
    return rendered


def _build_preflight_result(
    *,
    supported: bool,
    summary: dict[str, Any] | None,
    flashinfer_version: str | None,
    flashinfer_path: str | None,
    error: str | None = None,
) -> dict[str, Any]:
    """把 runtime patch summary 整理成 aggregated_summary 里的稳定字段。

    输入 summary 可能包含：
      existing: attach 前当前 runtime 的状态
      final:    attach 后的最终状态

    输出会展开成 existing_*_count / final_*_count 和异常 preview 字段。
    """
    result: dict[str, Any] = {
        "supported": supported,
        "flashinfer_version": flashinfer_version,
        "flashinfer_path": flashinfer_path,
    }
    if error:
        result["error"] = error
    if summary is None:
        result["mode"] = "unavailable"
        return result

    final_summary = summary.get("final") if isinstance(summary.get("final"), dict) else summary
    existing_summary = (
        summary.get("existing")
        if isinstance(summary.get("existing"), dict)
        else summary
    )

    final_attached = final_summary.get("attached", []) or []
    final_missing = final_summary.get("missing", []) or []
    final_failed = final_summary.get("failed", []) or []

    existing_attached = existing_summary.get("attached", []) or []
    existing_no_fi_trace = existing_summary.get("no_fi_trace", []) or []
    existing_missing = existing_summary.get("missing", []) or []
    existing_failed = existing_summary.get("failed", []) or []

    result.update({
        "mode": summary.get("mode", "unknown"),
        "discovery_status": summary.get("discovery_status"),
        "total": summary.get("total", 0),
        "existing_attached_count": len(existing_attached),
        "existing_no_fi_trace_count": len(existing_no_fi_trace),
        "existing_missing_count": len(existing_missing),
        "existing_failed_count": len(existing_failed),
        "final_attached_count": len(final_attached),
        "final_missing_count": len(final_missing),
        "final_failed_count": len(final_failed),
        "existing_no_fi_trace_preview": _preview_list(existing_no_fi_trace),
        "existing_missing_preview": _preview_list(existing_missing),
        "existing_failed_preview": _preview_pairs(existing_failed),
        "final_missing_preview": _preview_list(final_missing),
        "final_failed_preview": _preview_pairs(final_failed),
    })
    return result


def check_official_fi_trace_support(*, log_phase: LogFn) -> dict[str, Any]:
    """official fi_trace preflight 主入口。

    scheduler.py 在真正加载大模型前调用它。
    它做三件事：
      1. 调 fi_trace_runtime_patch 做 target preflight
      2. 确认当前 flashinfer 能 import，并记录版本/路径
      3. 根据 final_attached 是否非空判断 official fi_trace 是否可用

    返回值会写入 aggregated_summary.json["fi_trace_preflight"]。
    """
    import importlib

    summary: dict[str, Any] | None = None
    try:
        # 这一步可能已经完成 runtime attach；具体由 fi_trace_runtime_patch 决定。
        summary = _preflight_local_fi_trace_templates(log_phase)
    except Exception as exc:  # noqa: BLE001
        log_phase(f"fi_trace_preflight_targets:failed {type(exc).__name__}: {exc}")

    details: list[str] = []

    try:
        # 记录当前真正 import 到的 flashinfer 版本和路径，方便排查镜像问题。
        flashinfer = importlib.import_module("flashinfer")
        version = getattr(flashinfer, "__version__", "?")
        path = getattr(flashinfer, "__file__", "?")
        details.append(f"flashinfer version={version} path={path}")
    except Exception as exc:
        import_error = str(exc)
        details.append(f"flashinfer import failed: {exc}")
        supported = False
        flashinfer_version = None
        flashinfer_path = None
    else:
        import_error = None
        flashinfer_version = str(version)
        flashinfer_path = str(path)
        if summary is not None:
            # final_* 表示 preflight 之后的最终状态；
            # existing_* 表示 preflight 之前 runtime 自带的覆盖情况。
            final_summary = (
                summary.get("final") if isinstance(summary.get("final"), dict) else summary
            )
            existing_summary = (
                summary.get("existing")
                if isinstance(summary.get("existing"), dict)
                else summary
            )
            final_attached = final_summary.get("attached", []) or []
            final_missing = final_summary.get("missing", []) or []
            final_failed = final_summary.get("failed", []) or []
            existing_attached = existing_summary.get("attached", []) or []
            existing_no_fi_trace = existing_summary.get("no_fi_trace", []) or []
            # 只要最终有可用 .fi_trace target，就认为 official fi_trace 支持可用。
            supported = bool(final_attached)
            details.append(
                "targets "
                f"{summary.get('discovery_status', '?')} "
                f"mode={summary.get('mode', '?')} "
                f"total={summary.get('total', 0)} "
                f"existing_attached={len(existing_attached)} "
                f"existing_no_fi_trace={len(existing_no_fi_trace)} "
                f"final_attached={len(final_attached)} "
                f"final_missing={len(final_missing)} failed={len(final_failed)}"
            )
            if existing_no_fi_trace:
                details.append(
                    f"existing_no_fi_trace_preview={_preview_values(existing_no_fi_trace)}"
                )
            if final_missing:
                details.append(f"final_missing_preview={_preview_values(final_missing)}")
            if final_failed:
                details.append(f"final_failed_preview={_preview_values(final_failed)}")
        else:
            supported = False
            details.append("targets preflight unavailable")

    log_phase("fi_trace_preflight " + " | ".join(details))
    result = _build_preflight_result(
        supported=supported,
        summary=summary,
        flashinfer_version=flashinfer_version,
        flashinfer_path=flashinfer_path,
        error=import_error,
    )
    if not supported:
        # 默认要求 official fi_trace 可用；否则直接 fail fast，避免加载模型后才发现
        # fi_trace_out 为空。用户可用 PROBE_REQUIRE_OFFICIAL_FI_TRACE=0 放宽。
        msg = (
            "official FlashInfer fi_trace is not available in this Modal image; "
            "FLASHINFER_TRACE_DUMP will not produce fi_trace_out JSON"
        )
        if os.environ.get("PROBE_REQUIRE_OFFICIAL_FI_TRACE", "1") not in ("0", ""):
            raise RuntimeError(msg)
        log_phase(f"fi_trace_preflight:warning {msg}")
        result["warning"] = msg
    return result


def collect_fi_trace_definitions(fi_trace_dir: Path) -> list[dict[str, Any]]:
    """读取 official fi_trace dump 出来的 definition JSON。

    scheduler.py 会把返回值带回本地入口；本地再写到 output_dir/fi_trace_out/。
    这里不做 adapter parse，也不判断是否正式进入 definitions。
    """
    definitions: list[dict[str, Any]] = []
    if not fi_trace_dir.is_dir():
        return definitions
    for path in sorted(fi_trace_dir.glob("*.json")):
        try:
            definitions.append({
                "filename": path.name,
                "definition": json.loads(path.read_text(encoding="utf-8")),
            })
        except Exception as exc:  # noqa: BLE001
            definitions.append({
                "filename": path.name,
                "error": str(exc),
            })
    return definitions
