"""Helpers for rendering definitions from official FlashInfer trace templates.

The pipeline often runs against a FlashInfer wheel where ``fi_trace`` may not
include the newest templates yet. This helper loads only ``flashinfer/trace``
from an explicit local checkout or a pinned GitHub archive, without importing
the full FlashInfer package or any compiled extensions. It then asks the
official ``TraceTemplate`` to render the JSON schema.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import re
import shutil
import sys
import tarfile
import tempfile
import types
import urllib.request
from functools import lru_cache
from pathlib import PurePosixPath
from pathlib import Path
from typing import Any


_PKG = "_flashinfer_official_trace"
_DEFAULT_FLASHINFER_REPO = "https://github.com/flashinfer-ai/flashinfer.git"
_DEFAULT_FLASHINFER_REF = "90548eb322e18bb9ada7b8e98e25427d2ad81408"  # PR #2931 head


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _trace_root_from_repo(repo: Path) -> Path | None:
    root = repo / "flashinfer" / "trace"
    return root if (root / "template.py").is_file() else None


def _github_archive_url(repo: str, ref: str) -> str:
    repo = repo.strip().removesuffix(".git")
    if repo.startswith("git@github.com:"):
        repo = "https://github.com/" + repo[len("git@github.com:") :]
    if not repo.startswith("https://github.com/"):
        raise ValueError(
            "FLASHINFER_SOURCE_REPO must be a GitHub HTTPS repo URL "
            f"or git@github.com URL, got {repo!r}"
        )
    return f"{repo}/archive/{ref}.tar.gz"


def _extract_trace_from_archive(archive_path: Path, dst: Path) -> int:
    shutil.rmtree(dst, ignore_errors=True)
    copied = 0
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar:
            parts = PurePosixPath(member.name).parts
            if len(parts) < 3 or parts[1:3] != ("flashinfer", "trace"):
                continue
            rel_parts = parts[3:]
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


def _cached_github_trace_root() -> Path | None:
    repo = os.environ.get("FLASHINFER_SOURCE_REPO", _DEFAULT_FLASHINFER_REPO)
    ref = os.environ.get("FLASHINFER_SOURCE_REF", _DEFAULT_FLASHINFER_REF)
    if os.environ.get("FLASHINFER_SOURCE_DISABLE_DOWNLOAD", "") not in ("", "0", "false", "False"):
        return None
    cache_base = Path(
        os.environ.get(
            "FLASHINFER_SOURCE_CACHE_DIR",
            "~/.cache/flashinfer_trace/flashinfer_source",
        )
    ).expanduser()
    cache_root = cache_base / ref / "trace"
    if (cache_root / "template.py").is_file():
        return cache_root

    url = _github_archive_url(repo, ref)
    try:
        cache_root.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        cache_root = Path(tempfile.gettempdir()) / "flashinfer_trace" / "flashinfer_source" / ref / "trace"
        if (cache_root / "template.py").is_file():
            return cache_root
        cache_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="flashinfer-source-") as tmp:
        archive_path = Path(tmp) / "flashinfer.tar.gz"
        with urllib.request.urlopen(url, timeout=120) as response:
            with archive_path.open("wb") as out:
                shutil.copyfileobj(response, out)
        copied = _extract_trace_from_archive(archive_path, cache_root)
    if copied == 0:
        return None
    return cache_root if (cache_root / "template.py").is_file() else None


def _trace_root() -> Path | None:
    explicit = os.environ.get("FLASHINFER_SOURCE_DIR", "").strip()
    if explicit:
        root = _trace_root_from_repo(Path(explicit).expanduser())
        if root is None:
            raise ImportError(
                "FLASHINFER_SOURCE_DIR does not contain flashinfer/trace/template.py: "
                f"{explicit}"
            )
        return root

    return _cached_github_trace_root()


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _ensure_package() -> Path:
    root = _trace_root()
    if root is None:
        raise ImportError("official FlashInfer trace templates not found")

    if _PKG not in sys.modules:
        pkg = types.ModuleType(_PKG)
        pkg.__path__ = [str(root)]  # type: ignore[attr-defined]
        sys.modules[_PKG] = pkg
    templates_name = f"{_PKG}.templates"
    if templates_name not in sys.modules:
        templates_pkg = types.ModuleType(templates_name)
        templates_pkg.__path__ = [str(root / "templates")]  # type: ignore[attr-defined]
        sys.modules[templates_name] = templates_pkg
    if f"{_PKG}.template" not in sys.modules:
        _load_module(f"{_PKG}.template", root / "template.py")
    return root


@lru_cache(maxsize=None)
def _template_module(module_name: str):
    root = _ensure_package()
    full_name = f"{_PKG}.templates.{module_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    return _load_module(full_name, root / "templates" / f"{module_name}.py")


def render_definition(
    module_name: str,
    template_name: str,
    fi_api: str,
    kwargs: dict[str, Any],
    *,
    name: str | None = None,
) -> dict | None:
    """Render a definition using an official FlashInfer TraceTemplate if available."""
    try:
        module = _template_module(module_name)
        template = getattr(module, template_name)
        definition = template.build_fi_trace_fn(fi_api)(name=name, **kwargs)
        reference = definition.get("reference")
        if isinstance(reference, str) and "def run(" not in reference:
            definition["reference"] = re.sub(
                r"^(\s*)def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(",
                r"\1def run(",
                reference,
                count=1,
                flags=re.MULTILINE,
            )
            reference = definition["reference"]
        if isinstance(reference, str):
            imports: list[str] = []
            if "torch" in reference and "import torch" not in reference:
                imports.append("import torch")
            if "math." in reference and "import math" not in reference:
                imports.append("import math")
            if imports:
                definition["reference"] = "\n".join(imports) + "\n\n" + reference
        return definition
    except Exception:
        return None


def render_reference(
    module_name: str,
    reference_name: str,
) -> str | None:
    """Render an official reference function source as a definition ``run``."""
    try:
        module = _template_module(module_name)
        reference_fn = getattr(module, reference_name)
        reference = inspect.getsource(reference_fn)
        if "def run(" not in reference:
            reference = re.sub(
                r"^(\s*)def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(",
                r"\1def run(",
                reference,
                count=1,
                flags=re.MULTILINE,
            )
        imports: list[str] = []
        if "torch" in reference and "import torch" not in reference:
            imports.append("import torch")
        if "math." in reference and "import math" not in reference:
            imports.append("import math")
        if imports:
            reference = "\n".join(imports) + "\n\n" + reference
        return reference
    except Exception:
        return None
