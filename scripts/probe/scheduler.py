"""Run SGLang probe tracing on Modal GPU.

Probe mode: injects worker trace hooks via sitecustomize.py, runs one SGLang
inference pass, aggregates trace events into
aggregated_summary.json, and saves it locally under tmp/fi_probe/ unless
--output-dir is provided by the caller.

Usage:
    modal run scripts/probe/scheduler.py --model-name Qwen/Qwen2.5-7B-Instruct --tp-size 1

    # Save local probe summary to a custom directory
    modal run scripts/probe/scheduler.py \
        --model-name Qwen/Qwen2.5-7B-Instruct \
        --tp-size 1 \
        --output-dir tmp/run/Qwen_Qwen2.5-7B-Instruct/probe

    # Resume interrupted job
    modal run scripts/probe/scheduler.py --resume-function-call-id fc-xxxx --output-dir tmp/run/Qwen_Qwen2.5-7B-Instruct/probe
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import modal

_THIS_FILE = Path(__file__).resolve()
_SCRIPTS_IMPORT_DIR = (
    _THIS_FILE.parents[1] if _THIS_FILE.parent.name == "probe" else Path("/root/flashinfer-trace/scripts")
)
if str(_SCRIPTS_IMPORT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_IMPORT_DIR))

from probe.fi_trace_integration import (  # noqa: E402
    check_official_fi_trace_support,
    collect_fi_trace_definitions,
    ensure_flashinfer_patch_source,
    merge_local_flashinfer_patch,
)
from probe.sitecustomize_bundle import prepare_sitecustomize_bundle  # noqa: E402
from pipeline.runtime_defaults import (  # noqa: E402
    DEFAULT_EXTRA_PIP_PACKAGES,
    DEFAULT_FLASHINFER_BENCH_PROBE_PACKAGE,
    DEFAULT_MODAL_GPU,
    DEFAULT_MODAL_GPU_COUNT,
    DEFAULT_SGLANG_IMAGE,
    DEFAULT_SGLANG_PACKAGE,
    forwarded_runtime_env,
)

# ──────────────────────────────────────────────────────────────────────────────
# Modal App 定义
# ──────────────────────────────────────────────────────────────────────────────

# Modal App 名称，用于在 Modal 控制台标识此应用
app = modal.App("flashinfer-kernel-tracer")

# GPU 规格：通过环境变量在运行时覆盖，未设置时使用下方默认值
# 默认 A100-80GB × 2（适合 ≤35B 模型；更大模型需手动调整 GPU_COUNT）
# 覆盖示例：
#   MODAL_GPU=H100 modal run scripts/probe/scheduler.py --model-name ... --tp-size 1
#   MODAL_GPU=H100 MODAL_GPU_COUNT=4 modal run scripts/probe/scheduler.py --model-name ... --tp-size 4
_GPU_TYPE = os.environ.get("MODAL_GPU", DEFAULT_MODAL_GPU)   # 可选值：A100-80GB / H100 / A10G 等
_GPU_COUNT = int(os.environ.get("MODAL_GPU_COUNT", str(DEFAULT_MODAL_GPU_COUNT)))
# Modal GPU 规格字符串：多 GPU 时格式为 "A100-80GB:2"，单 GPU 时直接是 "A100-80GB"
_GPU_SPEC = f"{_GPU_TYPE}:{_GPU_COUNT}" if _GPU_COUNT > 1 else _GPU_TYPE

# ──────────────────────────────────────────────────────────────────────────────
# Modal 镜像构建
# ──────────────────────────────────────────────────────────────────────────────

_BASE_ENV = {
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "PYTHONFAULTHANDLER": "1",
    "PYTHONUNBUFFERED": "1",
    "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
    "HUGGING_FACE_HUB_TOKEN": os.environ.get(
        "HUGGING_FACE_HUB_TOKEN",
        os.environ.get("HF_TOKEN", ""),
    ),
}
_BASE_ENV.update(forwarded_runtime_env())


_LOCAL_FLASHINFER_PACKAGE_ENV = os.environ.get(
    "PROBE_LOCAL_FLASHINFER_PACKAGE",
    "",
).strip()
_LOCAL_FLASHINFER_PACKAGE = (
    Path(_LOCAL_FLASHINFER_PACKAGE_ENV).expanduser()
    if _LOCAL_FLASHINFER_PACKAGE_ENV
    else None
)
_REMOTE_FLASHINFER_PATCH = Path("/root/local_flashinfer_patch/flashinfer")
_DEFAULT_FLASHINFER_PATCH_REPO = "https://github.com/flashinfer-ai/flashinfer.git"
_DEFAULT_FLASHINFER_PATCH_REF = "90548eb322e18bb9ada7b8e98e25427d2ad81408"  # PR #2931 head
_FLASHINFER_PATCH_REPO = os.environ.get(
    "PROBE_FLASHINFER_PATCH_REPO",
    _DEFAULT_FLASHINFER_PATCH_REPO,
)
_FLASHINFER_PATCH_REF = os.environ.get(
    "PROBE_FLASHINFER_PATCH_REF",
    _DEFAULT_FLASHINFER_PATCH_REF,
)
_SGLANG_PACKAGE = os.environ.get("PROBE_SGLANG_PACKAGE", DEFAULT_SGLANG_PACKAGE)
_SGLANG_IMAGE = os.environ.get("PROBE_SGLANG_IMAGE", DEFAULT_SGLANG_IMAGE).strip()
_FLASHINFER_VERSION = os.environ.get("PROBE_FLASHINFER_VERSION", "").strip()
_EXTRA_PIP_PACKAGES = shlex.split(
    os.environ.get("PROBE_EXTRA_PIP_PACKAGES", DEFAULT_EXTRA_PIP_PACKAGES)
)
_SKIP_FLASHINFER_PATCH = os.environ.get("PROBE_SKIP_FLASHINFER_PATCH", "") not in (
    "",
    "0",
    "false",
    "False",
)
if (
    _LOCAL_FLASHINFER_PACKAGE
    and not _LOCAL_FLASHINFER_PACKAGE.is_dir()
    and not _SKIP_FLASHINFER_PATCH
):
    raise ValueError(
        "PROBE_LOCAL_FLASHINFER_PACKAGE points to a missing directory: "
        f"{_LOCAL_FLASHINFER_PACKAGE}"
    )
if _FLASHINFER_VERSION and "FLASHINFER_DISABLE_VERSION_CHECK" not in _BASE_ENV:
    # flashinfer-jit-cache is not published for every flashinfer-python release.
    # When experimenting with a newer Python package over an existing SGLang
    # image, bypass the cache-package version check and let preflight decide
    # whether imports still work.
    _BASE_ENV["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"


def _build_base_image() -> modal.Image:
    """Build the Modal image used by probe.

    Default path uses the pinned official SGLang image. Set
    PROBE_SGLANG_IMAGE="" to use the PyTorch + PyPI SGLang fallback path.
    """
    override_image = _SGLANG_IMAGE
    if override_image:
        image = (
            modal.Image.from_registry(override_image)
            .pip_install(DEFAULT_FLASHINFER_BENCH_PROBE_PACKAGE)
            .pip_install("huggingface_hub>=0.34,<1.0", "hf_transfer", "safetensors")
        )
        if _FLASHINFER_VERSION:
            image = image.pip_install(
                f"flashinfer-python=={_FLASHINFER_VERSION}",
                f"flashinfer-cubin=={_FLASHINFER_VERSION}",
            )
        if _EXTRA_PIP_PACKAGES:
            image = image.pip_install(*_EXTRA_PIP_PACKAGES)
        image = image.pip_install("nvidia-cudnn-cu12==9.16.0.29")
        return image.env(_BASE_ENV)

    # base_image：从公共 PyTorch Docker 镜像出发，安装 SGLang 和 FlashInfer 依赖
    # 这一层不包含本地代码，Modal 会对其进行缓存（只要依赖不变，后续 rebuild 直接复用）
    image = (
        modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
        .apt_install("libnuma1", "build-essential")
        .pip_install(_SGLANG_PACKAGE)
        .pip_install(DEFAULT_FLASHINFER_BENCH_PROBE_PACKAGE)
        .pip_install("nvidia-cudnn-cu12==9.16.0.29")
    )
    if _FLASHINFER_VERSION:
        image = image.pip_install(
            f"flashinfer-python=={_FLASHINFER_VERSION}",
            f"flashinfer-cubin=={_FLASHINFER_VERSION}",
        )
    if _EXTRA_PIP_PACKAGES:
        image = image.pip_install(*_EXTRA_PIP_PACKAGES)
    return image.env(_BASE_ENV)


base_image = _build_base_image()

# image：在 base_image 之上，把本地的 scripts/ 目录上传到容器
#
# add_local_dir() 的工作方式：
#   - 每次 `modal run` 时，Modal 将本地目录打包并上传到容器的文件系统（不是 Volume）
#   - 上传到的是容器内的普通文件系统，容器退出后这些文件随容器一起销毁
#   - Modal 会对内容做 hash 缓存：本地文件没变化时，不会重新上传（加速启动）
#
# scripts/ → /root/flashinfer-trace/scripts：
#   容器内 run_probe() 调用 inference_runner.py 等子脚本需要这个路径
image = base_image.add_local_dir("scripts", remote_path="/root/flashinfer-trace/scripts")

# Optional official-fi_trace patch:
# PyPI FlashInfer in the SGLang image can lag behind the local FlashInfer checkout.
# If explicitly requested, upload a local Python package and merge it into the
# installed site-packages package at runtime. Otherwise run_probe() downloads a
# pinned GitHub PR archive. Both paths keep the installed native/JIT artifacts,
# while replacing Python decorators/templates such as @flashinfer_api(trace=...).
if (
    _LOCAL_FLASHINFER_PACKAGE
    and _LOCAL_FLASHINFER_PACKAGE.is_dir()
    and not _SKIP_FLASHINFER_PATCH
):
    image = image.add_local_dir(
        str(_LOCAL_FLASHINFER_PACKAGE),
        remote_path=str(_REMOTE_FLASHINFER_PATCH),
    )

# ──────────────────────────────────────────────────────────────────────────────
# 路径 / 超时常量（均在云端容器内部使用）
# ──────────────────────────────────────────────────────────────────────────────

# probe 输出根目录（云端临时目录，容器结束后丢弃）
TRACE_OUTPUT_DIR                = Path("/tmp/probe-output")
# 官方 FlashInfer fi_trace dump 目录。子进程真实调用 FlashInfer API 时会
# 自动把 definition JSON 写到这里，父进程再打包返回给本地入口。
FI_TRACE_OUT_DIR                = TRACE_OUTPUT_DIR / "fi_trace_out"
# sitecustomize.py 注入包的存放目录（需要加入 PYTHONPATH）
INJECTION_DIR                   = Path("/tmp/probe-sitecustomize")
# 云端 sitecustomize.py 源文件路径（被复制进 sitecustomize 注入目录）
REMOTE_SITECUSTOMIZE_SOURCE_PATH = Path("/root/flashinfer-trace/scripts/probe/sitecustomize.py")
# 云端 runtime.py 源文件路径（被复制为 worker_trace_runtime.py）
REMOTE_RUNTIME_SOURCE_PATH      = Path("/root/flashinfer-trace/scripts/probe/runtime.py")
# 云端 fi_trace_runtime_patch.py 源文件路径（被复制进 sitecustomize 注入目录）
REMOTE_FI_TRACE_RUNTIME_PATCH_PATH = Path("/root/flashinfer-trace/scripts/probe/fi_trace_runtime_patch.py")
# 云端 inference_runner.py 的路径（作为子进程启动 SGLang 推理）
CHILD_RUNNER_PATH               = Path("/root/flashinfer-trace/scripts/probe/inference_runner.py")
# 子进程（inference_runner.py）写入推理结果的 JSON 文件
CHILD_RESULT_PATH               = TRACE_OUTPUT_DIR / "child_result.json"
# 子进程写入当前阶段状态的 JSON 文件（供父进程轮询监控进度）
CHILD_STATUS_PATH               = TRACE_OUTPUT_DIR / "child_status.json"

def _env_int(name: str, default: int) -> int:
    """Read a positive integer from the environment, falling back to default."""
    value = os.environ.get(name)
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


# SGLang 引擎初始化超时（秒）：模型加载 + 编译可能很慢，大模型默认给 60 分钟
ENGINE_INIT_TIMEOUT_SECONDS     = _env_int("PROBE_ENGINE_INIT_TIMEOUT_SECONDS", 3600)
# SGLang 生成推理超时（秒）：大模型 + probe 插桩开销大，大模型默认给 60 分钟
GENERATE_TIMEOUT_SECONDS        = _env_int("PROBE_GENERATE_TIMEOUT_SECONDS", 3600)
# 子进程被 SIGTERM 后，等待其优雅退出的宽限期（秒）
CHILD_KILL_GRACE_SECONDS        = 20
# SGLang 整体 watchdog 超时（传给 inference_runner.py）
SGLANG_WATCHDOG_TIMEOUT_SECONDS = _env_int("PROBE_SGLANG_WATCHDOG_TIMEOUT_SECONDS", 3600)
# Modal 容器总超时（秒）：必须大于 engine/generate watchdog，否则外层会先杀容器
PROBE_CONTAINER_TIMEOUT_SECONDS = _env_int(
    "PROBE_CONTAINER_TIMEOUT_SECONDS",
    max(ENGINE_INIT_TIMEOUT_SECONDS, GENERATE_TIMEOUT_SECONDS, SGLANG_WATCHDOG_TIMEOUT_SECONDS) + 900,
)
# 本地轮询 Modal 结果时，每次 .get() 最多等待多少秒再打印心跳
WAIT_HEARTBEAT_SECONDS          = 30
# 本地调用 function_call.get() 的单次超时（秒）
LOCAL_RESULT_POLL_TIMEOUT_SECONDS = 30
# 本地与 Modal 断线后，重新连接前等待的时间（秒）
LOCAL_REATTACH_DELAY_SECONDS    = 3


# ──────────────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────────────

def _log_phase(message: str) -> None:
    """打印带时间戳的阶段日志（flush=True 确保在 Modal 容器日志里实时可见）。"""
    print(f"[trace-phase {datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def _wait_for_modal_result(function_call: modal.FunctionCall, function_call_id: str):
    """在本地轮询 Modal 云端任务结果，直到任务完成或出现不可恢复的错误。

    - 每隔 LOCAL_RESULT_POLL_TIMEOUT_SECONDS 秒打印一次心跳日志
    - 如果网络断线（ConnectionError），自动重连并继续等待
    """
    start_time = time.time()
    reconnect_count = 0
    while True:
        try:
            # .get(timeout=...) 阻塞等待结果，超时后抛 TimeoutError（不是失败，继续循环）
            return function_call.get(timeout=LOCAL_RESULT_POLL_TIMEOUT_SECONDS)
        except TimeoutError:
            # 超时不代表任务失败，只是本次等待窗口到期，打印心跳后继续
            _log_phase(
                "local:waiting "
                f"function_call_id={function_call_id} "
                f"elapsed={int(time.time() - start_time)}s"
            )
        except modal.exception.ConnectionError as exc:
            # 网络断线：记录重连次数，等待片刻后重新获取 FunctionCall 句柄
            reconnect_count += 1
            _log_phase(
                "local:connection_lost "
                f"function_call_id={function_call_id} "
                f"attempt={reconnect_count} error={exc}"
            )
            time.sleep(LOCAL_REATTACH_DELAY_SECONDS)
            try:
                # 用 function_call_id 重新获取 FunctionCall 对象，继续等待
                function_call = modal.FunctionCall.from_id(function_call_id)
            except modal.exception.ConnectionError as reattach_exc:
                _log_phase(
                    "local:reattach_failed "
                    f"function_call_id={function_call_id} error={reattach_exc}"
                )


def _update_local_probe_metadata(output_dir: Path, **updates: Any) -> None:
    """Update probe_run.json so interrupted local clients can inspect/resume."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "probe_run.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except json.JSONDecodeError:
        payload = {}
    now = datetime.now().isoformat(timespec="seconds")
    payload.setdefault("created_at", now)
    payload.update(updates)
    payload["updated_at"] = now
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _signal_name(exitcode: int) -> str:
    """将子进程退出码转换为可读字符串（正数直接显示，负数转换为信号名如 SIGSEGV）。"""
    if exitcode >= 0:
        return str(exitcode)
    try:
        return signal.Signals(-exitcode).name
    except ValueError:
        return str(exitcode)


def _kill_process_group(pid: int, sig: int) -> None:
    """向整个进程组发送信号（os.killpg），忽略进程已退出的错误。

    用进程组而非单个 PID 是为了同时终止子进程派生的所有孙进程（如 SGLang worker）。
    """
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        # 进程已经退出，忽略
        pass


def _terminate_child(proc: subprocess.Popen) -> None:
    """优雅终止子进程：先发 SIGTERM，超时后再发 SIGKILL。

    步骤：
    1. 检查子进程是否已退出（poll() is not None），已退出则直接返回
    2. 发送 SIGTERM，等待 CHILD_KILL_GRACE_SECONDS 秒
    3. 如果仍未退出，强制发 SIGKILL
    """
    if proc.poll() is not None:
        return  # 子进程已退出，无需操作
    _log_phase(f"child:terminate pid={proc.pid}")
    _kill_process_group(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=CHILD_KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        # 宽限期内未退出，强制杀死
        _log_phase(f"child:kill pid={proc.pid}")
        _kill_process_group(proc.pid, signal.SIGKILL)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# 聚合函数：将多个 worker 的 raw/*.jsonl 汇总为一个 aggregated_summary.json
# ──────────────────────────────────────────────────────────────────────────────

# 聚合时 signature key 中被归一化为 "*" 的属性名。
# 这些属性在每一层的值都不同（如 layer_id=0/1/.../27），但对于 kernel 分类
# 来说各层是等价的，归一化后可将 N_layers 个重复 signature 折叠为 1 条，
# 大幅减小 aggregated_summary.json 体积。
_SIG_KEY_NORMALIZE_ATTRS = frozenset({"layer_id"})


def _normalize_sig_for_key(obj: Any) -> Any:
    """递归遍历 signature 对象，将 _SIG_KEY_NORMALIZE_ATTRS 中的属性值替换为 "*"。

    只影响 signature_key（用于去重的哈希字符串），不修改写入文件的实际 signature。
    """
    if isinstance(obj, dict):
        return {
            k: ("*" if k in _SIG_KEY_NORMALIZE_ATTRS else _normalize_sig_for_key(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_normalize_sig_for_key(item) for item in obj]
    return obj


def _aggregate_probe_trace_dir(trace_dir: Path) -> dict:
    """读取 trace_dir/raw/*.jsonl（各 GPU worker 独立写入的事件文件），
    聚合成一个统一的摘要字典，并写入 aggregated_summary.json。

    聚合内容：
    - trace_id_counts：每个 trace_id 的总调用次数
    - signature_counts：每个 (trace_id, signature) 组合的出现次数及首次调用栈
    - metadata：meta/*.json 中的进程元数据列表

    返回聚合字典（同时写入文件）。
    """
    raw_dir = trace_dir / "raw"    # 各 worker 写入原始事件的目录
    meta_dir = trace_dir / "meta"  # worker 写入元数据（进程信息、分类结果）的目录
    # 确保目录存在（即使 probe 没有产生任何事件）
    trace_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # ── 第零步：读取 inference_start_ts，用于过滤 warmup 阶段的事件 ──
    # inference_runner.py 在 Engine init 完成后写入此文件，
    # 时间戳之前的 FlashInfer 调用属于 CUDA graph warmup，不代表真实推理路径
    inference_ts_path = trace_dir / "inference_start_ts"
    inference_start_ts: float | None = None
    if inference_ts_path.exists():
        try:
            inference_start_ts = float(inference_ts_path.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            pass  # 文件损坏或无法读取时不过滤，退化为原有行为

    # 统计容器：
    trace_id_counts: Counter[str] = Counter()                    # trace_id → 总调用次数
    signature_counts: dict[str, dict] = {}                       # signature_key → 详细信息
    trace_ids_with_stack: set[str] = set()                       # 已记录过 stack 的 trace_id（每个只保留一份）
    trace_ids_with_param_names: dict[str, list[str]] = {}        # trace_id → param_names（每个只保留一份）
    param_name_conflicts: list[dict[str, Any]] = []              # 同一 trace_id 出现不同 param_names 的诊断记录
    param_name_conflict_keys: set[tuple[str, str]] = set()
    total_events = 0
    warmup_events = 0                                            # warmup 阶段被过滤掉的事件数
    raw_file_count = len(list(raw_dir.rglob("*.jsonl")))

    # ── 第一步：遍历所有 raw/*.jsonl 文件，按行解析事件 ──
    for raw_path in sorted(raw_dir.rglob("*.jsonl")):
        with raw_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue  # 跳过空行
                event = json.loads(line)

                # 过滤 warmup 事件：Engine init 期间的调用不计入 kernel inventory
                if inference_start_ts is not None:
                    event_time = event.get("time")
                    if event_time is not None and event_time < inference_start_ts:
                        warmup_events += 1
                        continue

                trace_id = event.get("trace_id", "unknown_trace_id")
                category = event.get("category", "unknown_category")
                signature = event.get("signature", {})
                total_events += 1
                trace_id_counts[trace_id] += 1

                # 用 JSON 序列化作为 signature 的唯一 key（稳定、可比较）。
                # 归一化 layer_id 等属性（用 "*" 替代具体值），使同一结构
                # 但 layer_id 不同的 signature 被折叠成同一条，避免 N_layers
                # 倍的 signature 爆炸（如 28 层 × 98 shape = 2744 条 → 98 条）。
                signature_key = json.dumps(
                    {"trace_id": trace_id, "category": category,
                     "signature": _normalize_sig_for_key(signature)},
                    sort_keys=True, ensure_ascii=False,
                )
                # 提取 param_names（每个 trace_id 只保留第一份，和 stack 策略一致）
                event_param_names = event.get("param_names")
                if event_param_names:
                    if trace_id not in trace_ids_with_param_names:
                        trace_ids_with_param_names[trace_id] = event_param_names
                    elif event_param_names != trace_ids_with_param_names[trace_id]:
                        conflict_key = (
                            trace_id,
                            json.dumps(event_param_names, sort_keys=True, ensure_ascii=False),
                        )
                        if conflict_key not in param_name_conflict_keys:
                            param_name_conflict_keys.add(conflict_key)
                            param_name_conflicts.append(
                                {
                                    "trace_id": trace_id,
                                    "kept": trace_ids_with_param_names[trace_id],
                                    "seen": event_param_names,
                                    "pid": event.get("pid"),
                                    "raw_file": str(raw_path),
                                }
                            )

                if signature_key not in signature_counts:
                    # 首次出现：初始化条目
                    # stack 只保留每个 trace_id 的第一份（减少文件体积），其余置 None
                    event_stack = event.get("stack")
                    if event_stack and trace_id not in trace_ids_with_stack:
                        trace_ids_with_stack.add(trace_id)
                        recorded_stack = event_stack
                    else:
                        recorded_stack = None
                    signature_counts[signature_key] = {
                        "trace_id": trace_id,
                        "category": category,
                        "signature": signature,
                        "count": 0,
                        "first_pid": event.get("pid"),
                        "stack": recorded_stack,
                    }
                signature_counts[signature_key]["count"] += 1
                # 补全调用栈：如果该 trace_id 还没有 stack，且本 event 有 stack，则补上
                if (
                    trace_id not in trace_ids_with_stack
                    and event.get("stack") is not None
                ):
                    trace_ids_with_stack.add(trace_id)
                    signature_counts[signature_key]["stack"] = event["stack"]

    # ── 第二步：读取进程元数据（meta/*.json）──
    metadata = []
    for meta_path in sorted(meta_dir.rglob("*.json")):
        metadata.append(json.loads(meta_path.read_text(encoding="utf-8")))

    # ── 第三步：将 param_names 附加到对应 trace_id 的首个 signature 条目 ──
    # param_names 和 stack 策略一致：每个 trace_id 只保留一份，记录在首个条目上
    if trace_ids_with_param_names:
        # 找到每个 trace_id 对应的首个 signature 条目并附加 param_names
        attached_tids: set[str] = set()
        for entry in signature_counts.values():
            tid = entry["trace_id"]
            if tid in trace_ids_with_param_names and tid not in attached_tids:
                entry["param_names"] = trace_ids_with_param_names[tid]
                attached_tids.add(tid)

    # ── 第四步：组装聚合字典 ──
    aggregated = {
        "trace_dir": str(trace_dir),
        "raw_file_count": raw_file_count,          # 有多少个 worker 写了 raw 文件
        "meta_file_count": len(metadata),           # 有多少个 worker 写了 meta 文件
        "total_events": total_events,               # 所有 raw 事件总数（不含 warmup）
        "warmup_events": warmup_events,             # warmup 阶段被过滤的事件数
        "unique_trace_ids": len(trace_id_counts),   # 不重复的 trace_id 数量
        "unique_signatures": len(signature_counts), # 不重复的 (trace_id, signature) 组合数
        "param_name_conflicts": param_name_conflicts,
        "top_trace_ids": [
            {"trace_id": tid, "count": cnt}
            for tid, cnt in trace_id_counts.most_common()   # 按调用次数降序排列
        ],
        "top_signatures": sorted(
            signature_counts.values(), key=lambda x: (-x["count"], x["trace_id"])
        ),
        "processes": metadata,
    }
    # ── 第五步：写入 aggregated_summary.json ──
    (trace_dir / "aggregated_summary.json").write_text(
        json.dumps(aggregated, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return aggregated


# ──────────────────────────────────────────────────────────────────────────────
# Modal 云端函数：在 GPU 容器内运行 probe
# ──────────────────────────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu=_GPU_SPEC,
    memory=131072,       # 128 GB 内存（大模型需要）
    timeout=PROBE_CONTAINER_TIMEOUT_SECONDS,  # 容器最长运行时间（比内部超时略长）
    retries=0,           # 不自动重试（失败时需人工介入）
    single_use_containers=True,  # 每次调用使用全新容器（避免状态污染）
    secrets=[modal.Secret.from_name("huggingface-secret")] if os.environ.get("USE_HF_SECRET") else [],
)
def run_probe(
    model_name: str,
    prompt: str = "Explain quantum computing in one sentence.",
    max_new_tokens: int = 96,
    tp_size: int = 2,
    probe_coverage: str = "fast",
    probe_prefill_mode: str = "default",
    probe_page_size: int = 0,
    probe_mem_fraction_static: float = 0.7,
    force_flashinfer_backends: bool = False,
    preflight_only: bool = False,
) -> dict:
    """在 Modal GPU 容器内执行 probe 追踪，返回聚合摘要字典。

    执行流程：
    1. prepare_sitecustomize_bundle()：准备注入包 + 设置环境变量
    2. subprocess.Popen(inference_runner.py)：启动子进程运行 SGLang 推理
    3. 父进程轮询 CHILD_STATUS_PATH，监控子进程阶段和超时
    4. 子进程完成后，_aggregate_probe_trace_dir() 聚合 raw/*.jsonl
    5. 返回聚合结果（由 Modal 序列化传回本地）
    """
    _log_phase("prepare_sitecustomize:start")
    prepare_sitecustomize_bundle(
        custom_site_dir=INJECTION_DIR,
        trace_dir=TRACE_OUTPUT_DIR,
        remote_sitecustomize_source_path=REMOTE_SITECUSTOMIZE_SOURCE_PATH,
        remote_runtime_source_path=REMOTE_RUNTIME_SOURCE_PATH,
        remote_fi_trace_runtime_patch_path=REMOTE_FI_TRACE_RUNTIME_PATCH_PATH,
        fi_trace_out_dir=FI_TRACE_OUT_DIR,
    )
    _log_phase("prepare_sitecustomize:done")
    probe_coverage = (probe_coverage or "fast").strip().lower()
    if probe_coverage not in {"fast", "full"}:
        raise ValueError(f"Unknown probe_coverage={probe_coverage!r}; expected fast or full")
    _log_phase(f"probe_coverage={probe_coverage}")
    probe_prefill_mode = (probe_prefill_mode or "default").strip().lower()
    if probe_prefill_mode not in {"default", "paged"}:
        raise ValueError(
            f"Unknown probe_prefill_mode={probe_prefill_mode!r}; expected default or paged"
        )
    _log_phase(f"probe_prefill_mode={probe_prefill_mode}")
    probe_page_size = probe_page_size if probe_page_size > 0 else None
    if probe_prefill_mode == "paged":
        _log_phase(
            f"probe_page_size={probe_page_size if probe_page_size is not None else 'sglang_default'}"
        )
    _log_phase(f"force_flashinfer_backends={force_flashinfer_backends}")
    _log_phase(f"probe_mem_fraction_static={probe_mem_fraction_static}")

    # official fi_trace 依赖 FlashInfer Python 层的 @flashinfer_api(trace=...)
    # 模板绑定。先准备本地挂载或 pinned GitHub FlashInfer Python patch，再
    # merge 到远端 pip 包并做 preflight；这样不必等模型加载一小时后才发现
    # fi_trace_out 为空。
    _log_phase("fi_trace_patch:start")
    if _SKIP_FLASHINFER_PATCH:
        _log_phase("fi_trace_patch:source skipped_by_env")
    else:
        ensure_flashinfer_patch_source(
            remote_flashinfer_patch=_REMOTE_FLASHINFER_PATCH,
            patch_repo=_FLASHINFER_PATCH_REPO,
            patch_ref=_FLASHINFER_PATCH_REF,
            log_phase=_log_phase,
        )
        merge_local_flashinfer_patch(
            remote_flashinfer_patch=_REMOTE_FLASHINFER_PATCH,
            log_phase=_log_phase,
        )
    _log_phase("fi_trace_patch:done")
    _log_phase("fi_trace_preflight:start")
    fi_trace_preflight = check_official_fi_trace_support(log_phase=_log_phase)
    _log_phase("fi_trace_preflight:done")

    if preflight_only:
        return {
            "model_name": model_name,
            "tp_size": tp_size,
            "total_events": 0,
            "unique_trace_ids": 0,
            "unique_signatures": 0,
            "top_trace_ids": [],
            "top_signatures": [],
            "processes": [],
            "result_text": None,
            "error": None,
            "fi_trace_preflight": fi_trace_preflight,
            "fi_trace_definitions": collect_fi_trace_definitions(FI_TRACE_OUT_DIR),
        }

    result_text = None
    error = None

    # ── 启动子进程：inference_runner.py 负责启动 SGLang 引擎并完成推理 ──
    # 使用 start_new_session=True 创建独立进程组，方便后续用 os.killpg 一次性终止
    _log_phase("child:spawn")
    child_cmd = [
        sys.executable,
        str(CHILD_RUNNER_PATH),
        "--model-name", model_name,
        "--prompt", prompt,
        "--max-new-tokens", str(max_new_tokens),
        "--tp-size", str(tp_size),
        "--watchdog-timeout", str(SGLANG_WATCHDOG_TIMEOUT_SECONDS),
        "--mem-fraction-static", str(probe_mem_fraction_static),
        "--result-path", str(CHILD_RESULT_PATH),   # 子进程写推理结果的文件
        "--status-path", str(CHILD_STATUS_PATH),   # 子进程写阶段状态的文件
        "--probe-coverage", probe_coverage,
        "--probe-prefill-mode", probe_prefill_mode,
    ]
    if probe_page_size is not None:
        child_cmd.extend(["--probe-page-size", str(probe_page_size)])
    if force_flashinfer_backends:
        child_cmd.append("--force-flashinfer-backends")
    child_env = os.environ.copy()
    if probe_prefill_mode == "paged":
        child_env["SGLANG_FLASHINFER_USE_PAGED"] = "1"
        _log_phase("child_env:SGLANG_FLASHINFER_USE_PAGED=1")

    child = subprocess.Popen(
        child_cmd,
        env=child_env,          # 继承父进程环境变量（包括 PYTHONPATH 和 SGLANG_WORKER_TRACE_* 等）
        start_new_session=True,  # 新进程组，方便统一 kill
    )
    _log_phase(f"child:started pid={child.pid}")

    # ── 父进程监控循环：轮询子进程状态 ──
    start_time = time.time()
    last_phase = "child:started"
    phase_start = start_time
    last_wait_log = start_time

    while child.poll() is None:   # 子进程仍在运行时持续轮询
        time.sleep(5)
        if CHILD_STATUS_PATH.exists():
            # 读取子进程写入的状态文件（包含当前阶段名和时间戳）
            status = json.loads(CHILD_STATUS_PATH.read_text(encoding="utf-8"))
            phase = status.get("phase", last_phase)
            ts = float(status.get("ts", phase_start))
            if phase != last_phase:
                # 阶段切换：打印日志，重置阶段计时器
                last_phase = phase
                phase_start = ts
                detail = status.get("detail")
                if detail:
                    _log_phase(f"parent:phase phase={phase} detail={detail}")
                else:
                    _log_phase(f"parent:phase phase={phase}")

        now = time.time()
        # 每隔 WAIT_HEARTBEAT_SECONDS 打印一次心跳日志（证明父进程仍在运行）
        if now - last_wait_log >= WAIT_HEARTBEAT_SECONDS:
            _log_phase(
                "parent:waiting "
                f"phase={last_phase} "
                f"phase_elapsed={int(now - phase_start)}s "
                f"total_elapsed={int(now - start_time)}s"
            )
            last_wait_log = now

        # 超时检测：推理阶段超时（generate 卡住）
        if last_phase == "generate:start" and now - phase_start > GENERATE_TIMEOUT_SECONDS:
            error = (
                f"Generate timed out after {GENERATE_TIMEOUT_SECONDS}s; "
                "terminating child before container-level crash/retry."
            )
            _terminate_child(child)
            break

        # 超时检测：引擎初始化超时（模型加载/编译卡住）
        if last_phase != "generate:start" and now - start_time > ENGINE_INIT_TIMEOUT_SECONDS:
            error = (
                f"Engine init timed out after {ENGINE_INIT_TIMEOUT_SECONDS}s; "
                "terminating child before container-level crash/retry."
            )
            _terminate_child(child)
            break

    # ── 子进程已退出：检查退出码 ──
    if child.returncode not in (None, 0) and error is None:
        error = (
            f"Child exited abnormally with exit code {child.returncode} "
            f"({_signal_name(child.returncode)})."
        )

    # ── 读取子进程写入的推理结果文件 ──
    if CHILD_RESULT_PATH.exists():
        child_result = json.loads(CHILD_RESULT_PATH.read_text(encoding="utf-8"))
        result_text = child_result.get("result_text")   # SGLang 生成的文本
        if child_result.get("error"):
            error = child_result["error"]

    # ── 聚合 raw/*.jsonl → aggregated_summary.json ──
    _log_phase("aggregate:start")
    aggregated = _aggregate_probe_trace_dir(TRACE_OUTPUT_DIR)
    _log_phase("aggregate:done")

    # 在聚合字典里附加本次 probe 的元信息，一起返回给本地
    aggregated["model_name"] = model_name
    aggregated["tp_size"] = tp_size
    aggregated["probe_coverage"] = probe_coverage
    aggregated["probe_prefill_mode"] = probe_prefill_mode
    aggregated["probe_page_size"] = probe_page_size if probe_prefill_mode == "paged" else None
    aggregated["force_flashinfer_backends"] = force_flashinfer_backends
    aggregated["result_text"] = result_text   # SGLang 生成的文本（用于验证推理是否正常）
    aggregated["error"] = error               # 错误信息（None 表示成功）
    aggregated["fi_trace_preflight"] = fi_trace_preflight
    aggregated["fi_trace_definitions"] = collect_fi_trace_definitions(FI_TRACE_OUT_DIR)
    return aggregated


# ──────────────────────────────────────────────────────────────────────────────
# Modal 本地入口：在本地机器上调用，提交任务到云端并等待结果
# ──────────────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    model_name: str = "",
    tp_size: int = 0,
    prompt: str = "Explain quantum computing in one sentence.",
    max_new_tokens: int = 96,
    output_dir: str = "",                  # 非空时把 aggregated_summary.json 写到此目录
    resume_function_call_id: str = "",     # 非空时跳过提交，直接 reattach 到已有任务
    probe_coverage: str = "fast",           # fast=默认快速发现；full=广覆盖 probe
    probe_prefill_mode: str = "default",     # default=普通 probe；paged=启用 piecewise graph 触发 paged-prefill
    probe_page_size: int = 0,                # paged probe 的 SGLang page_size；0=交给 SGLang 默认值
    probe_mem_fraction_static: float = 0.7,  # SGLang Engine mem_fraction_static
    force_flashinfer_backends: bool = False, # 强制细粒度 attention/sampling backend 使用 FlashInfer
    preflight_only: bool = False,           # 只检查远端 FlashInfer fi_trace 支持，不加载模型
):
    """本地入口（modal run 时执行这里）。

    主要逻辑：
    1. 提交 run_probe.spawn() 到 Modal，或 reattach 到已有 function_call
    2. 等待云端结果（_wait_for_modal_result）
    3. 把结果写入 output_dir/aggregated_summary.json；如果用户没有指定
       output_dir，则写入 tmp/fi_probe/{model_name}_{timestamp}/probe/
    4. 打印摘要统计
    """
    function_call_id = resume_function_call_id.strip()
    if not function_call_id:
        if not model_name:
            raise ValueError("--model-name is required when submitting a new probe job")
        if tp_size <= 0:
            raise ValueError(
                "--tp-size must be a positive integer when submitting a new probe job"
            )

    import re

    # 先确定本地输出目录。即使本地等待过程断开，也能留下
    # probe_run.json，后续用里面的 function_call_id reattach。
    if output_dir:
        local_save_dir = Path(output_dir)
    else:
        run_label = model_name or f"resume_{function_call_id[:12]}"
        safe_model_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", run_label).strip("_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        local_save_dir = Path("tmp") / "fi_probe" / f"{safe_model_name}_{timestamp}" / "probe"
    local_save_dir.mkdir(parents=True, exist_ok=True)

    # ── 提交或 reattach ──
    if function_call_id:
        # resume 模式：本地之前已提交过，因网络中断等原因重新 attach
        print(f"🔁 Reattaching to Modal function call: {function_call_id}")
        function_call = modal.FunctionCall.from_id(function_call_id)
    else:
        # 正常模式：提交新任务（spawn = 异步提交，不阻塞）
        print(f"🚀 Submitting probe job to Modal for model: {model_name} ...")
        function_call = run_probe.spawn(
            model_name=model_name,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            tp_size=tp_size,
            probe_coverage=probe_coverage,
            probe_prefill_mode=probe_prefill_mode,
            probe_page_size=probe_page_size,
            probe_mem_fraction_static=probe_mem_fraction_static,
            force_flashinfer_backends=force_flashinfer_backends,
            preflight_only=preflight_only,
        )
        function_call_id = function_call.object_id
        print(f"📌 Modal function_call_id: {function_call_id}")
        # 打印 resume 命令，方便用户在网络断线后手动 reattach
        print(
            "If local polling is interrupted, rerun with "
            f"--resume-function-call-id {function_call_id}"
        )
    _update_local_probe_metadata(
        local_save_dir,
        status="submitted",
        function_call_id=function_call_id,
        model_name=model_name,
        tp_size=tp_size,
        probe_coverage=probe_coverage,
        probe_prefill_mode=probe_prefill_mode,
        probe_page_size=probe_page_size if probe_prefill_mode == "paged" and probe_page_size > 0 else None,
        force_flashinfer_backends=force_flashinfer_backends,
        preflight_only=preflight_only,
    )
    print(f"📎 Modal probe metadata saved to: {local_save_dir / 'probe_run.json'}")

    # ── 等待云端结果 ──
    try:
        result = _wait_for_modal_result(function_call, function_call_id)
    except Exception as exc:
        _update_local_probe_metadata(
            local_save_dir,
            status="local_polling_failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        print("\n" + "!" * 80)
        print("❌ Local polling failed before probe output was saved.")
        print("!" * 80)
        print(f"\nfunction_call_id: {function_call_id}")
        print(f"resume command: modal run scripts/probe/scheduler.py --resume-function-call-id {function_call_id} --output-dir {local_save_dir}")
        print(f"probe run metadata: {local_save_dir / 'probe_run.json'}")
        raise

    # ── 处理错误 ──
    if result.get("error"):
        _update_local_probe_metadata(
            local_save_dir,
            status="remote_failed",
            error_type=type(result["error"]).__name__,
            error=str(result["error"]),
        )
        print("\n" + "!" * 80)
        print("❌ Probe failed.")
        print("!" * 80)
        print("\n[Error Details]:")
        print(result["error"])
        raise SystemExit(1)

    # ── 保存结果到本地 ──
    # 默认路径格式：tmp/fi_probe/Qwen_Qwen3.5-35B-A3B_20260415_181234/probe/aggregated_summary.json。
    # run / 调试模式可传 --output-dir，让 summary 和后续产物放在同一个隔离目录。

    # 官方 fi_trace JSON 不塞进 aggregated_summary.json；单独落盘到
    # output_dir/fi_trace_out，后续 pipeline 会按 op_type stage 到 definitions/。
    fi_trace_definitions = result.pop("fi_trace_definitions", [])
    fi_trace_out_dir = local_save_dir / "fi_trace_out"
    written_fi_trace = 0
    if fi_trace_definitions:
        fi_trace_out_dir.mkdir(parents=True, exist_ok=True)
        for item in fi_trace_definitions:
            if item.get("error"):
                print(
                    f"⚠️  Skipping invalid fi_trace JSON {item.get('filename')}: {item['error']}",
                    flush=True,
                )
                continue
            definition = item.get("definition")
            filename = item.get("filename")
            if not isinstance(definition, dict) or not filename:
                continue
            (fi_trace_out_dir / filename).write_text(
                json.dumps(definition, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            written_fi_trace += 1

    result["fi_trace_definition_count"] = written_fi_trace
    if written_fi_trace:
        result["fi_trace_out_dir"] = str(fi_trace_out_dir)

    local_save_path = local_save_dir / "aggregated_summary.json"
    local_save_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    _update_local_probe_metadata(
        local_save_dir,
        status="succeeded",
        aggregated_summary=str(local_save_path),
        fi_trace_definition_count=written_fi_trace,
    )

    # ── 打印摘要 ──
    print("\n" + "=" * 80)
    print(f"✅ Probe finished. Results saved to: {local_save_path}")
    print(f"total_events:       {result.get('total_events', 0)}")
    print(f"unique_trace_ids:   {result.get('unique_trace_ids', 0)}")
    print(f"unique_signatures:  {result.get('unique_signatures', 0)}")
    print(f"fi_trace_defs:      {written_fi_trace}")

    if result.get("result_text"):
        print(f"\nresult_text: {result['result_text'][:300]}")

    print("\nTop 10 trace_ids preview (see JSON for all):")
    for item in result.get("top_trace_ids", [])[:10]:
        print(f"  {item['trace_id']}: {item['count']}")
    print("...")
    print("=" * 80)
