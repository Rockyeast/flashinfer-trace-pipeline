#!/usr/bin/env python3
"""Workload collector using Python hooks instead of FLASHINFER_DUMP_DIR.

The collector injects sitecustomize.py, intercepts FlashInfer plan()/run()
calls, writes captured tensors to local /tmp inside the Modal container, and
post-processes them into JSONL workloads. It only collects kernels with
fi_api: tags; GEMM is intentionally not collected/submitted by default.

Usage:
    modal run scripts/collect_workloads_modal.py \\
        --model-path Qwen/Qwen3.5-35B-A3B \\
        --definitions gqa_paged_decode_h8_kv1_d256_ps1 gdn_decode_qk8_v16_d128 \\
        --tp 2

    modal run scripts/collect_workloads_modal.py \\
        --model-path deepseek-ai/DeepSeek-V3 \\
        --op-type sampling \\
        --tp 8
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import modal

from collect_workloads.batching import (
    merge_batch_diagnostics,
    parse_batch_sizes,
    rounds_for_batch_size,
)
from collect_workloads.definitions import (
    collectable_definition_names,
    load_definition_payloads,
)
from collect_workloads.hook_specs import (
    extract_official_fi_api,
    parse_def_specs,
)
from collect_workloads.sitecustomize import build_sitecustomize
from collect_workloads.workload_builder import process_captures
from artifact_schemas import validate_definition
from pipeline.runtime_defaults import (
    DEFAULT_EXTRA_PIP_PACKAGES,
    DEFAULT_FLASHINFER_BENCH_COLLECT_PACKAGE,
    DEFAULT_MODAL_GPU,
    DEFAULT_MODAL_GPU_COUNT,
    DEFAULT_SGLANG_IMAGE,
    DEFAULT_SGLANG_PACKAGE,
    forwarded_runtime_env,
)

app = modal.App("flashinfer-collect")

# 持久化 Volume，缓存 HuggingFace 模型权重，避免每次重新下载
model_cache = modal.Volume.from_name("flashinfer-model-cache", create_if_missing=True)
_MODEL_CACHE_DIR = "/root/hf_cache"

_GPU_TYPE = os.environ.get("MODAL_GPU", DEFAULT_MODAL_GPU)
_GPU_COUNT = int(os.environ.get("MODAL_GPU_COUNT", str(DEFAULT_MODAL_GPU_COUNT)))
_GPU_SPEC = f"{_GPU_TYPE}:{_GPU_COUNT}" if _GPU_COUNT > 1 else _GPU_TYPE
_SGLANG_IMAGE = os.environ.get("PROBE_SGLANG_IMAGE", DEFAULT_SGLANG_IMAGE).strip()
_SGLANG_PACKAGE = os.environ.get("PROBE_SGLANG_PACKAGE", DEFAULT_SGLANG_PACKAGE)
_FLASHINFER_VERSION = os.environ.get("PROBE_FLASHINFER_VERSION", "").strip()
_EXTRA_PIP_PACKAGES = shlex.split(
    os.environ.get("PROBE_EXTRA_PIP_PACKAGES", DEFAULT_EXTRA_PIP_PACKAGES)
)

def _build_collect_image() -> modal.Image:
    if _SGLANG_IMAGE:
        image = (
            modal.Image.from_registry(_SGLANG_IMAGE)
            .pip_install(DEFAULT_FLASHINFER_BENCH_COLLECT_PACKAGE)
            .pip_install("huggingface_hub>=0.34,<1.0", "hf_transfer", "safetensors", "pydantic>=2")
        )
        if _FLASHINFER_VERSION:
            image = image.pip_install(
                f"flashinfer-python=={_FLASHINFER_VERSION}",
                f"flashinfer-cubin=={_FLASHINFER_VERSION}",
            )
        if _EXTRA_PIP_PACKAGES:
            image = image.pip_install(*_EXTRA_PIP_PACKAGES)
    else:
        image = (
            modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
            .apt_install("libnuma1", "build-essential", "git")
            .pip_install(_SGLANG_PACKAGE)
            .pip_install(DEFAULT_FLASHINFER_BENCH_COLLECT_PACKAGE)
            .pip_install("nvidia-cudnn-cu12==9.16.0.29")
            .pip_install("huggingface_hub>=0.34,<1.0", "hf_transfer", "safetensors", "pydantic>=2")
        )
        if _FLASHINFER_VERSION:
            image = image.pip_install(
                f"flashinfer-python=={_FLASHINFER_VERSION}",
                f"flashinfer-cubin=={_FLASHINFER_VERSION}",
            )
        if _EXTRA_PIP_PACKAGES:
            image = image.pip_install(*_EXTRA_PIP_PACKAGES)
    base_env = {
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTHONFAULTHANDLER": "1",
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": _MODEL_CACHE_DIR,  # 让 HuggingFace 把模型缓存到 Volume 目录
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "HUGGING_FACE_HUB_TOKEN": os.environ.get(
            "HUGGING_FACE_HUB_TOKEN",
            os.environ.get("HF_TOKEN", ""),
        ),
        "PYTHONPATH": "/root/flashinfer-trace/scripts",
        "FLASHINFER_DISABLE_VERSION_CHECK": "1" if _FLASHINFER_VERSION else os.environ.get("FLASHINFER_DISABLE_VERSION_CHECK", ""),
        "SGLANG_DISABLE_CUDNN_CHECK": os.environ.get("SGLANG_DISABLE_CUDNN_CHECK", "1"),
    }
    base_env.update(forwarded_runtime_env())
    return image.env(base_env)


# Modal 容器镜像：安装 SGLang + FlashInfer + 辅助库
# add_local_dir/add_local_file：把本地的 definitions/ 和 hook 脚本打包进镜像
collect_image = (
    _build_collect_image()
    .add_local_dir("definitions", remote_path="/root/flashinfer-trace/definitions")
    .add_local_dir("scripts/collect_workloads", remote_path="/root/flashinfer-trace/scripts/collect_workloads")
    .add_local_dir("scripts/pipeline", remote_path="/root/flashinfer-trace/scripts/pipeline")
    .add_local_file("scripts/artifact_schemas.py", remote_path="/root/flashinfer-trace/scripts/artifact_schemas.py")
    .add_local_file("scripts/fixtures/sharegpt_100.json", remote_path="/root/flashinfer-trace/scripts/fixtures/sharegpt_100.json")
)


def _log(msg: str) -> None:
    print(f"[collect {datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


def _log_package_versions(prefix: str = "collect") -> None:
    """Print package versions and paths from inside the Modal container."""
    import importlib

    for name in ("sglang", "flashinfer", "flashinfer_bench"):
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "?")
            path = getattr(mod, "__file__", "?")
            print(f"[{prefix}] package {name}: version={version} path={path}", flush=True)
        except Exception as e:
            print(f"[{prefix}] package {name}: import failed: {e}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: 跑 SGLang 推理（18 轮），hook 通过 PYTHONPATH 注入
# ──────────────────────────────────────────────────────────────────────────────

# 18 轮推理配置：(batch_size, prompt_tokens, max_new_tokens)
# 覆盖不同 batch_size（1~64）和 prompt 长度（50~800 token），
# 让 hook 采集到尽可能多样化的 tensor shape 组合
_ROUNDS = [
    (1, 50, 8), (1, 300, 8), (1, 800, 8),
    (2, 50, 8), (2, 300, 8), (2, 800, 8),
    (4, 50, 8), (4, 300, 8), (4, 800, 8),
    (8, 50, 8), (8, 300, 8), (8, 800, 8),
    (16, 50, 8), (16, 300, 8), (16, 800, 8),
    (32, 50, 8), (32, 300, 8),
    (64, 50, 8),
]


@dataclass(frozen=True)
class RunnerConfig:
    """Configuration for one subprocess SGLang inference pass."""

    model_path: str
    tp: int
    page_size: int
    capture_dir: Path
    sitecustomize_dir: Path
    quantization: str | None = None
    cpu_offload_gb: float = 0.0
    rounds: list[tuple[int, int, int]] | None = None
    prompt_source: str = "sharegpt"
    sharegpt_path: str = "/root/flashinfer-trace/scripts/fixtures/sharegpt_100.json"
    sharegpt_max_new_tokens: int = 96
    cuda_graph_max_bs: int = -1
    dtype: str = ""
    mem_fraction_static: float = -1.0
    trust_remote_code: bool = False
    enable_piecewise_cuda_graph: bool = False
    force_flashinfer_backends: bool = False


@dataclass(frozen=True)
class WorkloadCollectConfig:
    """Configuration for one Modal workload collection run."""

    model_path: str
    definition_names: list[str]
    tp: int = 2
    page_size: int = 0
    cuda_graph_max_bs: int = -1
    dtype: str = ""
    mem_fraction_static: float = -1.0
    trust_remote_code: bool = False
    enable_piecewise_cuda_graph: bool = False
    force_flashinfer_backends: bool = False
    cpu_offload_gb: float = 0.0
    quantization: str | None = None
    replace: bool = False
    debug_hooks: bool = False
    output_dir: str = ""
    prompt_source: str = "sharegpt"
    sharegpt_max_new_tokens: int = 96
    streaming_collect: bool = True
    streaming_batch_sizes: list[int] | str | None = None
    workloads_per_batch: int = 4
    max_dups_per_axes: int = 2
    definition_payloads: dict[str, str] | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "WorkloadCollectConfig":
        """Build config from a Modal-serializable dict."""
        return cls(**payload)

    def to_payload(self) -> dict[str, Any]:
        """Return a Modal-serializable dict."""
        return asdict(self)

# SGLang 推理脚本（以字符串形式存储，运行时写到临时文件再执行）
# 独立为一个子进程的原因：SGLang 会 fork TP worker 进程，
# 子进程里的环境变量必须在 import sglang 之前设好，
# 所以通过 argv[1] 传 JSON payload，让 TP worker 继承 env 并读取 config
_SGLANG_RUNNER = r'''
import os, sys, json, time

# 在模块级别设置环境变量，确保 SGLang fork 出的 TP worker 进程能继承
_payload = json.loads(sys.argv[1])
_env = _payload["env"]
for k, v in _env.items():
    os.environ[k] = v

if __name__ == '__main__':
    cfg = _payload["config"]
    model_path = cfg["model_path"]
    tp = int(cfg["tp"])
    page_size = int(cfg["page_size"])
    rounds = cfg["rounds"]
    quant = cfg.get("quantization")
    cpu_offload = float(cfg["cpu_offload_gb"])
    prompt_source = cfg["prompt_source"]
    sharegpt_path = cfg["sharegpt_path"]
    sharegpt_max_tokens = int(cfg["sharegpt_max_new_tokens"])
    cuda_graph_max_bs = int(cfg["cuda_graph_max_bs"])
    dtype = cfg["dtype"]
    mem_fraction_static = float(cfg["mem_fraction_static"])
    trust_remote_code = bool(cfg["trust_remote_code"])
    enable_piecewise_cuda_graph = bool(cfg["enable_piecewise_cuda_graph"])
    force_flashinfer_backends = bool(cfg["force_flashinfer_backends"])

    import importlib
    import sglang  # hook 在 sitecustomize.py 里已经装好，这里 import 时 patch 自动生效

    for _pkg in ("sglang", "flashinfer", "flashinfer_bench"):
        try:
            _mod = importlib.import_module(_pkg)
            print(
                f"[collect-runner] package {_pkg}: "
                f"version={getattr(_mod, '__version__', '?')} "
                f"path={getattr(_mod, '__file__', '?')}",
                flush=True,
            )
        except Exception as _e:
            print(f"[collect-runner] package {_pkg}: import failed: {_e}", flush=True)

    engine_kwargs = dict(
        model_path=model_path,
        tp_size=tp,
        attention_backend="flashinfer",
        disable_cuda_graph=True,            # 禁普通 CUDA graph（旧机制）
        disable_piecewise_cuda_graph=not enable_piecewise_cuda_graph,
        mem_fraction_static=0.7,            # 留足 KV cache 空间
        log_level="info",
    )
    if force_flashinfer_backends:
        engine_kwargs.update(
            prefill_attention_backend="flashinfer",
            decode_attention_backend="flashinfer",
            sampling_backend="flashinfer",
        )
    if enable_piecewise_cuda_graph:
        # Paged-prefill collection must force SGLang's FlashInfer backend to
        # use BatchPrefillWithPagedKVCacheWrapper for execution, not just plan.
        engine_kwargs["enable_deterministic_inference"] = True
    if page_size > 0:
        engine_kwargs["page_size"] = page_size
    # -1 means omit the argument and use SGLang's default; 0 is a real value
    # used by older paged-prefill runs, so keep it reproducible.
    if cuda_graph_max_bs >= 0:
        engine_kwargs["cuda_graph_max_bs"] = cuda_graph_max_bs
    if quant:
        engine_kwargs["quantization"] = quant
    if cpu_offload > 0:
        engine_kwargs["cpu_offload_gb"] = cpu_offload
    if dtype:
        engine_kwargs["dtype"] = dtype
    if mem_fraction_static >= 0:
        engine_kwargs["mem_fraction_static"] = mem_fraction_static
    if trust_remote_code:
        engine_kwargs["trust_remote_code"] = True
    try:
        import inspect
        from sglang.srt.server_args import ServerArgs

        _supported = set(inspect.signature(ServerArgs).parameters)
        _dropped = sorted(k for k in engine_kwargs if k not in _supported)
        if _dropped:
            print(f"[collect-runner] engine_kwargs_filter:dropped unsupported={_dropped}", flush=True)
        engine_kwargs = {k: v for k, v in engine_kwargs.items() if k in _supported}
    except Exception as _e:
        print(f"[collect-runner] engine_kwargs_filter:skip failed_to_inspect_server_args={_e}", flush=True)

    print(f"[collect-runner] engine_kwargs={json.dumps(engine_kwargs, sort_keys=True)}", flush=True)
    engine = sglang.Engine(**engine_kwargs)

    # 18 个不同主题的 prompt，轮流使用，避免 KV cache 命中导致 prefill kernel 不触发
    _BASE   = ("You are an expert in GPU kernel optimization. Explain: ")
    _TOPICS = [
        "paged KV cache and memory fragmentation reduction",
        "tensor parallelism for multi-head attention",
        "FlashAttention tiling and IO-aware computation",
        "mixture-of-experts routing and load balancing",
        "RMSNorm versus LayerNorm computational differences",
        "speculative decoding draft model verification",
        "continuous batching in LLM serving systems",
        "quantization tradeoffs FP8 INT4 and GPTQ",
        "GQA grouped query attention and KV head sharing",
        "paged attention and vLLM memory management",
        "CUDA warp-level parallelism and shared memory",
        "multi-head latent attention MLA KV compression",
        "gated delta networks linear attention recurrence",
        "expert parallelism for MoE inference",
        "prefix caching and radix attention",
        "chunked prefill for decode-prefill balance",
        "ring attention and sequence parallelism",
        "FlashInfer batch decode wrapper plan and run",
    ]

    def _make_prompt(approx_tokens: int, idx: int) -> str:
        """生成约 approx_tokens 个 token 的 prompt（通过重复 topic 文字凑长度）。"""
        topic = _TOPICS[idx % len(_TOPICS)]
        text  = _BASE + topic
        target_chars = approx_tokens * 6  # 粗略估计：1 token ≈ 6 字符
        while len(text) < target_chars:
            text += " " + topic
        return text[:target_chars]

    try:
        if prompt_source == "sharegpt":
            with open(sharegpt_path, "r", encoding="utf-8") as f:
                target_prompts = json.load(f)
            ranked_prompts = sorted(target_prompts, key=len)

            def _build_sharegpt_scenarios(prompts, max_new_cap, requested_batch_sizes):
                ranked = ranked_prompts
                n = len(ranked)
                default_specs = [
                    ("b1_long", 1, "long", 96),
                    ("b2_medium", 2, "medium", 64),
                    ("b4_medium", 4, "medium", 32),
                    ("b8_short", 8, "short", 32),
                    ("b16_short", 16, "short", 16),
                    ("b32_very_short", 32, "very_short", 8),
                    ("b64_very_short", 64, "very_short", 8),
                ]
                default_by_batch = {spec[1]: spec for spec in default_specs}

                def _auto_spec(batch_size):
                    if batch_size <= 1:
                        return (f"b{batch_size}_long", batch_size, "long", 96)
                    if batch_size <= 4:
                        return (f"b{batch_size}_medium", batch_size, "medium", 64)
                    if batch_size <= 16:
                        return (f"b{batch_size}_short", batch_size, "short", 32)
                    return (f"b{batch_size}_very_short", batch_size, "very_short", 8)

                if requested_batch_sizes:
                    specs = [
                        default_by_batch.get(batch_size, _auto_spec(batch_size))
                        for batch_size in requested_batch_sizes
                    ]
                else:
                    specs = default_specs

                def _pool(bucket, batch_size):
                    if bucket == "very_short":
                        return ranked[:max(n // 4, 1)]
                    if bucket == "short":
                        return ranked[:max(batch_size, n // 2, 1)]
                    if bucket == "medium":
                        lo, hi = n // 4, max(n * 3 // 4, n // 4 + batch_size)
                        return ranked[lo:min(hi, n)] or ranked
                    return ranked[-max(batch_size, n // 4, 1):]

                scenarios = []
                for idx, (name, batch_size, bucket, max_new) in enumerate(specs):
                    pool = _pool(bucket, batch_size)
                    batch = [pool[(idx * 7 + i) % len(pool)] for i in range(batch_size)]
                    scenarios.append((name, batch, min(max_new, max_new_cap)))
                return scenarios

            requested_batch_sizes = sorted({int(r[0]) for r in rounds}) if rounds else []
            prompt_scenarios = _build_sharegpt_scenarios(
                target_prompts,
                sharegpt_max_tokens,
                requested_batch_sizes,
            )

            sampling_configs = [
                {"temperature": 0.7, "top_k": 50, "top_p": 0.9},
                {"temperature": 0.7, "top_k": 50, "top_p": 1.0},
                {"temperature": 0.7, "top_k": -1, "top_p": 0.9},
            ]

            for cfg_idx, cfg in enumerate(sampling_configs, 1):
                print(f"  sharegpt sampling_config={cfg_idx}/{len(sampling_configs)}: {cfg}", flush=True)
                for batch_idx, (scenario_name, prompts, max_new_tokens) in enumerate(prompt_scenarios, 1):
                    run_cfg = {**cfg, "max_new_tokens": max_new_tokens}
                    t0 = time.time()
                    outputs = engine.generate(prompt=prompts, sampling_params=run_cfg)
                    elapsed = time.time() - t0
                    n_ok = len(outputs) if outputs else 0
                    print(
                        f"    scenario={batch_idx}/{len(prompt_scenarios)} {scenario_name},"
                        f" batch_size={len(prompts):3d}, max_new_tokens={max_new_tokens}:"
                        f" {n_ok}/{len(prompts)} ok ({elapsed:.1f}s)",
                        flush=True,
                    )

            # ── Prefix-cache stress：更稳定触发真实 paged prefill ───────────────
            # Round 1 使用固定长 shared_prefix 建立 radix/prefix cache；
            # Round 2 使用完全相同的 shared_prefix 加不同 long suffix。
            # 这样新请求命中已分页的 prefix KV，同时对 suffix 做 extend/prefill，
            # 比把模型输出 resp 拼回 prompt 更容易走 BatchPrefillWithPagedKVCacheWrapper。
            def _make_long_suffix(approx_tokens, idx):
                topic = _TOPICS[(idx * 5 + 3) % len(_TOPICS)]
                text = (
                    f"\n\nSuffix request {idx}: compare implementation details, "
                    f"failure modes, and benchmark implications for {topic}."
                )
                target_chars = approx_tokens * 6
                while len(text) < target_chars:
                    text += " " + topic
                return text[:target_chars]

            print("  [prefix-cache] running shared-prefix + long-suffix extend...", flush=True)
            long_pool = ranked_prompts[-max(len(ranked_prompts) // 4, 4):]
            mt_batch_sizes = [bs for bs in requested_batch_sizes if bs <= 64] or [1]
            for mt_bs in mt_batch_sizes[:3]:   # 最多跑 3 种 batch size，避免耗时过长
                shared_source = long_pool[(mt_bs * 3) % len(long_pool)]
                shared_prefix = (
                    shared_source.rstrip()
                    + "\n\nShared cached prefix marker. The following requests reuse this exact prefix."
                )

                # Round 1：建立 prefix/radix cache。只生成 1 个 token，重点是缓存 prompt KV。
                t0 = time.time()
                seed_out = engine.generate(
                    prompt=[shared_prefix],
                    sampling_params={"temperature": 0.0, "max_new_tokens": 1},
                )
                seed_elapsed = time.time() - t0

                # Round 2：同一 shared_prefix + 较长 suffix，期望触发 paged prefill/extend。
                suffix_tokens = 192 if mt_bs <= 4 else 96
                mt_prompts_r2 = [
                    shared_prefix + _make_long_suffix(suffix_tokens, mt_bs * 100 + i)
                    for i in range(mt_bs)
                ]
                t0 = time.time()
                mt_out2 = engine.generate(
                    prompt=mt_prompts_r2,
                    sampling_params={"temperature": 0.0, "max_new_tokens": 1},
                )
                elapsed = time.time() - t0
                print(
                    f"    [prefix-cache] bs={mt_bs} seed={len(seed_out or [])} ok"
                    f" ({seed_elapsed:.1f}s), extend={len(mt_out2 or [])} ok"
                    f" ({elapsed:.1f}s), suffix~{suffix_tokens}t",
                    flush=True,
                )
        else:
            prompt_idx = 0
            for B, prompt_tokens, max_tokens in rounds:
                # 构造 B 条 prompt，依次使用不同 topic 避免重复
                prompts = [_make_prompt(prompt_tokens, prompt_idx + i) for i in range(B)]
                prompt_idx += B
                t0 = time.time()
                outputs = engine.generate(
                    prompt=prompts,
                    sampling_params={"max_new_tokens": max_tokens, "temperature": 0.7, "top_k": 50, "top_p": 0.9},
                )
                elapsed = time.time() - t0
                n_ok = len(outputs) if outputs else 0
                print(
                    f"  batch_size={B:2d}, prompt~{prompt_tokens:4d}t,"
                    f" max_tokens={max_tokens}: {n_ok}/{B} ok ({elapsed:.1f}s)",
                    flush=True,
                )
    finally:
        engine.shutdown()
'''


def _run_sglang_with_hooks(config: RunnerConfig) -> None:
    """把 sitecustomize_dir 加到 PYTHONPATH，以子进程方式启动 SGLang 推理。

    子进程启动时 Python 自动执行 sitecustomize.py → hook 装好
    → 推理过程中 FlashInfer plan()/run() 被 hook 截获 → .pt 文件写到 capture_dir
    """
    rounds = config.rounds or _ROUNDS

    # 继承当前环境变量，把 sitecustomize_dir 加到 PYTHONPATH 最前面
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(config.sitecustomize_dir) + (":" + existing_pp if existing_pp else "")
    )
    env["FLASHINFER_USE_CUDA_NORM"] = "1"  # 避免旧版 CUDA 上 CuTe DSL 问题
    # 必须在 subprocess env 里直接设置 TORCHDYNAMO_DISABLE，
    # 确保 Python 启动时（在 import torch 之前）就已经生效。
    # 仅放在 runner_env（argv[1] JSON）里太晚——sitecustomize 阶段如果触发了
    # 任何间接 torch import，dynamo 就已经初始化了。
    env["TORCHDYNAMO_DISABLE"] = "1"

    # TP worker 子进程通过 argv[1] JSON 接收需要继承的环境变量
    runner_env = {
        "FLASHINFER_USE_CUDA_NORM": "1",
        "PYTHONPATH": env["PYTHONPATH"],
        "FI_CAPTURE_DIR": str(config.capture_dir),
        "TORCHDYNAMO_DISABLE": "1",
    }
    if config.enable_piecewise_cuda_graph:
        runner_env["SGLANG_FLASHINFER_USE_PAGED"] = "1"
    # 把 HF token 等必要的环境变量透传给子进程
    for key in ("HUGGING_FACE_HUB_TOKEN", "HF_TOKEN", "HF_HOME"):
        if key in env:
            runner_env[key] = env[key]

    # 把 _SGLANG_RUNNER 字符串写成临时 .py 文件再执行
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(_SGLANG_RUNNER)
        script_path = f.name

    try:
        runner_payload = {
            "env": runner_env,
            "config": {
                "model_path": config.model_path,
                "tp": config.tp,
                "page_size": config.page_size,
                "rounds": rounds,
                "quantization": config.quantization,
                "cpu_offload_gb": config.cpu_offload_gb,
                "prompt_source": config.prompt_source,
                "sharegpt_path": config.sharegpt_path,
                "sharegpt_max_new_tokens": config.sharegpt_max_new_tokens,
                "cuda_graph_max_bs": config.cuda_graph_max_bs,
                "dtype": config.dtype,
                "mem_fraction_static": config.mem_fraction_static,
                "trust_remote_code": config.trust_remote_code,
                "enable_piecewise_cuda_graph": config.enable_piecewise_cuda_graph,
                "force_flashinfer_backends": config.force_flashinfer_backends,
            },
        }
        cmd = [sys.executable, script_path, json.dumps(runner_payload)]
        import subprocess as _sp
        _sp.run(cmd, check=True, env=env)
    finally:
        os.unlink(script_path)  # 清理临时文件


# ──────────────────────────────────────────────────────────────────────────────
# Modal 主函数
# ──────────────────────────────────────────────────────────────────────────────

@app.function(
    image=collect_image,
    gpu=_GPU_SPEC,
    timeout=7200,
    volumes={_MODEL_CACHE_DIR: model_cache},  # 挂载 Volume，模型下载后持久保存，下次直接用缓存
    secrets=[modal.Secret.from_name("huggingface-secret")] if os.environ.get("USE_HF_SECRET") else [],
)
def run_collection(config_payload: dict[str, Any]) -> dict:
    """在 Modal GPU 容器里运行 workload 采集（Python hook 方式）。

    流程：
      1. 解析 definition 文件 → 生成 hook 配置
      2. 写 sitecustomize.py（含 hook 注入代码）
      3. 子进程按 batch_size 分轮跑 SGLang 推理（hook 拦截 FlashInfer 调用 → .pt 文件）
      4. 每轮后处理 .pt → append JSONL + safetensors
      5. 把结果文件读回内存返回给本地（JSONL 用文本，safetensors 用 base64）

    output_dir: 产物输出目录（JSONL + safetensors），不影响 definitions 读取路径。
                为空时写到 /root/flashinfer-trace（默认行为，与改动前一致）。
    definition_payloads: 本地读取到的 definition JSON 内容。传入时优先使用这些内容，
                         避免 Modal 远端只看到镜像里固定打包的 definitions/。
    """
    config = WorkloadCollectConfig.from_payload(config_payload)
    model_path = config.model_path
    definition_names = config.definition_names
    tp = config.tp
    page_size = config.page_size
    cuda_graph_max_bs = config.cuda_graph_max_bs
    dtype = config.dtype
    mem_fraction_static = config.mem_fraction_static
    trust_remote_code = config.trust_remote_code
    enable_piecewise_cuda_graph = config.enable_piecewise_cuda_graph
    force_flashinfer_backends = config.force_flashinfer_backends
    cpu_offload_gb = config.cpu_offload_gb
    quantization = config.quantization
    replace = config.replace
    debug_hooks = config.debug_hooks
    output_dir = config.output_dir
    prompt_source = config.prompt_source
    sharegpt_max_new_tokens = config.sharegpt_max_new_tokens
    streaming_collect = config.streaming_collect
    streaming_batch_sizes = config.streaming_batch_sizes
    definition_payloads = config.definition_payloads

    _log_package_versions()
    workloads_per_batch = int(config.workloads_per_batch)
    max_dups_per_axes = max(1, int(config.max_dups_per_axes))

    if definition_payloads:
        defs_dir = Path(tempfile.mkdtemp(prefix="fi-definitions-"))
        for name, content in definition_payloads.items():
            try:
                defn = json.loads(content)
                validate_definition(defn)
            except json.JSONDecodeError as e:
                return {"status": "error", "message": f"Invalid definition JSON for {name}: {e}"}
            except Exception as e:
                return {"status": "error", "message": f"Invalid definition schema for {name}: {e}"}
            op_type = defn.get("op_type", "unknown")
            out = defs_dir / "definitions" / op_type / f"{name}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(content)
        _log(f"Using {len(definition_payloads)} definition payloads from local --definitions-dir")
    else:
        defs_dir = Path("/root/flashinfer-trace")        # fallback：镜像里打包的默认 definitions/
    trace_dir = Path(output_dir) if output_dir else defs_dir  # 产物输出到 output_dir 或默认路径

    # 查找 definition 文件
    def_files = []
    missing = []
    for name in definition_names:
        matches = list(defs_dir.glob(f"definitions/**/{name}.json"))
        if matches:
            def_files.append(matches[0])
        else:
            missing.append(name)

    if missing:
        _log(f"WARNING: missing definitions: {missing}")
    if not def_files:
        return {"status": "error", "message": "No valid definitions found"}

    # Collect workloads only for real FlashInfer APIs. target_api definitions
    # may be generated for diagnostics, but they are not part of collect.
    fi_def_files = []
    skipped = []
    for path in def_files:
        defn = json.loads(path.read_text())
        validate_definition(defn)
        has_hook_api = extract_official_fi_api(defn) is not None
        if has_hook_api:
            fi_def_files.append(path)
        else:
            skipped.append(path.stem)

    if skipped:
        _log(f"Skipping {len(skipped)} definitions without fi_api:flashinfer.* tag: {skipped}")
    if not fi_def_files:
        return {"status": "error", "message": "No hookable API definitions to collect"}

    # 过滤：跳过 axes/inputs shape 里含有 "?" 的 definition
    # "?" 表示 probe 阶段没有捕获到该参数的实际值，强行传给 SGLang 会导致错误
    # 且会牵连同批次其他 kernel 的采集，提前隔离是最安全的做法
    unresolved = []
    valid_fi_def_files = []
    for path in fi_def_files:
        defn = json.loads(path.read_text())
        validate_definition(defn)
        axes_vals = list(defn.get("axes", {}).values())
        input_shapes = [
            dim
            for inp in defn.get("inputs", {}).values()
            for dim in (inp.get("shape") or [])
        ]
        has_question = any("?" in str(v) for v in axes_vals + input_shapes)
        if has_question:
            unresolved.append(path.stem)
            _log(f"WARNING: Skipping {path.stem} — unresolved '?' in axes/inputs "
                 f"(probe did not capture required params). "
                 f"axes={defn.get('axes', {})}")
        else:
            valid_fi_def_files.append(path)

    if unresolved:
        _log(f"Skipped {len(unresolved)} definitions with unresolved params: {unresolved}")
    fi_def_files = valid_fi_def_files

    if not fi_def_files:
        return {"status": "error", "message": "No valid (fully-resolved) hookable API definitions to collect"}

    _log(f"Collecting {len(fi_def_files)} definitions: {[f.stem for f in fi_def_files]}")
    _log(f"Prompt source: {prompt_source}")

    # ── Step 1: 解析 definition 文件 → hook specs ─────────────────────────────
    # parse_def_specs() 读取每个 definition 的 inputs/tags，
    # 判断 is_wrapper / needs_plan，生成 {fi_api: spec} 配置字典
    specs = parse_def_specs(fi_def_files)
    _log(f"Hook specs: {list(specs.keys())}")

    # 使用容器本地 NVMe（/tmp），读写速度快，不走网络
    sitecustomize_dir = Path("/tmp/collect-sitecustomize")
    sitecustomize_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

    base_rounds = _ROUNDS[:2] if debug_hooks else _ROUNDS
    results: dict[str, int] = defaultdict(int)

    def _run_capture_pass(
        *,
        label: str,
        rounds: list[tuple[int, int, int]],
        replace_outputs: bool,
        max_entries: int,
    ) -> dict:
        """Run one inference/sanitize pass and return process diagnostics."""
        capture_dir = Path(tempfile.mkdtemp(prefix=f"fi_captures_{label}_"))
        try:
            # ── Step 2: 生成 sitecustomize.py，注入 hook ──────────────────────
            sc_path = sitecustomize_dir / "sitecustomize.py"
            sc_path.write_text(build_sitecustomize(specs, str(capture_dir), debug=debug_hooks))
            _log(f"sitecustomize.py written to {sc_path} -> {capture_dir} (debug_hooks={debug_hooks})")

            # ── Step 3: 跑 SGLang 推理，hook 拦截 FlashInfer 调用 → 写 .pt 文件 ──
            _log(f"Starting SGLang inference pass {label} ({len(rounds)} round(s))...")
            t0 = time.time()
            _run_sglang_with_hooks(RunnerConfig(
                model_path=model_path,
                tp=tp,
                page_size=page_size,
                capture_dir=capture_dir,
                sitecustomize_dir=sitecustomize_dir,
                quantization=quantization,
                cpu_offload_gb=cpu_offload_gb,
                rounds=rounds,
                prompt_source=prompt_source,
                sharegpt_max_new_tokens=sharegpt_max_new_tokens,
                cuda_graph_max_bs=cuda_graph_max_bs,
                dtype=dtype,
                mem_fraction_static=mem_fraction_static,
                trust_remote_code=trust_remote_code,
                enable_piecewise_cuda_graph=enable_piecewise_cuda_graph,
                force_flashinfer_backends=force_flashinfer_backends,
            ))
            _log(f"Inference pass {label} done in {time.time() - t0:.1f}s")

            # ── Step 4: 本轮后处理 .pt → append JSONL + safetensors ────────────
            _log(f"Processing captures for pass {label}...")
            return process_captures(
                capture_dir=capture_dir,
                def_files=fi_def_files,
                trace_dir=trace_dir,
                replace=replace_outputs,
                max_entries=max_entries,
                max_dups_per_axes=max_dups_per_axes,
                log=_log,
            )
        finally:
            shutil.rmtree(capture_dir, ignore_errors=True)

    try:
        if streaming_collect:
            try:
                batch_sizes = parse_batch_sizes(streaming_batch_sizes) or [1, 2, 4, 8, 16, 32, 64]
            except ValueError as e:
                return {"status": "error", "message": str(e)}
            diagnostics: dict = {
                "_streaming": {
                    "enabled": True,
                    "batch_sizes": batch_sizes,
                    "workloads_per_batch": workloads_per_batch,
                    "max_dups_per_axes": max_dups_per_axes,
                    "prompt_source": prompt_source,
                }
            }
            _log(
                "Streaming collect enabled: "
                f"batch_sizes={batch_sizes}, workloads_per_batch={workloads_per_batch}, "
                f"max_dups_per_axes={max_dups_per_axes}"
            )
            for batch_index, batch_size in enumerate(batch_sizes, 1):
                batch_rounds = rounds_for_batch_size(base_rounds, batch_size)
                process_result = _run_capture_pass(
                    label=f"bs{batch_size}",
                    rounds=batch_rounds,
                    replace_outputs=replace and batch_index == 1,
                    max_entries=workloads_per_batch,
                )
                for def_name, count in process_result["counts"].items():
                    results[def_name] += count
                merge_batch_diagnostics(
                    diagnostics,
                    batch_size=batch_size,
                    batch_diagnostics=process_result["diagnostics"],
                )
        else:
            diagnostics = {
                "_streaming": {
                    "enabled": False,
                    "max_dups_per_axes": max_dups_per_axes,
                    "prompt_source": prompt_source,
                }
            }
            _log(
                "Single-pass collect enabled: "
                "max_entries=unlimited, "
                f"max_dups_per_axes={max_dups_per_axes}"
            )
            process_result = _run_capture_pass(
                label="single",
                rounds=base_rounds,
                replace_outputs=replace,
                max_entries=0,
            )
            results.update(process_result["counts"])
            diagnostics.update(process_result["diagnostics"])
    except Exception as e:
        return {"status": "error", "message": f"SGLang inference or capture processing failed: {e}"}

    results = dict(results)
    total_workloads = sum(results.values())
    _log(f"Done. {total_workloads} workloads across {len(results)} definitions.")

    # 把生成的文件读回内存，通过 Modal 返回值传给本地
    # JSONL 用文本传输，safetensors 二进制用 base64 编码
    import base64
    workload_files: dict[str, str] = {}
    blob_files: dict[str, str] = {}
    workloads_dir = trace_dir / "workloads"
    blob_dir_root = trace_dir / "blob"
    if workloads_dir.exists():
        for jsonl in workloads_dir.rglob("*.jsonl"):
            key = str(jsonl.relative_to(trace_dir))
            workload_files[key] = jsonl.read_text()
    if blob_dir_root.exists():
        for st in blob_dir_root.rglob("*.safetensors"):
            key = str(st.relative_to(trace_dir))
            blob_files[key] = base64.b64encode(st.read_bytes()).decode()

    return {
        "status": "success",
        "definitions_collected": list(results.keys()),
        "definitions_skipped": skipped,
        "definitions_missing": missing,
        "workload_counts": results,
        "collector_diagnostics": diagnostics,
        "total_workloads": total_workloads,
        "workload_files": workload_files,   # {相对路径: jsonl 文本}
        "blob_files": blob_files,           # {相对路径: base64 编码的 safetensors 字节}
    }


@app.local_entrypoint()
def main(
    model_path: str = "Qwen/Qwen3.5-35B-A3B",
    definitions: str = "",      # 逗号分隔的 definition 名列表
    op_type: str = "",          # 按 op_type 目录批量处理
    definitions_dir: str = "definitions",
    tp: int = 2,
    page_size: int = 0,
    cuda_graph_max_bs: int = -1,
    dtype: str = "",
    mem_fraction_static: float = -1.0,
    trust_remote_code: bool = False,
    enable_piecewise_cuda_graph: bool = False,
    force_flashinfer_backends: bool = False,
    cpu_offload_gb: float = 0.0,
    quantization: str = "",
    replace: bool = False,      # True 则覆盖已有 JSONL，False 则追加
    debug_hooks: bool = False,  # True 则只跑 2 轮推理，用于调试 hook
    output_dir: str = "",       # 产物输出目录，为空则写到仓库默认位置（workloads/ 和 blob/）
    prompt_source: str = "sharegpt",
    sharegpt_max_new_tokens: int = 96,
    streaming_collect: bool = True,
    streaming_batch_sizes: str = "1,2,4,8,16,32,64",
    workloads_per_batch: int = 4,
    max_dups_per_axes: int = 2,
):
    """本地入口：解析 definition 列表，派发到 Modal GPU 执行采集，把结果写回本地。"""
    if prompt_source not in ("sharegpt", "synthetic"):
        print("ERROR: --prompt-source must be 'sharegpt' or 'synthetic'", file=sys.stderr)
        sys.exit(1)

    definitions_root = Path(definitions_dir)

    def_names: list[str] = []

    if definitions:
        # 直接指定 definition 名（逗号分隔）
        def_names = [d.strip() for d in definitions.split(",") if d.strip()]

    elif op_type:
        # 按 op_type 目录批量处理该类型下所有 collectable FlashInfer definition
        defs_dir = definitions_root / op_type
        if not defs_dir.exists():
            print(f"ERROR: no definitions directory for op_type: {op_type}", file=sys.stderr)
            sys.exit(1)
        def_names = collectable_definition_names(definitions_root, op_type)

    else:
        print("ERROR: specify --definitions or --op-type", file=sys.stderr)
        sys.exit(1)

    if not def_names:
        print("No definitions to collect.")
        return

    definition_payloads, missing_local_defs = load_definition_payloads(definitions_root, def_names)
    if missing_local_defs:
        print(f"WARNING: missing definitions under {definitions_root}: {missing_local_defs}", file=sys.stderr)
    if not definition_payloads:
        print(f"ERROR: no definition JSONs found under {definitions_root}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print("Workload Collection (Python hooks)")
    print(f"{'='*60}")
    print(f"Model:       {model_path}")
    print(f"TP:          {tp}")
    print(f"Prompts:     {prompt_source}")
    print(f"Piecewise CUDA graph: {'enabled' if enable_piecewise_cuda_graph else 'disabled'}")
    print(f"Force FlashInfer backends: {'enabled' if force_flashinfer_backends else 'disabled'}")
    print(f"Streaming:   {streaming_collect}")
    if streaming_collect:
        print(f"Batches:     {streaming_batch_sizes}")
        print(f"Per batch:   {workloads_per_batch}")
    else:
        print("Max/def:     (none)")
    print(f"Max dup axes:{max_dups_per_axes}")
    print(f"Output:      {output_dir or '(default: workloads/ and blob/)'}")
    print(f"Definitions dir: {definitions_root}")
    print(f"Definitions: {len(def_names)}")
    for d in def_names:
        print(f"  - {d}")
    print(f"{'='*60}\n")

    config = WorkloadCollectConfig(
        model_path=model_path,
        definition_names=def_names,
        tp=tp,
        page_size=page_size,
        cuda_graph_max_bs=cuda_graph_max_bs,
        dtype=dtype,
        mem_fraction_static=mem_fraction_static,
        trust_remote_code=trust_remote_code,
        enable_piecewise_cuda_graph=enable_piecewise_cuda_graph,
        force_flashinfer_backends=force_flashinfer_backends,
        cpu_offload_gb=cpu_offload_gb,
        quantization=quantization or None,
        replace=replace,
        debug_hooks=debug_hooks,
        output_dir=output_dir,
        prompt_source=prompt_source,
        sharegpt_max_new_tokens=sharegpt_max_new_tokens,
        streaming_collect=streaming_collect,
        streaming_batch_sizes=streaming_batch_sizes,
        workloads_per_batch=workloads_per_batch,
        max_dups_per_axes=max_dups_per_axes,
        definition_payloads=definition_payloads,
    )
    result = run_collection.remote(config.to_payload())

    # 打印摘要（不含大文件内容字段）
    display = {k: v for k, v in result.items() if k not in ("workload_files", "blob_files")}
    print(f"\n{'='*60}")
    print(json.dumps(display, indent=2, ensure_ascii=False))
    print(f"{'='*60}")

    if result.get("status") == "success":
        # 把 Modal 容器里生成的文件写到本地
        import base64 as _b64
        local_base = Path(output_dir) if output_dir else Path(".")
        diagnostics = result.get("collector_diagnostics")
        if diagnostics:
            diag_path = local_base / "collector_diagnostics.json"
            diag_path.parent.mkdir(parents=True, exist_ok=True)
            diag_path.write_text(
                json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"Wrote diagnostics: {diag_path}")
        written = []
        for rel_path, content in result.get("workload_files", {}).items():
            out = local_base / rel_path
            out.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if (out.exists() and not replace) else "w"
            with open(out, mode) as f:
                f.write(content)
            written.append(str(out))
        for rel_path, b64 in result.get("blob_files", {}).items():
            out = local_base / rel_path
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(_b64.b64decode(b64))
            written.append(str(out))
        if written:
            print(f"\nWritten locally ({len(written)} files):")
            for w in written[:10]:
                print(f"  {w}")
            if len(written) > 10:
                print(f"  ... and {len(written) - 10} more")
        print(f"\n✅ Done! {result['total_workloads']} workloads collected.")
    else:
        print(f"\n❌ Failed: {result.get('message', 'unknown error')}")
        sys.exit(1)
