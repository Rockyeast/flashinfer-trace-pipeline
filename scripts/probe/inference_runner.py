"""在独立子进程中运行 SGLang 推理，驱动 probe 采集数据。

由 probe/scheduler.py 通过 subprocess 启动，不直接调用。
probe 通过 sitecustomize.py 从外部注入，本脚本对此透明——
它只是正常跑推理，probe hook 在 import sglang 时已悄悄装好。

状态通过 --status-path 文件实时上报给父进程（用于 watchdog 超时检测），
推理结果通过 --result-path 文件返回。
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path


def _log_phase(message: str) -> None:
    """打印带时间戳的阶段日志，flush=True 确保父进程能实时看到输出。"""
    print(f"[trace-phase {datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def _log_package_versions() -> None:
    """打印远程容器里的关键包版本和安装路径，用于确认实际运行环境。"""
    for name in ("sglang", "flashinfer", "flashinfer_bench"):
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "?")
            path = getattr(mod, "__file__", "?")
            _log_phase(f"package {name}: version={version} path={path}")
        except Exception as e:
            _log_phase(f"package {name}: import failed: {e}")


def _filter_supported_engine_kwargs(engine_kwargs: dict) -> dict:
    """过滤当前 SGLang 版本不支持的 Engine 参数，避免新旧镜像参数不兼容。"""
    try:
        import inspect
        from sglang.srt.server_args import ServerArgs

        supported = set(inspect.signature(ServerArgs).parameters)
    except Exception as e:
        _log_phase(f"engine_kwargs_filter:skip failed_to_inspect_server_args={e}")
        return engine_kwargs

    dropped = sorted(k for k in engine_kwargs if k not in supported)
    if dropped:
        _log_phase(f"engine_kwargs_filter:dropped unsupported={dropped}")
    return {k: v for k, v in engine_kwargs.items() if k in supported}


def _write_status(path: Path, phase: str, detail: str | None = None) -> None:
    """将当前阶段写入状态文件（JSON），父进程 watchdog 读取此文件判断子进程是否卡死。

    phase: 形如 "engine_init:start" / "generate:done" 的阶段标识字符串
    """
    payload = {"phase": phase, "ts": time.time()}
    if detail is not None:
        payload["detail"] = detail
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_result(path: Path, result_text: str | None, error: str | None) -> None:
    """将推理结果或错误信息写入结果文件（JSON），父进程读取后判断本次运行是否成功。"""
    path.write_text(
        json.dumps({"result_text": result_text, "error": error}, ensure_ascii=False),
        encoding="utf-8",
    )


def _probe_coverage(value: str | None = None) -> str:
    """返回 probe 覆盖档位。

    fast: 默认快速发现路径，少跑场景，省时间和费用。
    full: 更广覆盖路径，用于最终检查或怀疑 fast 漏 kernel 时。
    """
    coverage = (value or os.environ.get("PROBE_COVERAGE", "fast")).strip().lower()
    if coverage not in {"fast", "full"}:
        _log_phase(f"unknown PROBE_COVERAGE={coverage!r}; falling back to fast")
        return "fast"
    return coverage


def _probe_prefill_mode(value: str | None = None) -> str:
    """Return which SGLang prefill path this runner should try to exercise."""
    mode = (value or os.environ.get("PROBE_PREFILL_MODE", "default")).strip().lower()
    if mode not in {"default", "paged"}:
        _log_phase(f"unknown PROBE_PREFILL_MODE={mode!r}; falling back to default")
        return "default"
    return mode


def _probe_mem_fraction_static() -> float:
    """Return SGLang Engine mem_fraction_static for large-model probe runs."""
    raw = os.environ.get("PROBE_MEM_FRACTION_STATIC", "0.7").strip()
    try:
        value = float(raw)
    except ValueError:
        _log_phase(f"invalid PROBE_MEM_FRACTION_STATIC={raw!r}; falling back to 0.7")
        return 0.7
    if not 0.0 < value < 1.0:
        _log_phase(f"out-of-range PROBE_MEM_FRACTION_STATIC={value!r}; falling back to 0.7")
        return 0.7
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)          # HuggingFace 模型路径或名称
    parser.add_argument("--prompt", required=True)              # 推理 prompt，或 "__SHAREGPT__" 表示从文件读
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--tp-size", type=int, required=True)   # Tensor Parallelism 并行度
    parser.add_argument("--watchdog-timeout", type=int, required=True)  # Engine watchdog 超时秒数
    parser.add_argument("--mem-fraction-static", type=float, default=_probe_mem_fraction_static())
    parser.add_argument("--result-path", required=True)         # 结果文件路径
    parser.add_argument("--status-path", required=True)         # 状态文件路径（供父进程监控）
    parser.add_argument(
        "--probe-coverage",
        choices=["fast", "full"],
        default=os.environ.get("PROBE_COVERAGE", "fast"),
        help="Probe workload coverage profile: fast (default) or full.",
    )
    parser.add_argument(
        "--probe-prefill-mode",
        choices=["default", "paged"],
        default=os.environ.get("PROBE_PREFILL_MODE", "default"),
        help="Internal probe pass: default disables piecewise graph; paged enables it.",
    )
    parser.add_argument(
        "--probe-page-size",
        type=int,
        default=int(os.environ.get("PROBE_PAGE_SIZE", "0")),
        help=(
            "Optional SGLang Engine page_size for paged-prefill probe passes. "
            "Use 0 to omit page_size and let SGLang choose its runtime default."
        ),
    )
    parser.add_argument(
        "--force-flashinfer-backends",
        action="store_true",
        help="Force supported SGLang attention/sampling backend knobs to flashinfer.",
    )
    args = parser.parse_args()

    result_path = Path(args.result_path)
    status_path = Path(args.status_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 阶段 1：子进程启动 ──────────────────────────────────────────────────
    _write_status(status_path, "child:entered")
    _log_phase("child:entered")

    # ── 阶段 2：import sglang ───────────────────────────────────────────────
    # 此时 probe 的 _wrapped_import hook 触发，flashinfer 模块被自动 patch
    _write_status(status_path, "import:sglang:start")
    _log_phase("import:sglang:start")
    import sglang as sgl

    _write_status(status_path, "import:sglang:done")
    _log_phase("import:sglang:done")

    # ── 阶段 3：import torch ────────────────────────────────────────────────
    _write_status(status_path, "import:torch:start")
    _log_phase("import:torch:start")
    import torch

    _write_status(status_path, "import:torch:done")
    _log_phase("import:torch:done")

    # ── 阶段 4：打印环境信息 ────────────────────────────────────────────────
    _write_status(status_path, "env:probe:start")
    _log_phase("env:probe:start")
    print(f"model={args.model_name}", flush=True)
    print(f"tp_size={args.tp_size}", flush=True)
    print(f"torch={torch.__version__}", flush=True)
    _log_package_versions()
    print(f"gpu_count={torch.cuda.device_count()}", flush=True)
    if torch.cuda.is_available():
        print(f"gpu_0={torch.cuda.get_device_name(0)}", flush=True)
    _write_status(status_path, "env:probe:done")
    _log_phase("env:probe:done")

    llm = None
    result_text = None
    error = None
    try:
        # ── 阶段 5：初始化 SGLang Engine ────────────────────────────────────
        _write_status(status_path, "engine_init:start")
        _log_phase("engine_init:start")
        prefill_mode = _probe_prefill_mode(args.probe_prefill_mode)
        _log_phase(f"probe_prefill_mode={prefill_mode}")
        probe_page_size = args.probe_page_size if args.probe_page_size > 0 else None
        if prefill_mode == "paged":
            _log_phase(
                f"probe_page_size={probe_page_size if probe_page_size is not None else 'sglang_default'}"
            )
        mem_fraction_static = args.mem_fraction_static
        _log_phase(f"probe_mem_fraction_static={mem_fraction_static}")
        engine_kwargs = dict(
            model_path=args.model_name,
            trust_remote_code=True,
            dtype="bfloat16",
            attention_backend="flashinfer",
            disable_cuda_graph=True,  # 禁用 CUDA graph，确保 probe hook 在每次 kernel 调用时都能触发
            disable_piecewise_cuda_graph=prefill_mode != "paged",
            mem_fraction_static=mem_fraction_static,
            tp_size=args.tp_size,
            watchdog_timeout=args.watchdog_timeout,
        )
        if args.force_flashinfer_backends:
            engine_kwargs.update(
                prefill_attention_backend="flashinfer",
                decode_attention_backend="flashinfer",
                sampling_backend="flashinfer",
            )
        if prefill_mode == "paged":
            # Force SGLang's FlashInfer prefill dispatch away from the ragged wrapper.
            # flashinfer_backend.py computes use_ragged=False when deterministic
            # inference is enabled, which exposes BatchPrefillWithPagedKVCacheWrapper
            # forward/run calls instead of only its plan step.
            engine_kwargs["enable_deterministic_inference"] = True
        if prefill_mode == "paged" and probe_page_size is not None:
            engine_kwargs["page_size"] = probe_page_size
        engine_kwargs = _filter_supported_engine_kwargs(engine_kwargs)
        _log_phase(f"engine_kwargs={json.dumps(engine_kwargs, sort_keys=True)}")
        llm = sgl.Engine(**engine_kwargs)
        _write_status(status_path, "engine_init:done")
        _log_phase("engine_init:done")

        # 写入 inference 起始时间戳，聚合阶段用此过滤 warmup 期间的 event
        # （CUDA graph warmup 在 Engine init 期间发生，会触发 FlashInfer 调用，
        #  这些调用不代表真实推理路径，应从 kernel inventory 的 call_count 中排除）
        trace_dir = Path(os.environ.get("SGLANG_WORKER_TRACE_DIR", "/tmp/probe-output"))
        inference_ts_path = trace_dir / "inference_start_ts"
        inference_ts_path.write_text(str(time.time()), encoding="utf-8")
        _log_phase(f"inference_start_ts written to {inference_ts_path}")

        # ── 阶段 6：构造推理批次 ────────────────────────────────────────────
        _write_status(status_path, "generate:start")
        _log_phase("generate:start")

        # 支持两种 prompt 来源：
        # 1. "__SHAREGPT__"：从文件读取 100 条真实 ShareGPT 多轮对话数据
        # 2. 普通字符串：直接用命令行传入的 prompt
        if args.prompt == "__SHAREGPT__":
            _log_phase(
                "Loading authentic ShareGPT Vicuna unfiltered workload (100 multi-turn samples)..."
            )
            try:
                with open("/root/flashinfer-trace/scripts/fixtures/sharegpt_100.json", "r", encoding="utf-8") as f:
                    target_prompts = json.load(f)
                _log_phase(
                    f"Successfully loaded {len(target_prompts)} ShareGPT conversational prompts."
                )
            except Exception as e:
                _log_phase(f"Fallback to default due to IO error: {e}")
                target_prompts = "Explain quantum computing in one sentence."
        else:
            target_prompts = args.prompt

        if isinstance(target_prompts, list):
            def _build_sharegpt_scenarios(prompts: list[str], max_new_cap: int):
                ranked = sorted(prompts, key=len)
                n = len(ranked)
                coverage = _probe_coverage(args.probe_coverage)
                if coverage == "full":
                    specs = [
                        ("b1_long", 1, "long", 96),
                        ("b2_medium", 2, "medium", 64),
                        ("b4_medium", 4, "medium", 32),
                        ("b8_short", 8, "short", 32),
                        ("b16_short", 16, "short", 16),
                        ("b32_very_short", 32, "very_short", 8),
                        ("b64_very_short", 64, "very_short", 8),
                    ]
                else:
                    specs = [
                        ("b1_medium", 1, "medium", 8),
                        ("b8_short", 8, "short", 8),
                        ("b32_very_short", 32, "very_short", 4),
                    ]

                def _pool(bucket: str, batch_size: int) -> list[str]:
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

            prompt_scenarios = _build_sharegpt_scenarios(target_prompts, args.max_new_tokens)
        else:
            prompt_scenarios = [("single_prompt", target_prompts, args.max_new_tokens)]

        coverage = _probe_coverage(args.probe_coverage)
        _log_phase(f"probe_coverage={coverage}")
        # 分别触发 top_k+top_p / 纯 top_k / 纯 top_p。fast 和 full 都保留
        # 三组 sampling，避免 fast probe 漏掉 sampling 变体；fast 的提速来自
        # 减少 scenario 数量和 max_new_tokens，而不是减少 sampling 覆盖。
        sampling_configs = [
            {"temperature": 0.7, "top_k": 50, "top_p": 0.9},
            {"temperature": 0.7, "top_k": 50, "top_p": 1.0},
            {"temperature": 0.7, "top_k": -1, "top_p": 0.9},
        ]

        # ── 阶段 7：执行推理（sampling 配置 × 场景）─────────────────────────
        # probe 只负责发现候选 kernel；正式 workload 覆盖由 collect 阶段完成。
        output = []
        for cfg_idx, cfg in enumerate(sampling_configs, 1):
            _log_phase(f"sampling_config {cfg_idx}/{len(sampling_configs)}: {cfg}")
            for batch_idx, (scenario_name, batch, max_new_tokens) in enumerate(prompt_scenarios):
                n = len(batch) if isinstance(batch, list) else 1  # 单条 prompt 不是 list，计为 1
                run_cfg = {**cfg, "max_new_tokens": max_new_tokens}
                _log_phase(
                    f"  scenario {batch_idx+1}/{len(prompt_scenarios)}:"
                    f" {scenario_name}, {n} prompts, max_new_tokens={max_new_tokens}"
                )  # 上报进度给父进程 watchdog
                out = llm.generate(batch, sampling_params=run_cfg)  # 同步推理：内部触发 prefill/decode/sampling kernel，probe hook 在调用时截获
                output.append(out)  # 生成的文本内容对 probe 无意义，但需要保留引用确保调用完整执行

        _write_status(status_path, "generate:done")
        _log_phase("generate:done")

        result_text = str(output)  # 推理结果只用于确认跑完，内容本身不重要
    except Exception:
        # 推理过程中任何异常都捕获，写入 error 后在 finally 里上报给父进程
        _log_phase("run:error")
        error = traceback.format_exc()
    finally:
        # ── 阶段 8：关闭 Engine，写结果文件 ────────────────────────────────
        if llm is not None:
            try:
                _log_phase("shutdown:start")
                llm.shutdown()
                _log_phase("shutdown:done")
            except Exception:
                if error is None:
                    error = traceback.format_exc()
        time.sleep(2)  # 等待 TP worker 进程完全退出，确保 probe 数据全部落盘
        _write_result(result_path, result_text=result_text, error=error)


if __name__ == "__main__":
    main()
