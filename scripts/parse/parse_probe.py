#!/usr/bin/env python3
"""Parse probe-mode aggregated_summary.json into a structured kernel inventory.

This is Step 2 of the v2 pipeline. It takes the raw probe output and extracts
a clean list of kernel definitions that the model actually uses at runtime.

中文导览：
    这个脚本是 parse 阶段的总入口。它不重新跑模型，只读取 probe 产出的
    aggregated_summary.json，然后把“原始 trace 证据”压缩成 kernel_inventory.json。

    你可以把它理解成：

        summary 里的 trace_id/signature
            -> 分类成 op_type / variant
            -> 从 signature 抽关键参数
            -> 调 adapter 生成 definition_name / params / fi_api
            -> 组装 kernels / deferred / unmatched / LLM diagnostics
            -> 写出 kernel_inventory.json

    注意：
        probe 阶段现在只记录事实；正式分类主要来源是 scripts/adapters/。

Usage:
    python scripts/parse/parse_probe.py \
        tmp/run/Qwen_Qwen3.5-35B-A3B_.../probe/aggregated_summary.json \
        --model-name Qwen/Qwen3.5-35B-A3B \
        --definitions-dir definitions/

    # With HuggingFace config for cross-validation
    python scripts/parse/parse_probe.py \
        tmp/run/Qwen_Qwen3.5-35B-A3B_.../probe/aggregated_summary.json \
        --model-name Qwen/Qwen3.5-35B-A3B \
        --hf-config path/to/config.json

Output: kernel_inventory.json in the same directory as the input file.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from adapters import (  # noqa: E402
    build_kernels as build_adapter_kernels,
    classify_trace_id as classify_adapter_trace_id,
)
from parse.config_enrichment import enrich_with_hf_config  # noqa: E402
from parse.diagnostics import (  # noqa: E402
    detect_model_backend,
    ignored_noise_reason,
    split_deferred_kernels,
)
from parse.inventory_helpers import (  # noqa: E402
    KERNEL_STATE_EXISTING,
    KERNEL_STATE_NEEDS_CONFIG,
    KERNEL_STATE_NEW,
    dedup_kernels_by_definition_name,
    get_kernel_state,
    is_existing_kernel,
    is_new_kernel,
    merge_matched_kernel,
    needs_config_kernel,
    observed_fi_api_from_trace_id,
    set_kernel_state,
)
from parse.rules import classify_trace_id  # noqa: E402


# 这个文件只做“总控”和“数据流编排”。
# 具体职责已经拆到 scripts/parse/：
#
#   api.py                 fi_api / target_api 边界规则
#   adapters/              新主线：单个 op_type 的 classify/build/generate 适配器
#   rules.py               少量人工确认的例外分类规则
#   adapters/extractors.py 从 signature 里抽 hidden_size/head_dim 等参数
#   config_enrichment.py   用 HF config 补 signature 里缺的静态维度
#   diagnostics.py         整理 noise/unmatched/deferred/LLM 分类诊断
#   inventory_helpers.py   合并/去重/规范化最终 kernel entry


# ──────────────────────────────────────────────────────────────────────────────
# Lazy import helper for LLM classifier (no hard anthropic dependency)
# ──────────────────────────────────────────────────────────────────────────────


def _get_llm_classifier():
    """Lazy-import llm_classify to avoid hard anthropic dependency."""
    # 延迟加载 llm_classify.py，只有真正需要 LLM 分类时才导入
    # 这样不安装 anthropic 包的用户也能正常使用其他功能
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "llm_classify",
            Path(__file__).parent / "llm_classify.py",
        )
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod
    except Exception as exc:  # noqa: BLE001
        print(f"  [1b] Cannot load llm_classify.py: {exc}", file=sys.stderr)
        return None



# ──────────────────────────────────────────────────────────────────────────────
# Main parser
# ──────────────────────────────────────────────────────────────────────────────


def parse_probe(
    summary_path: Path,
    model_name: str,
    definitions_dir: Path | None = None,
    hf_config: dict | None = None,
    llm_classify: bool = False,
    llm_model: str = "claude-sonnet-4-6",
    llm_base_url: str | None = None,
) -> dict:
    """Parse aggregated_summary.json and return kernel inventory.

    Args:
        summary_path: Path to aggregated_summary.json.
        model_name: Model identifier (e.g. "Qwen/Qwen3.5-35B-A3B").
        definitions_dir: Where existing definitions live (for inventory_status checks).
        hf_config: HuggingFace config dict for dimension resolution.
        llm_classify: If True, unknown trace_ids are sent to the LLM for
            diagnostic suggestions instead of being silently dropped. LLM
            suggestions are not promoted into formal kernels automatically.
        llm_model: Anthropic model ID for classification.
        llm_base_url: Optional proxy base URL for the Anthropic-compatible API.

    解析 aggregated_summary.json，返回 kernel 清单（字典）。

    参数说明：
        summary_path: aggregated_summary.json 的路径。
        model_name: 模型标识符（如 "Qwen/Qwen3.5-35B-A3B"）。
        definitions_dir: 已有 definition 文件所在目录（用于检查 definition 是否已存在）。
        hf_config: HuggingFace 模型配置字典，用于补全维度信息。
        llm_classify: 为 True 时，未知 trace_id 会发给 LLM 生成诊断建议；
            LLM 建议不会自动进入正式 kernels。
        llm_model: 用于分类的 Anthropic 模型 ID。
        llm_base_url: 可选的 Anthropic 兼容 API 代理地址。
    """

    # ──────────────────────────────────────────────────────────────────────
    # 1. 读取 summary，并建立两个最常用索引
    # ──────────────────────────────────────────────────────────────────────
    # aggregated_summary.json 里最重要的是：
    #   top_trace_ids:   trace_id -> 调用次数
    #   top_signatures:  trace_id -> 参数 shape/dtype/attrs 样本
    #
    # 后面分类主要看 trace_id，参数提取主要看 signatures。
    with open(summary_path, encoding="utf-8") as f:
        data = json.load(f)

    # top_trace_ids：probe 阶段统计的 trace_id 调用次数列表
    # top_signatures：每个 trace_id 对应的函数签名样本（含 shape、dtype 等）
    trace_ids = data.get("top_trace_ids", [])
    signatures = data.get("top_signatures", [])

    # 构建 trace_id → 调用次数 的字典，方便后面按调用量过滤
    tid_counts: dict[str, int] = {
        item["trace_id"]: item["count"] for item in trace_ids
    }

    # 按 trace_id 分组签名，方便后面提取参数时按 kernel 类型查找
    sigs_by_tid: dict[str, list[dict]] = defaultdict(list)
    for sig in signatures:
        sigs_by_tid[sig["trace_id"]].append(sig)

    # matched_kernels 是 parse 中间态，不是最终 inventory。
    # 它把同一个 op_type/variant 下的多个 trace_id 和 signature 先聚起来，
    # 后面再交给 adapter 做 family-specific 参数提取。
    matched_kernels: dict[str, dict] = {}  # key = "op_type:variant"，聚合相同类型的 kernel
    unmatched_interesting: list[dict] = []  # 无法识别的 trace_id，用于调试
    ignored_noise: list[dict] = []  # 已确认的工具函数噪声，不进入 unmatched
    llm_classified_trace_ids: list[dict] = []  # LLM 对未知 trace 的建议，只作诊断

    # ──────────────────────────────────────────────────────────────────────
    # 2. 遍历每个 trace_id：分类、过滤噪声、必要时生成 LLM 诊断建议
    # ──────────────────────────────────────────────────────────────────────
    # 这一层只回答：
    #   这个 trace_id 大概属于哪个 op_type / variant？
    #
    # 它还不会生成最终 definition_name，因为 definition_name 需要结合
    # signature 里的具体参数，例如 hidden_size/head_dim/vocab_size。
    for tid, count in tid_counts.items():
        # 过滤掉调用次数极少的噪声（通常是初始化或一次性调用）
        if count < 5:
            continue

        matched = False

        # ── 主路径：adapter 优先，manual rules 兜底 ───────────────────────
        # probe JSON 只提供原始 trace 事实；adapter/rules 负责正式分类。
        source = "adapter"
        local_result = classify_adapter_trace_id(tid)
        if local_result is None:
            source = "rules"
            local_result = classify_trace_id(tid)
        if local_result is not None:
            op_type, variant = local_result
            observed_api = observed_fi_api_from_trace_id(tid)
            if (
                observed_api is None
                and op_type == "sampling"
                and source == "adapter"
                and not tid.startswith("flashinfer.")
            ):
                # Non-FlashInfer adapter evidence is a hook target, not fi_api.
                # merge_matched_kernel() records it as target_api.
                observed_api = tid
            key = f"{op_type}:{variant}"  # 聚合 key，如 "rmsnorm:bf16"
            merge_matched_kernel(
                matched_kernels,
                key=key,
                op_type=op_type,
                variant=variant,
                fi_api=observed_api,
                source=source,
                trace_id=tid,
                count=count,
                signatures=sigs_by_tid.get(tid, []),
            )
            matched = True

        if not matched:
            # 已知 helper/noise 直接进入 ignored_noise。
            # 这样 unmatched 里保留的就是更值得人看的未知 trace。
            noise_reason = ignored_noise_reason(tid)
            if noise_reason is not None:
                ignored_noise.append({
                    "trace_id": tid,
                    "count": count,
                    "reason": noise_reason,
                })
                matched = True

        if not matched:
            if llm_classify:
                # ── 备用路径 B：LLM 诊断未知 trace_id ───────────────────
                # 本地规则没分类出来时，调用 LLM 识别这个 trace_id 是什么 kernel。
                # 注意：LLM 结果只作为诊断建议，不会合并进 matched_kernels，
                # 也不会生成正式 kernel entry。人工确认后再手动加到 adapter
                # 或 rules.py 的小型例外表。
                classifier = _get_llm_classifier()  # 获取 LLM 分类器实例（懒加载）
                if classifier is not None:
                    llm_result = classifier.classify_trace_id_via_llm(
                        trace_id=tid,
                        signatures=sigs_by_tid.get(tid, []),  # 把该 trace_id 的签名样本一起发给 LLM，辅助判断
                        model=llm_model,
                        base_url=llm_base_url,
                    )
                    if llm_result is not None:
                        llm_classified_trace_ids.append({
                            "trace_id": tid,
                            "count": count,
                            "suggested_op_type": llm_result.get("op_type"),
                            "suggested_variant": llm_result.get("variant"),
                            "suggested_fi_api": llm_result.get("fi_api"),
                            "suggested_target_api": llm_result.get("target_api"),
                            "source": "llm",
                            "reasoning": llm_result.get("reasoning", ""),
                            "reason": (
                                "LLM classified this trace_id for review only. "
                                "It was not promoted into formal kernels automatically."
                            ),
                        })
                        matched = True

            if not matched:
                # 主路径和 LLM 都没认出来，记录下来供调试用
                unmatched_interesting.append({
                    "trace_id": tid,
                    "count": count,
                })

    # ──────────────────────────────────────────────────────────────────────
    # 3. 打印分类阶段诊断
    # ──────────────────────────────────────────────────────────────────────
    # unmatched 表示“当前规则还不认识”；ignored_noise 表示“认识它是噪声”。
    # 这两个都不是最终 kernels，只是帮助后续扩展规则。
    if unmatched_interesting:
        unmatched_interesting.sort(key=lambda x: x["count"], reverse=True)
        print(
            f"  ⚠️  {len(unmatched_interesting)} unmatched trace_ids "
            f"(top: {unmatched_interesting[0]['trace_id']} x{unmatched_interesting[0]['count']})",
            file=sys.stderr,
        )
    if ignored_noise:
        ignored_noise.sort(key=lambda x: x["count"], reverse=True)
        print(
            f"  ℹ️  Ignored {len(ignored_noise)} known helper/noise trace_ids "
            f"(top: {ignored_noise[0]['trace_id']} x{ignored_noise[0]['count']})",
            file=sys.stderr,
        )

    model_backend = detect_model_backend(data)

    # ──────────────────────────────────────────────────────────────────────
    # 4. 按 op_type 调用 adapter：从 signature 提取参数并生成 kernel entry
    # ──────────────────────────────────────────────────────────────────────
    # 每种 kernel 类型有不同的参数提取逻辑，统一封装在对应 adapter 内。
    kernels: list[dict] = []
    kernels.extend(build_adapter_kernels(matched_kernels))

    # ──────────────────────────────────────────────────────────────────────
    # 5. 标记 inventory 状态：existing / new / needs_config
    # ──────────────────────────────────────────────────────────────────────
    # existing：definitions/ 目录下已存在同名文件，不需要重新生成
    # needs_config：definition_name 里含 NEEDS_CONFIG，缺少 HF config 无法确定参数
    # new：首次发现，需要生成新的 definition JSON
    if definitions_dir and definitions_dir.exists():
        existing_defs = {
            p.stem for p in definitions_dir.rglob("*.json")
        }
        for kernel in kernels:
            if kernel["definition_name"] in existing_defs:
                set_kernel_state(kernel, KERNEL_STATE_EXISTING)
            elif "NEEDS_CONFIG" in kernel["definition_name"]:
                set_kernel_state(kernel, KERNEL_STATE_NEEDS_CONFIG)
            else:
                set_kernel_state(kernel, KERNEL_STATE_NEW)
    else:
        for kernel in kernels:
            if "NEEDS_CONFIG" in kernel["definition_name"]:
                set_kernel_state(kernel, KERNEL_STATE_NEEDS_CONFIG)
            else:
                set_kernel_state(kernel, KERNEL_STATE_NEW)

    # ──────────────────────────────────────────────────────────────────────
    # 6. 可选 config enrichment：用 HF config 补缺失维度
    # ──────────────────────────────────────────────────────────────────────
    # 填充完之后重新检查 existing_defs，避免把已存在的 definition 误标为 new
    if hf_config:
        existing_defs_for_enrich = (
            {p.stem for p in definitions_dir.rglob("*.json")}
            if definitions_dir and definitions_dir.exists()
            else set()
        )
        enrich_with_hf_config(
            kernels,
            hf_config,
            existing_defs_for_enrich,
            tp_size=data.get("tp_size"),
        )

    # ──────────────────────────────────────────────────────────────────────
    # 7. 去重 + deferred 诊断
    # ──────────────────────────────────────────────────────────────────────
    # 去重：不同 trace 层可能最终解析成同一个 definition_name
    # 例如 SGLang wrapper 和 FlashInfer run 都指向同一个 gqa_paged_decode。
    # 下游只需要一个 definition 条目，重复项会污染统计和 collect 列表。
    all_kernel_entries = dedup_kernels_by_definition_name(kernels)
    # deferred：只看到 plan/begin_forward 或 wrapper-only，证据不够进入默认主线。
    kernels, deferred_kernels = split_deferred_kernels(all_kernel_entries, model_backend)
    llm_classified_trace_ids.sort(key=lambda x: x["count"], reverse=True)

    # ──────────────────────────────────────────────────────────────────────
    # 8. 组装最终 inventory dict
    # ──────────────────────────────────────────────────────────────────────
    # 这里还只是返回 Python dict；真正写文件在 main() 里。
    inventory = {
        "model": model_name,
        "probe_file": str(summary_path),
        "total_trace_ids": data.get("unique_trace_ids", 0),
        "total_events": data.get("total_events", 0),
        "model_backend": model_backend,
        "kernels": kernels,
        "deferred_kernels": deferred_kernels,
        "summary": {
            "total": len(kernels),
            "existing": sum(1 for k in kernels if is_existing_kernel(k)),
            "new": sum(1 for k in kernels if is_new_kernel(k)),
            "needs_config": sum(1 for k in kernels if needs_config_kernel(k)),
            "observed_run": len(kernels),
            "deferred": len(deferred_kernels),
            "unmatched": len(unmatched_interesting),
            "ignored_noise": len(ignored_noise),
            "llm_classified_trace_ids": len(llm_classified_trace_ids),
        },
        "unmatched_trace_ids": unmatched_interesting[:20],  # top 20 by count
        "ignored_noise_trace_ids": ignored_noise[:20],  # top 20 by count
        "llm_classified_trace_ids": llm_classified_trace_ids[:20],  # LLM classification audit trail
    }

    return inventory


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def main():
    """CLI entry point.

    命令行入口：
      1. 解析参数
      2. 读取可选 HF config
      3. 调 parse_probe()
      4. 写出 kernel_inventory.json
      5. 打印人类可读 summary
    """
    parser = argparse.ArgumentParser(
        description="Parse probe output into kernel inventory",
    )
    parser.add_argument(
        "summary_path",
        type=Path,
        help="Path to aggregated_summary.json from probe mode",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model name (e.g. Qwen/Qwen3.5-35B-A3B)",
    )
    parser.add_argument(
        "--definitions-dir",
        type=Path,
        default=Path("definitions"),
        help="Path to definitions directory for existing-check",
    )
    parser.add_argument(
        "--hf-config",
        type=Path,
        default=None,
        help="Path to HuggingFace config.json for dimension resolution",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: kernel_inventory.json next to input)",
    )
    parser.add_argument(
        "--llm-classify",
        action="store_true",
        help=(
            "Use LLM to suggest classifications for unknown trace_ids. "
            "Suggestions are recorded for review and are not promoted automatically."
        ),
    )
    parser.add_argument(
        "--llm-model",
        default="claude-sonnet-4-6",
        help="Anthropic model ID for LLM classification (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="Proxy base URL for Anthropic-compatible API (e.g. https://api.aipaibox.com)",
    )
    args = parser.parse_args()

    # 输入 summary 必须存在；parse 不会自动跑 probe。
    if not args.summary_path.exists():
        print(f"ERROR: {args.summary_path} not found", file=sys.stderr)
        sys.exit(1)

    # HF config 是可选的：只有某些 kernel 缺少维度信息时才需要。
    hf_config = None
    if args.hf_config and args.hf_config.exists():
        with open(args.hf_config, encoding="utf-8") as f:
            hf_config = json.load(f)

    # 主调用：所有 parse 逻辑都在 parse_probe() 里完成。
    inventory = parse_probe(
        summary_path=args.summary_path,
        model_name=args.model_name,
        definitions_dir=args.definitions_dir,
        hf_config=hf_config,
        llm_classify=args.llm_classify,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
    )

    # 默认写到 summary 同目录下的 kernel_inventory.json。
    # pipeline 也可以通过 --output 指定到 run 目录。
    output_path = args.output or args.summary_path.parent / "kernel_inventory.json"
    output_path.write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 打印简短摘要，方便终端里快速确认 parse 结果。
    print(f"\n{'='*60}")
    print(f"Model: {inventory['model']}")
    print(f"Total probe events: {inventory['total_events']:,}")
    print(f"Unique trace_ids: {inventory['total_trace_ids']}")
    print(f"{'='*60}")
    print(f"\nKernel Inventory ({inventory['summary']['total']} kernels):")
    print(f"  ✅ Existing: {inventory['summary']['existing']}")
    print(f"  🆕 New:      {inventory['summary']['new']}")
    print(f"  ⚠️  Needs config: {inventory['summary']['needs_config']}")
    print(f"  ▶️  Observed run:  {inventory['summary'].get('observed_run', 0)}")
    print(f"  ⏸️  Deferred:      {inventory['summary'].get('deferred', 0)}")
    print(f"  ❔ Unmatched:     {inventory['summary'].get('unmatched', 0)}")
    print(f"  🧹 Ignored noise: {inventory['summary'].get('ignored_noise', 0)}")
    print(f"  🤖 LLM classified: {inventory['summary'].get('llm_classified_trace_ids', 0)}")
    print()

    for kernel in inventory["kernels"]:
        status_icon = {"existing": "✅", "new": "🆕", "needs_config": "⚠️"}.get(
            get_kernel_state(kernel), "❓"
        )
        fi = kernel.get("fi_api") or "—"
        target = kernel.get("target_api") or "—"
        runtime = kernel.get("runtime_status", "observed_run")
        print(
            f"  {status_icon} {kernel['definition_name']:50s} "
            f"op={kernel['op_type']:12s} runtime={runtime:13s} fi_api={fi} target_api={target}"
        )

    if inventory.get("deferred_kernels"):
        print("\nDeferred kernels (diagnostic only; not generated/collected by default):")
        for kernel in inventory["deferred_kernels"]:
            runtime = kernel.get("runtime_status", "unknown")
            fi = kernel.get("fi_api") or "—"
            target = kernel.get("target_api") or "—"
            print(
                f"  ⏸️  {kernel['definition_name']:50s} "
                f"op={kernel['op_type']:12s} runtime={runtime:13s} fi_api={fi} target_api={target}"
            )

    print(f"\n📄 Output saved to: {output_path}")


if __name__ == "__main__":
    main()
