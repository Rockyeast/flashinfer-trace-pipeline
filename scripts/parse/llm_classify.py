#!/usr/bin/env python3
"""LLM-based classifier for unknown trace_ids in parse/parse_probe.py.

When a trace_id fails to match local parse rules, this module asks an LLM to:
  1. Classify the trace_id into (op_type, variant, fi_api/target_api)
  2. Generate a regex_pattern that would match it

The result is cached for review only. It does not modify parse rules or promote
anything into the formal kernel inventory by itself.

Usage (standalone):
    python scripts/parse/llm_classify.py \
        --trace-id "flashinfer.rope.apply_rope_pos_ids" \
        --signatures '[{"signature": {"args": [{"shape": [8, 32, 128]}]}}]' \
        --base-url https://api.aipaibox.com \
        --model claude-sonnet-4-6
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from textwrap import dedent


# ──────────────────────────────────────────────────────────────────────────────
# Few-shot examples (shown to LLM so it understands the schema)
# ──────────────────────────────────────────────────────────────────────────────
# 少样本示例（Few-shot）：给 LLM 展示输入/输出格式，让它理解任务结构
# 每个示例包含：trace_id（函数名）、signatures（张量形状/类型）、result（期望输出）
# LLM 看了这些例子后，对新的 trace_id 就知道该输出什么格式

_FEW_SHOT_EXAMPLES = [
    {
        "trace_id": "flashinfer.norm.rmsnorm",
        "signatures": [
            {
                "trace_id": "flashinfer.norm.rmsnorm",
                "signature": {
                    "args": [
                        {"shape": [4, 4096], "dtype": "bfloat16"},
                        {"shape": [4096], "dtype": "bfloat16"},
                    ]
                },
            }
        ],
        "result": {
            "op_type": "rmsnorm",
            "variant": "rmsnorm",
            "fi_api": "flashinfer.norm.rmsnorm",
            "regex_pattern": r"^flashinfer\.norm\.rmsnorm$",
            "reasoning": "This is flashinfer.norm.rmsnorm — RMSNorm on (batch, hidden) with a weight vector of hidden_size.",
        },
    },
    {
        "trace_id": "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run",
        "signatures": [
            {
                "trace_id": "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run",
                "signature": {
                    "args": [
                        {"shape": [8, 32, 128], "dtype": "bfloat16"},
                    ]
                },
            }
        ],
        "result": {
            "op_type": "gqa_paged",
            "variant": "decode",
            "fi_api": "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper",
            "regex_pattern": r"^flashinfer\.decode\.BatchDecodeWithPagedKVCacheWrapper\.run$",
            "reasoning": "Paged KV-cache decode attention wrapper. 3-D query tensor (batch, heads, head_dim).",
        },
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ──────────────────────────────────────────────────────────────────────────────

# 目前已知的 op_type 类别列表，会被注入到 LLM prompt 里作为参考
# LLM 可以从这里选一个，也可以自己发明新的 snake_case 名字
_KNOWN_OP_TYPES = [
    "gqa_paged",
    "gqa_ragged",
    "mla_paged",
    "dsa_paged",
    "moe",
    "gdn",
    "gemm",
    "rmsnorm",
    "sampling",
    "quantization",
    "quantize_fp4",
    "cascade_merge",
    "comm",
]


def _build_prompt(
    trace_id: str,
    signatures: list[dict],
) -> str:
    """Build the LLM classification prompt."""
    # 构建发给 LLM 的 prompt，包含三部分：
    # 1. 已知 op_type 名称列表
    # 2. Few-shot 示例（教 LLM 输出格式）
    # 3. 待分类的 trace_id 及其签名

    # 签名最多取 6 条，避免 prompt 太长
    sigs_for_prompt = signatures[:6]
    sigs_json = json.dumps(sigs_for_prompt, indent=2, ensure_ascii=False)

    # 把 few-shot 示例拼成文本块，用 --- 分隔
    few_shot_blocks = []
    for ex in _FEW_SHOT_EXAMPLES:
        few_shot_blocks.append(
            f"trace_id: {ex['trace_id']}\n"
            f"signatures (sample): {json.dumps(ex['signatures'][:2], separators=(',', ':'))}\n"
            f"→ result:\n```json\n{json.dumps(ex['result'], indent=2)}\n```"
        )
    few_shot_text = "\n\n---\n\n".join(few_shot_blocks)

    known_op_types_str = ", ".join(f'"{t}"' for t in _KNOWN_OP_TYPES)

    # dedent 用于去掉 Python 缩进，让 prompt 格式整洁
    prompt = dedent(f"""
        You are analyzing a FlashInfer/SGLang GPU kernel trace to classify an
        unknown trace_id.

        ## Task

        Given an unknown trace_id and its call signatures (tensor shapes/dtypes),
        output a JSON classification with:
        - `op_type`: the kernel family (see known types below, or invent a new snake_case name)
        - `variant`: the specific variant within op_type (e.g. "decode", "prefill", "rmsnorm")
        - `fi_api`: the FlashInfer Python API string, or null.
          IMPORTANT: fi_api must be null unless the value starts with "flashinfer.".
          Do not infer or invent a FlashInfer fi_api for sglang.* or sgl_kernel.*
          traces, even if the implementation may call FlashInfer internally.
        - `target_api`: a non-FlashInfer Python hook target such as "sgl_kernel.foo"
          or "sglang.srt.layers.foo", or null if none.
        - `regex_pattern`: a Python regex that would match this trace_id (and similar variants)
        - `reasoning`: a one-sentence explanation

        Known op_types: {known_op_types_str}
        (You may define a new op_type if none fits.)
        Prefer concrete benchmark kernel families over umbrella categories.
        For example, use `gelu_tanh_and_mul` or `silu_and_mul`, not `activation`.

        ## Few-shot examples

        {few_shot_text}

        ---

        ## Unknown trace_id to classify

        trace_id: `{trace_id}`

        signatures (sample, up to 6):
        ```json
        {sigs_json}
        ```

        ## Output format

        Respond with ONLY a JSON block:

        ```json
        {{
          "op_type": "...",
          "variant": "...",
          "fi_api": "flashinfer...." or null,
          "target_api": "sgl_kernel...." / "sglang...." or null,
          "regex_pattern": "...",
          "reasoning": "..."
        }}
        ```

        Do not include any text outside the JSON block.
    """).strip()

    return prompt


# ──────────────────────────────────────────────────────────────────────────────
# Response parser
# ──────────────────────────────────────────────────────────────────────────────

# 正则：匹配 LLM 回复里的 ```json ... ``` 代码块（忽略大小写）
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _sanitize_json_strings(json_str: str) -> str:
    """Escape literal newlines/tabs inside JSON string values."""
    # LLM 有时会在 JSON 字符串值里直接插入换行/制表符（不转义），导致 json.loads 报错
    # 这里逐字符扫描，只在"字符串内部"（in_string=True）时把裸换行转成 \\n 等转义序列
    result = []
    in_string = False
    i = 0
    while i < len(json_str):
        ch = json_str[i]
        if ch == '"' and (i == 0 or json_str[i - 1] != "\\"):
            # 遇到非转义的引号，切换"是否在字符串内部"状态
            in_string = not in_string
            result.append(ch)
        elif in_string:
            # 在字符串内部，裸控制字符需要转义
            if ch == "\n":
                result.append("\\n")
            elif ch == "\r":
                result.append("\\r")
            elif ch == "\t":
                result.append("\\t")
            else:
                result.append(ch)
        else:
            result.append(ch)
        i += 1
    return "".join(result)


def _parse_llm_response(response_text: str) -> dict | None:
    """Extract and parse the JSON block from LLM response."""
    # 先尝试从 ```json ... ``` 代码块里提取 JSON
    match = _JSON_BLOCK_RE.search(response_text)
    if not match:
        # 没有代码块就直接当作裸 JSON 解析（兜底）
        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            return None

    json_str = match.group(1).strip()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # 第一次解析失败，尝试修复 JSON 字符串里的裸换行后再解析
        try:
            return json.loads(_sanitize_json_strings(json_str))
        except json.JSONDecodeError:
            return None


def _normalize_api_fields(result: dict) -> dict:
    """Keep fi_api reserved for FlashInfer and move other APIs to target_api."""
    result = dict(result)
    fi_api = result.get("fi_api")
    target_api = result.get("target_api")

    if isinstance(fi_api, str) and fi_api:
        if fi_api.startswith("flashinfer."):
            result["target_api"] = target_api or None
        else:
            result["fi_api"] = None
            result["target_api"] = target_api or fi_api
    elif isinstance(target_api, str) and target_api.startswith("flashinfer."):
        result["fi_api"] = target_api
        result["target_api"] = None
    else:
        result["fi_api"] = None
        result["target_api"] = target_api or None

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Local JSON cache — avoids re-calling LLM for the same trace_id
# ──────────────────────────────────────────────────────────────────────────────
# 本地缓存：以 trace_id 为 key，存储 LLM 返回的分类结果
# 好处：同一个 trace_id 只调用一次 LLM，后续直接读缓存，省钱省时间
# 缓存文件默认放在 scripts/parse/llm_classify_cache.json，所有模型/运行共享

# Default cache location: scripts/parse/llm_classify_cache.json
# (shared across all models/runs on the same machine)
_DEFAULT_CACHE_PATH = Path(__file__).parent / "llm_classify_cache.json"


def _load_cache(cache_path: Path) -> dict:
    """Load the LLM classification cache from disk. Returns {} on any error."""
    # 读取缓存文件，返回 dict（trace_id → result）
    # 任何错误（文件不存在、JSON 损坏等）都静默返回空 dict，不影响主流程
    try:
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        pass
    return {}


def _save_cache(cache: dict, cache_path: Path) -> None:
    """Persist the cache to disk (atomic-ish write)."""
    # 把更新后的缓存写回文件（整体覆盖写，非增量追加）
    # sort_keys=True 让文件内容稳定，方便 git diff
    try:
        cache_path.write_text(
            json.dumps(cache, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [1b] Warning: cannot write cache to {cache_path}: {exc}", file=sys.stderr)


# Main public entry point
# ──────────────────────────────────────────────────────────────────────────────


def classify_trace_id_via_llm(
    trace_id: str,
    signatures: list[dict],
    *,
    model: str = "claude-sonnet-4-6",
    base_url: str | None = None,
    api_key: str | None = None,
    cache_file: Path | None = _DEFAULT_CACHE_PATH,
) -> dict | None:
    """Classify an unknown trace_id using an LLM.

    Args:
        trace_id: The unmatched trace_id string.
        signatures: List of signature dicts from the probe output.
        model: Anthropic model ID.
        base_url: Optional proxy base URL (e.g. https://api.aipaibox.com).
        api_key: Anthropic API key (falls back to ANTHROPIC_API_KEY env var).
        cache_file: Path to local JSON cache file. Pass None to disable caching.

    Returns:
        dict with keys: op_type, variant, fi_api, target_api, regex_pattern, reasoning
        or None on failure.
    """
    # ── 缓存查询 ──────────────────────────────────────────────────────────────
    # 先查本地缓存，命中则直接返回，不调用 LLM（节省 API 费用和延迟）
    if cache_file is not None:
        cache = _load_cache(cache_file)
        if trace_id in cache:
            print(
                f"  [1b] ✅ Cache hit for {trace_id!r} — skipping LLM call",
                file=sys.stderr,
            )
            return _normalize_api_fields(cache[trace_id])
    else:
        cache = {}  # cache_file=None 时禁用缓存，但仍需要空 dict 供后续赋值

    # ── 导入 anthropic SDK（懒导入，避免没装包时启动就报错）────────────────
    try:
        import anthropic
    except ImportError:
        print(
            "  [1b] anthropic package not installed. "
            "Run: pip install anthropic",
            file=sys.stderr,
        )
        return None

    # ── API Key 和 Base URL 处理 ───────────────────────────────────────────
    # 优先用参数传入的 key，其次读环境变量 ANTHROPIC_API_KEY
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        print("  [1b] No API key found. Set ANTHROPIC_API_KEY.", file=sys.stderr)
        return None

    # base_url 支持代理（如 aipaibox），不设置则用 Anthropic 官方地址
    effective_base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL")

    # ── 构建 prompt 并调用 LLM ───────────────────────────────────────────────
    prompt = _build_prompt(trace_id, signatures)

    print(
        f"  [1b] Calling LLM to classify trace_id: {trace_id!r}",
        file=sys.stderr,
    )

    try:
        # 根据是否有代理 URL 来初始化 Anthropic 客户端
        if effective_base_url:
            client = anthropic.Anthropic(api_key=key, base_url=effective_base_url)
        else:
            client = anthropic.Anthropic(api_key=key)

        # 发送单轮对话请求，max_tokens=1024 足够容纳一个 JSON 结果
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = message.content[0].text  # 取第一段文本内容
    except Exception as exc:  # noqa: BLE001
        print(f"  [1b] LLM API error: {exc}", file=sys.stderr)
        return None

    # ── 解析 LLM 回复 ────────────────────────────────────────────────────────
    # 从回复里提取 JSON，解析失败则打印原始回复帮助 debug
    result = _parse_llm_response(response_text)
    if result is None:
        print(
            f"  [1b] Failed to parse LLM response for {trace_id!r}",
            file=sys.stderr,
        )
        print(f"  [1b] Raw response:\n{response_text[:800]}", file=sys.stderr)
        return None
    result = _normalize_api_fields(result)

    # ── 字段校验 ─────────────────────────────────────────────────────────────
    # 确保 LLM 返回了必要字段（fi_api/target_api 和 reasoning 是可选的）
    required = {"op_type", "variant", "regex_pattern"}
    missing = required - set(result.keys())
    if missing:
        print(
            f"  [1b] LLM result missing fields {missing} for {trace_id!r}",
            file=sys.stderr,
        )
        return None

    op_type = result["op_type"]
    variant = result["variant"]
    fi_api = result.get("fi_api")          # 仅允许 flashinfer.*；否则已归一化为 None
    target_api = result.get("target_api")  # 非 FlashInfer hook target，如 sgl_kernel.*
    regex_pattern = result["regex_pattern"]
    reasoning = result.get("reasoning", "")  # 分类理由，仅用于打印

    print(
        f"  [1b] 🤖 Classified {trace_id!r} → op_type={op_type!r}, "
        f"variant={variant!r}, fi_api={fi_api!r}, target_api={target_api!r}",
        file=sys.stderr,
    )
    if reasoning:
        print(f"  [1b]    Reasoning: {reasoning}", file=sys.stderr)

    result_dict = {
        "op_type": op_type,
        "variant": variant,
        "fi_api": fi_api,
        "target_api": target_api,
        "regex_pattern": regex_pattern,
        "reasoning": reasoning,
    }

    # ── 写入缓存 ──────────────────────────────────────────────────────────────
    # 把本次 LLM 结果存入缓存，下次同一 trace_id 直接命中
    if cache_file is not None:
        cache[trace_id] = result_dict
        _save_cache(cache, cache_file)

    return result_dict


# ──────────────────────────────────────────────────────────────────────────────
# CLI (standalone usage)
# ──────────────────────────────────────────────────────────────────────────────
# 命令行入口：可以直接 python parse/llm_classify.py --trace-id xxx --signatures '[...]'
# 主要用途：手动测试单个 trace_id 的分类，并把结果缓存到本地 JSON


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM classifier for unknown trace_ids (parse/llm_classify)"
    )
    parser.add_argument(
        "--trace-id",
        required=True,
        help="The trace_id to classify (e.g. 'flashinfer.rope.apply_rope_pos_ids')",
    )
    parser.add_argument(
        "--signatures",
        default="[]",
        help="JSON array of signature dicts from probe output",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Anthropic model ID",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Proxy base URL for Anthropic-compatible API (e.g. https://api.aipaibox.com)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Anthropic API key (default: ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=_DEFAULT_CACHE_PATH,
        help=(
            f"Path to local JSON cache file (default: {_DEFAULT_CACHE_PATH}). "
            "Pass empty string '' to disable caching."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        # 不指定 --out 则把 JSON 结果打印到 stdout，方便管道使用
        help="Write JSON result to this file (default: print to stdout)",
    )
    args = parser.parse_args()

    # 解析 --signatures 参数（JSON 字符串 → Python list）
    try:
        signatures = json.loads(args.signatures)
    except json.JSONDecodeError as exc:
        print(f"ERROR: --signatures is not valid JSON: {exc}", file=sys.stderr)
        sys.exit(1)

    # 空字符串 "" 表示禁用缓存
    cache_file: Path | None = args.cache_file if str(args.cache_file) != "" else None

    result = classify_trace_id_via_llm(
        trace_id=args.trace_id,
        signatures=signatures,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        cache_file=cache_file,
    )

    if result is None:
        print("ERROR: Classification failed", file=sys.stderr)
        sys.exit(1)

    output_json = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        # 写到指定文件
        args.out.write_text(output_json, encoding="utf-8")
        print(f"Saved to {args.out}")
    else:
        # 打印到 stdout
        print(output_json)


if __name__ == "__main__":
    main()
