# 内部机制说明

[English](INTERNALS.md) | **中文**

本文档记录当前 pipeline 的内部边界，供开发和调试使用。日常操作命令见
[README_ZH.md](README_ZH.md)。

---

## 主线数据流

当前主线按下面的顺序走：

```text
probe/scheduler.py
  -> probe/inference_runner.py
  -> aggregated_summary.json
  -> parse/parse_probe.py
  -> kernel_inventory.json
  -> generate_definitions.py
  -> collect_workloads_modal.py
  -> generate_baseline_solutions.py
  -> generate_tests.py
  -> audit_schemas.py
  -> flashinfer-bench validate
```

run root 还会生成 `definition_index.json`。它从实际 definition/workload/blob/test
文件扫描得到，用于 review 当前 run 的产物状态，不参与 parse、definition 生成或 collect 决策。

核心原则：

- probe 只记录事实，不在 worker 里决定 op_type。
- parse 只把真实观测到的 `flashinfer.*` API 当作正式 `fi_api` 证据。
- `sgl_kernel.*` / `sglang.*` 这类非 FlashInfer hook 目标只能作为诊断或 LLM proposal，不进入默认 definition/collect 主线。
- definition 优先使用官方 fi_trace 输出；没有 staged definition 但有
  `flashinfer.*` API 证据时，再由 adapter 基于官方 TraceTemplate/reference 生成。

---

## Probe 与 fi_trace

`probe/scheduler.py` 负责 Modal 调度；FlashInfer fi_trace 相关逻辑集中在
`probe/fi_trace_integration.py`：

| 模块 | 职责 |
|------|------|
| `probe/scheduler.py` | Modal image、GPU、输入参数、结果收集 |
| `probe/fi_trace_integration.py` | fi_trace patch、preflight、输出传输 |
| `probe/runtime.py` | Python hook/runtime patch 注入 |
| `probe/inference_runner.py` | 子进程里实际启动 SGLang serving |

probe 输出主要包括：

- `aggregated_summary.json`：runtime hook 聚合结果。
- `fi_trace_out/*.json`：如果当前 FlashInfer image 支持官方 fi_trace，则这里会有官方 definition JSON。
- `fi_trace_staged_definitions.json`：pipeline 把可接受的官方 fi_trace JSON stage 到 definitions 后生成的 manifest。

官方 fi_trace JSON 进入 definitions 前会在 `run_pipeline.py` 的 stage 步骤做轻量检查。目前策略是：

- 默认接受未来官方 op_type。
- 只 block 明确不合法的 GQA head 关系，例如 `num_kv_heads > num_qo_heads` 或 `num_qo_heads % num_kv_heads != 0`。

---

## Parse 与 Adapter

当前正式分类主线在 `scripts/adapters/`，每个 adapter 负责一个或一组 op_type 的三件事：

```text
classify_trace_id(trace_id)
build_kernels(matched_kernels)
generate_definition(kernel, model_tag, tp)
```

统一入口在 `scripts/adapters/__init__.py`：

| 函数 | 用途 |
|------|------|
| `classify_trace_id()` | 逐个 adapter 尝试识别 trace_id |
| `build_kernels()` | 从 signature/attrs 中提取 const 参数，生成 inventory kernel entry |
| `generate_definition()` | 调对应 adapter 生成 definition，优先贴官方 TraceTemplate |

`scripts/parse/rules.py` 只保留少量人工审核过的例外规则。正常新增 kernel family
时，不应该把主要逻辑塞回 `rules.py`，而是新增或扩展 adapter。

`scripts/adapters/extractors.py` 只放可复用的参数提取 helper。它不直接产出 inventory，
也不直接写 definition。

---

## Definition 来源优先级

`generate_definitions.py` 负责决定某个 kernel 要不要进入 adapter 生成。
当前规则收敛成四类：

```text
skip_existing
skip_not_flashinfer_api
skip_needs_config
generate_with_adapter
```

official fi_trace 已经 staged、已有 definition、非 `flashinfer.*` API、
或 `NEEDS_CONFIG` 都不会进入 adapter。其他带 `fi_api: flashinfer.*` 的条目才交给
`scripts/adapters/` 尝试生成 definition。

具体含义：

| decision | 含义 |
|----------|------|
| `skip_existing` | 官方 fi_trace 已经 staged，或目标 definition 已存在 |
| `skip_not_flashinfer_api` | 没有 `flashinfer.*` API 证据，不进入正式 definition 生成 |
| `skip_needs_config` | 有 FlashInfer 线索，但仍缺 HF config 补全信息 |
| `generate_with_adapter` | 交给 `scripts/adapters/` 生成 definition |

这层只做来源决策，不做 family-specific 参数提取；参数提取仍在 adapter 里。

---

## LLM 诊断分类

`parse/llm_classify.py` 只做诊断建议，不直接进入正式 inventory。

触发条件：本地 adapter/manual rule 都没命中，且显式开启 `--llm-classify`。

输出位置：

- `kernel_inventory.json` 里的 `llm_classified_trace_ids`（程序标记，不进 definition 生成）
- `scripts/parse/llm_classify_cache.json`（跨 run 复用的 API 调用缓存）
- `tmp/run/<run>/llm_diagnostics/parse_rules/`（review-only proposals，供人工审阅）

LLM 建议不自动修改源码规则，也不自动生成正式 kernel entry。如果建议靠谱，人工
review 后用 `tools/apply_llm_kernel_proposal.py` 生成 adapter draft，再补全参数提取
和 definition 生成逻辑，改名注册。操作步骤见 [README_ZH.md](README_ZH.md)。

---

## Collect 边界

默认 collect 只吃标准 definition：

```text
definitions/{op_type}/{definition_name}.json
```

`collect_workloads_modal.py` 使用 Python hook 定向收集真实 serving workload。
它不会把 target_api-only 诊断目标放入主线 collect。

Collect 本身不再暴露独立的 ragged/paged 模式。它读取已有 definitions 后自动分组：

| definition 类型 | collect 行为 |
|-----------------|--------------|
| ragged/default definitions | 走默认 SGLang prefill 路径 |
| paged-prefill definitions | 启用 piecewise CUDA graph 走 paged-prefill 路径 |

`--paged` / `--both` 只控制 probe 阶段是否额外覆盖 paged-prefill 路径；
collect 根据实际存在的 definition 自动决定需要跑哪些 pass。

`collector_diagnostics.json` 是排查 collect 失败的第一入口：

- `candidates=0, discarded=0`：没有形成候选 workload。
- `_capture_summary`：确认对应 `fi_api` 是否被 hook 到。
- `--collect-debug-hooks`：远端 stdout 里看 `PATCHED` / `CALLED` / `SAVED`。

---

## Validation 边界

pipeline 末尾默认调用官方：

```bash
flashinfer-bench validate --checks layout,definition,workload
```

这只验证当前 pipeline 负责的数据集层产物，不等价于完整 benchmark/eval trace
验证。solution、eval trace、reference test 是否提交，是 PR 策略问题，不由这个默认
validation 自动覆盖。
