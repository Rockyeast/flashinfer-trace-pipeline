# Pipeline Internals

**English** | [中文](INTERNALS_ZH.md)

This document covers internal boundaries of the current pipeline, for development and debugging. For day-to-day operation commands see [README.md](README.md).

---

## Main Data Flow

The main pipeline runs in this order:

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

The run root also produces `definition_index.json`, scanned from actual
definition/workload/blob/test files. It is a review-only artifact showing the
current state of run outputs; it does not participate in parse, definition
generation, or collect decisions.

Core principles:

- The probe records facts only; op_type is never decided inside the worker.
- The parser only accepts actually observed `flashinfer.*` API calls as formal `fi_api` evidence.
- `sgl_kernel.*` / `sglang.*` and other non-FlashInfer hook targets are diagnostic or LLM-proposal only; they do not enter the default definition/collect pipeline.
- Definitions prefer official fi_trace output; when no staged definition exists but `flashinfer.*` API evidence does, the adapter generates one based on the official TraceTemplate/reference.

---

## Probe and fi_trace

`probe/scheduler.py` handles Modal scheduling; FlashInfer fi_trace logic is concentrated in `probe/fi_trace_integration.py`:

| Module | Responsibility |
|--------|---------------|
| `probe/scheduler.py` | Modal image, GPU, input args, result collection |
| `probe/fi_trace_integration.py` | fi_trace patch, preflight, output transport |
| `probe/runtime.py` | Python hook / runtime patch injection |
| `probe/inference_runner.py` | Launches SGLang serving inside the subprocess |

Main probe outputs:

- `aggregated_summary.json`: aggregated runtime hook results.
- `fi_trace_out/*.json`: official definition JSON from FlashInfer fi_trace, when the image supports it.
- `fi_trace_staged_definitions.json`: manifest written after the pipeline stages accepted fi_trace JSON into definitions.

Official fi_trace JSON goes through a lightweight check in the `run_pipeline.py` stage step before entering definitions. Current policy:

- Accept future official op_types by default.
- Only block clearly invalid GQA head relationships, e.g. `num_kv_heads > num_qo_heads` or `num_qo_heads % num_kv_heads != 0`.

---

## Parse and Adapters

The formal classification pipeline lives in `scripts/adapters/`. Each adapter covers one or a group of op_types and implements three methods:

```text
classify_trace_id(trace_id)
build_kernels(matched_kernels)
generate_definition(kernel, model_tag, tp)
```

The unified entry point is `scripts/adapters/__init__.py`:

| Function | Purpose |
|----------|---------|
| `classify_trace_id()` | Try each adapter in turn to identify a trace_id |
| `build_kernels()` | Extract const parameters from signature/attrs; produce an inventory kernel entry |
| `generate_definition()` | Call the matching adapter to generate a definition, preferring the official TraceTemplate |

`scripts/parse/rules.py` retains only a small set of manually reviewed exception rules. New kernel families should go into a new or extended adapter, not back into `rules.py`.

`scripts/adapters/extractors.py` holds reusable parameter extraction helpers only. It does not produce inventory entries or write definitions directly.

---

## Definition Source Priority

`generate_definitions.py` decides whether each kernel enters adapter-based generation. The current rules collapse to four outcomes:

```text
skip_existing
skip_not_flashinfer_api
skip_needs_config
generate_with_adapter
```

Kernels whose official fi_trace is already staged, that already have a definition file, that lack `flashinfer.*` API evidence, or that are marked `NEEDS_CONFIG` are all skipped. Only entries with `fi_api: flashinfer.*` are passed to `scripts/adapters/` for definition generation.

| Decision | Meaning |
|----------|---------|
| `skip_existing` | Official fi_trace already staged, or target definition file already exists |
| `skip_not_flashinfer_api` | No `flashinfer.*` API evidence; not eligible for formal definition generation |
| `skip_needs_config` | FlashInfer evidence present but HF config data still needed to fill parameters |
| `generate_with_adapter` | Passed to `scripts/adapters/` for definition generation |

This layer makes source decisions only; family-specific parameter extraction stays in the adapter.

---

## LLM Diagnostic Classification

`parse/llm_classify.py` produces diagnostic suggestions only; results never enter the formal inventory directly.

Trigger: no local adapter or manual rule matched, and `--llm-classify` was explicitly passed.

Outputs:

- `llm_classified_trace_ids` in `kernel_inventory.json` (program-side marker; excluded from definition generation)
- `scripts/parse/llm_classify_cache.json` (API call cache, reused across runs)
- `tmp/run/<run>/llm_diagnostics/parse_rules/` (review-only proposals for human inspection)

LLM suggestions do not automatically modify source rules or generate formal kernel entries. If a suggestion looks correct, use `tools/apply_llm_kernel_proposal.py` after human review to produce an adapter draft, then complete the parameter extraction and definition generation logic and rename it to register. For operational steps see [README.md](README.md).

---

## Collect Boundaries

Collect consumes standard definitions only:

```text
definitions/{op_type}/{definition_name}.json
```

`collect_workloads_modal.py` uses a Python hook to collect real serving workloads in a targeted way. It does not include target_api-only diagnostic targets in the main collect pipeline.

Collect no longer exposes separate ragged/paged modes. It reads existing definitions and groups them automatically:

| Definition type | Collect behavior |
|-----------------|-----------------|
| ragged/default definitions | Uses the default SGLang prefill path |
| paged-prefill definitions | Enables piecewise CUDA graph for the paged-prefill path |

`--paged` / `--both` only control whether the probe stage additionally covers the paged-prefill path; collect decides which passes to run based on the definitions that actually exist.

`collector_diagnostics.json` is the first place to check when collect fails:

- `candidates=0, discarded=0`: no candidate workloads were formed.
- `_capture_summary`: confirms whether the corresponding `fi_api` was hooked.
- `--collect-debug-hooks`: look for `PATCHED` / `CALLED` / `SAVED` in remote stdout.

---

## Validation Boundaries

The pipeline calls the official validator by default at the end:

```bash
flashinfer-bench validate --checks layout,definition,workload
```

This validates only the dataset-layer artifacts this pipeline is responsible for. It is not equivalent to full benchmark/eval trace validation. Whether to submit solutions, eval traces, and reference tests is a PR policy question, not covered automatically by this default validation.
