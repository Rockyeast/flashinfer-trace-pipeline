# Candidate Targets Reference

Use this file when writing or repairing `proposal/candidate_targets.json`.

## Entry Shape

Each target candidate should have at least:

```json
{
  "name": "...",
  "role": "target",
  "definition_name": "... or null",
  "definition_source": "fitrace|agent|manual|unknown",
  "status": "candidate",
  "target": "flashinfer... / sglang... / torch... or null",
  "backend": "flashinfer|sglang_kernel|torch|unknown",
  "op_type": "...",
  "variant": "decode|prefill|... or null",
  "probe_mode": "default|paged|both",
  "page_size": null,
  "dispatch": null,
  "collect": false,
  "companion_attrs": [],
  "capture": {
    "full_args": [],
    "full_kwargs": [],
    "structural_attr_tokens": ["indptr", "indices", "last_page", "page_len", "seq_len", "offset", "mask", "block_table"]
  },
  "evidence": [
    {"kind": "source_location", "value": "path:line or function"}
  ],
  "review_note": "..."
}
```

## General Rules

- `status` stays `candidate` unless the user or reviewer explicitly confirms it.
- `role` is usually `target`. Use `role: "warmup"` only for a reviewed
  warmup-window callable proposal, not for kernels or workloads.
- `backend` describes the runtime/backend family: `flashinfer`,
  `sglang_kernel`, `torch`, or `unknown`.
- `target` should be a hookable Python dotted path when available.
- If no reliable target exists, use `target: null` and `collect: false`.
- Put uncertainty in `review_note`; do not hide uncertainty by filling guessed
  values.
- Do not propose plain GEMM, matmul, or linear targets for standard model
  onboarding unless the user explicitly asks for GEMM coverage. Mention them in
  architecture notes only when they explain model structure or call flow.

## FlashInfer / fitrace Rules

- If source shows `@flashinfer_api(trace=...)`, set
  `definition_source: "fitrace"`.
- For fitrace candidates, `definition_name` is optional or only an expected
  preview. Do not treat an agent-guessed name as the source of truth. The final
  definition name and schema should come from FlashInfer trace/template
  generation.
- Derive `op_type` from the FlashInfer trace template bound by
  `@flashinfer_api(trace=...)`, not from the wrapper/function name.
- Stage words such as `prefill` and `decode` belong in `variant` or tags unless
  the trace template itself defines separate op types.
- Follow the current model/source call chain into SGLang backends and
  FlashInfer APIs. Every reachable `@flashinfer_api(trace=...)` callable on the
  selected backend path must either appear in `candidate_targets.json` or be
  listed in `review_checklist.md` with a source-backed reason for skipping it.
- For FlashInfer wrapper classes, choose the function decorated with
  `@flashinfer_api(trace=...)` as `target`. In current FlashInfer attention
  wrappers this is usually `.run`; `.forward` may be a deprecated runtime alias
  that calls `.run` but is not itself the fitrace API.
- If a FlashInfer-looking callable lacks evidence of `@flashinfer_api(trace=...)`,
  do not assume fitrace support. Set `definition_source: "unknown"` or
  `"manual"` and explain the uncertainty.
- `backend: flashinfer` with `collect: true` uses same-run fitrace definitions.
  In that case `definition_name` is only a preview and the final schema comes
  from fitrace dump.

## Companion Capture

`companion_attrs` lists sibling methods on the same hooked object whose
arguments must be merged into the `run` capture. Use it when a required
parameter is passed to a companion method that stashes it on the instance and
calls `run` without it.

The core hooks each named companion, binds its arguments by signature, and
merges them into the `run` payload. It never guesses values.

Set `companion_attrs` only with source evidence:

- When `target` is a FlashInfer attention wrapper `.run` and the definition
  needs scalars like `sm_scale`, `logits_soft_cap`, or `window_left`, add the
  method that actually receives them.
- For plan-time structural parameters such as paging/index layout, add the
  planning method when source shows the value is set there and not re-passed to
  `run`.
- Leave it `[]` or omit it when every required parameter already appears in the
  `run` signature. Do not add companions speculatively.

## Dispatch

- If `page_size` is set, include a reviewed `dispatch` object that extracts the
  runtime value from call arguments.
- Do not rely on `op_type` or launch settings to imply page size; the core only
  executes explicit dispatch rules.
- If one hook callable can produce multiple semantically distinct shape
  families, split it into multiple candidate targets instead of writing one
  polymorphic definition name.
- Each split target must carry a `dispatch` rule and expected `dispatch_value`.
- Use `dispatch.field` as the event/debug field name, such as `page_size` or
  `hidden_size`.
- Use `dispatch_value` for the reviewed expected value when that field is not
  already a built-in target field.

## Capture

Every `role: "target"` candidate should include a proposed `capture` object.
The deterministic core requires reviewed runtime targets to declare capture
strategy explicitly; it does not keep op-specific tensor dump rules in
`capture.py`.

- Float tensors are summarized by default.
- Only float tensors explicitly listed in `capture.full_args` or
  `capture.full_kwargs` are saved with real values.
- `capture.full_args` lists positional argument indexes that must be saved as
  full tensors.
- `capture.full_kwargs` lists keyword argument names that must be saved as full
  tensors/scalars.
- `capture.structural_attr_tokens` lists wrapper attribute name tokens whose
  tensor values are structural metadata worth preserving when a large object is
  summarized.

## Non-FI / non-fitrace Targets

- Use `definition_source: "agent"` only when the proposal includes matching
  review-only files under `proposal/definitions/` and
  `proposal/definition_hints/`.
- Use `definition_source: "manual"` when a human/special script must define the
  schema instead.
- Non-FI definition drafts must include an official-style `reference` with a
  top-level `run(...)`; helper functions are allowed only when `run(...)` calls
  them.
- Non-FI targets may use `collect: true` only after the reviewer promotes the
  draft into reviewed config. `definition_name` must be non-empty,
  `definition_source` must be `agent` or `manual`, and final files must exist
  under `<run_dir>/config/definitions/<op_type>/<definition_name>.json` and,
  when needed, `<run_dir>/config/definition_hints/<op_type>/<definition_name>.json`.
- Hints are required whenever definition inputs cannot be found by exact kwargs
  name, need positional/attribute lookup, shape squeeze handling, derived axes,
  tensor slicing, or real tensor preservation. Do not rely on core guesses.
- For module-level non-FI functions, positional `arg_index` starts at the first
  real function argument. Do not add a `self` offset.

## Warmup Entries

Warmup candidates are allowed in `candidate_targets.json`, but they are not
workload targets. A warmup entry proposes the callable whose execution window
should mark nested kernel events as `is_warmup=true`.

Use this shape:

```json
{
  "name": "cuda_graph_capture_warmup",
  "role": "warmup",
  "status": "candidate",
  "target": null,
  "module": "sglang.srt.model_executor.cuda_graph_runner",
  "attr": "CudaGraphRunner.capture",
  "backend": "sglang_kernel",
  "collect": false,
  "definition_source": "manual",
  "op_type": "warmup",
  "variant": null,
  "probe_mode": "default",
  "evidence": [
    {"kind": "source_location", "value": "sglang/srt/model_executor/cuda_graph_runner.py:CudaGraphRunner.capture"}
  ],
  "review_note": "Marks CUDA graph capture dummy forward traffic as warmup; verify with hook_status installed=true and parse_report warmup_events>0."
}
```

Rules:

- `module` and `attr` are required; the core will not guess them.
- `target` can be `null`; this entry controls a warmup window and is not
  collected as a workload.
- Do not include `capture` on warmup entries.
- Prefer the outer callable that semantically wraps the whole CUDA graph capture
  phase when source evidence shows it encloses dummy forward traffic.
- Put narrower implementation callables in `review_checklist.md` as
  alternatives unless the outer callable is absent or runtime diagnostics show
  the outer candidate does not work.
