# Workload Collection Notes

Use this file when proposing `probe_mode`, `collect`, capture scope, and reviewed run coverage inputs.

## Probe vs collect

Probe answers: can the reviewed target be triggered at runtime, and what evidence does it produce?

Collect answers: can we build representative workload rows for the reviewed definition, with enough coverage for benchmark use?

A probe can be intentionally small, for example one token or one simple prompt. Collection should use coverage inputs selected for the definition family.

## Collect flag

- `collect: true`: automatic workload collection is expected; `target` must be
  hookable. For FlashInfer/fitrace targets, same-run fitrace provides the
  definition. For non-FI targets, reviewed definition/hints must be promoted
  into `config/` before runtime collect.
- `collect: false`: do not collect. Use this for review-only candidates,
  rejected candidates, or targets that need a future special script/human
  intervention.

## Target policy

The main path only hooks reviewed targets. The agent may propose targets, but it must not auto-approve them.

Targets should be hookable dotted Python paths. Do not stop at high-level model methods if a more precise backend/wrapper target exists.

## Warmup

Use explicit `is_warmup` marking when available. Do not use call count alone as a warmup filter:

- warmup may call real kernels
- real requests may call low-frequency kernels
- low count does not mean warmup

## Tensor storage policy

Do not dump every tensor blindly.

- Large fp16/bf16/fp32 activation/cache tensors are usually represented as random tensors with real shape/dtype.
- Structural tensors must preserve real values: int tensors, indices, indptr, positions, page tables, routing indices, and similar metadata.
- Scalars such as scale values can be stored as scalar values.
- Special kernels such as sampling may require custom handling because values may affect semantics.

## Attention path hints

For ragged prefill, SGLang route can depend on runtime flags. The official old workflow notes that ragged may require disabling piecewise CUDA graph and avoiding deterministic inference flags that force paged behavior.

For paged prefill, deterministic/paged routing flags may be needed depending on SGLang version and backend.

These flags are not proof of correctness. They are runtime strategy hints and should be verified by events and parse reports.

## Companion capture

FlashInfer attention wrappers split a call across methods. `.forward(...)` receives scalars such as `sm_scale`, stashes them on the instance (`self._sm_scale = sm_scale`), then calls `self.run(...)` without re-passing them. Plan/index layout is similarly set in `.plan`/`.begin_forward`. So when the hooked `target` is `.run`, those parameters never appear in the `run` args/kwargs and the captured workload is incomplete.

Do not solve this by guessing defaults (e.g. `1/sqrt(head_dim)`) in core or sanitizer. Instead declare the sibling methods in the candidate `companion_attrs`:

- scalars (`sm_scale`, `logits_soft_cap`, `window_left`): usually `["forward", "forward_return_lse"]`
- plan-time structural params: usually `["plan", "begin_forward"]`

The core hooks each named companion, binds its arguments by signature, stashes them on the instance, and merges them into the next `run` capture. It only records values that were actually passed; it never fabricates a default. Set `companion_attrs` only with source evidence, and record uncertainty in `review_note` rather than guessing which method carries a value.

## Reports

Use runtime outputs as feedback into the proposal loop:

- `reports/run_report.json#parse_report`: matched/missing/ignored target summary
- `reports/run_report.json`: machine-readable run ledger, including collect,
  definition audit, internal validation, export, and upstream validation when available
- `reports/review.md`: human-readable review summary

Reports are evidence for review. They are not automatic approval.

## Collection run baseline

The reviewed collect run should stay close to the official collection shape:

- ShareGPT prompt source, using the checked-in 100 prompt fixture when no external dataset is supplied.
- Tiered prompt scenarios by batch size: `1, 2, 4, 8, 16, 32, 64`.
- Scenario generation lengths: `96, 64, 32, 32, 16, 8, 8` tokens respectively.
- Sampling supplemental configs:
  - `temperature=0.7, top_k=50, top_p=0.9`
  - `temperature=0.7, top_k=50, top_p=1.0`
  - `temperature=0.7, top_k=-1, top_p=0.9`

This is still not a substitute for human review. It only defines the default coverage shape used after targets are reviewed.

## Backend field

Use `backend` to separate semantic existence from current collection support:

- `flashinfer`: FlashInfer API or wrapper; may use `collect: true` when target and definition are reviewed.
- `sglang_kernel`: SGLang native kernel path. Use `collect: true` only when the
  proposal includes reviewed non-FI definition/hints drafts and a human later
  promotes them into `config/`; otherwise use `collect: false`.
- `torch`: PyTorch/nn/matmul path. Use `collect: true` only with reviewed
  definition/hints and a hookable target; otherwise use `collect: false`.
- `unknown`: evidence is weak or source path is unresolved; use `collect: false`.

Do not drop a real model component just because it is not FlashInfer-backed.
Keep it as a review-only candidate when definition/hints are not ready; propose
non-FI collect only when the schema and hints are reviewable.
