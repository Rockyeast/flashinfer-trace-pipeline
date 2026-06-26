# Definition Standards Reference

Use this file as the compact reference for FlashInfer trace definition proposals. It summarizes the useful standards from the official `extract-kernel-definitions` skill, but this redesign still stays review-only.

## Core meaning

A definition describes the semantic contract of one benchmarkable kernel/workload family: name, op_type, variant, axes, inputs, outputs, tags, and the expected reference/solution boundary.

A definition is not the same thing as one runtime call. Runtime calls are evidence. A reviewed definition is the dataset schema that later workloads must satisfy.

For scalar inputs, write `"shape": null`. Do not write `"shape": []` for scalar values; official validation treats that as a zero-dimensional tensor contract, not a Python scalar.

## Definition source priority

Use the current FlashInfer trace source as the highest authority when it exists.

- If the candidate callable is decorated with `@flashinfer_api(trace=...)`, set `definition_source: "fitrace"`. The agent should discover the hook target and cite the trace/template evidence, but should not invent the final definition schema. `definition_name` may be omitted or written only as an expected preview.
- For fitrace-backed candidates, derive the official definition from the FlashInfer `TraceTemplate` / dispatch function and the real captured arguments. Existing dataset filenames are compatibility examples, not the source of truth.
- If the callable is under `flashinfer.*` but no `@flashinfer_api(trace=...)` evidence is found, do not assume fitrace support. Mark it `definition_source: "unknown"` or `"manual"` until reviewed.
- For SGLang/Triton/torch kernels without fitrace, the agent may propose `definition_source: "agent"`, but that remains a review-only draft and needs human approval before becoming an official definition.

## Naming

Prefer `{op_type}_{variant}_{key_params}`.

Common abbreviations:

- `h`: num heads or hidden size, depending on op_type
- `kv`: num_kv_heads
- `d`: head_dim
- `ps`: page_size
- `ckv`: compressed_kv_dim
- `kpe`: key positional encoding dim
- `e`: num experts
- `i`: intermediate size
- `topk`: selected experts / sparse topk
- `ng`: group count
- `kg`: topk group
- `v`: vocab size

Examples:

- `gqa_paged_decode_h32_kv8_d128_ps1`
- `gqa_paged_prefill_h32_kv8_d128_ps64`
- `gqa_ragged_h32_kv8_d128`
- `mla_paged_decode_h16_ckv512_kpe64_ps1`
- `dsa_sparse_decode_h16_ckv512_kpe64_topk256_ps1`
- `gdn_decode_qk4_v8_d128`
- `rmsnorm_h4096`
- `fused_add_rmsnorm_h7168`
- `top_k_sampling_from_probs_v129280`

## Tags

Use tags to make review and collection reproducible:

- `status:verified` or `status:unverified`
- `stage:decode` or `stage:prefill`
- `model:<slug>` when the definition is known to be used by a model
- `tp:<N>` only when tensor parallelism changes the definition constants
- `ep:<N>` only when expert parallelism changes the definition constants
- `quantization:<format>` when relevant
- `routing:<kind>` for MoE routing when relevant
- `fi_api:<dotted FlashInfer API>` only when a real FlashInfer API exists

Do not add `tp`/`ep` tags to parallelism-agnostic kernels such as rmsnorm, gemm, rope, and sampling.

## FlashInfer API mapping hints

These are hints, not automatic approval:

- `gqa_paged` decode: `flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper`
- `gqa_paged` prefill: `flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper`
- `gqa_ragged`: `flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper`
- `mla_paged`: `flashinfer.mla.BatchMLAPagedAttentionWrapper`
- `rmsnorm`: `flashinfer.norm.rmsnorm` or `flashinfer.norm.fused_add_rmsnorm`
- `gdn`: `flashinfer.gdn.*`
- `sampling`: `flashinfer.sampling.*`
- `moe`: FlashInfer support varies; verify the exact callable

If no reliable FlashInfer API exists, keep `target` null or use `collect: false` until a hookable target is reviewed.

## Parallelism rules

For attention/GDN, TP changes local head counts. Definitions should use post-TP local constants if the workload is captured under TP.

For MoE, EP changes `num_local_experts`. Definitions should distinguish global expert count from local expert count.

For rmsnorm/gemm/rope/sampling, TP/EP generally do not change the definition contract. Do not encode TP/EP just because the model was served with TP/EP.

## Review requirement

Do not generate official definitions directly from this reference. Use it to judge whether `candidate_targets.json` and later definition drafts are coherent. Official definition files are written only after human review.
