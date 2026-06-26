# Non-FI Proposal Patterns

Use this file when proposing `definition_source: "agent"` targets under
`proposal/definitions/` and `proposal/definition_hints/`.

These are proposal-time patterns. They keep independent agents consistent before
human review. Runtime repair rules live in `flashinfer_trace/definition_repairs/`;
do not copy Python implementation details here.

## General Rules

- Add a pattern here only for a reusable kernel family, not for one model.
- Follow the current source signature. If the source signature differs from the
  pattern below, cite the source and explain the difference in `review_note`.
- `capture.full_args` should include tensor inputs and scalar parameters needed
  by the definition reference. Do not omit scalar args such as `eps` when they
  appear in the runtime signature.
- Definition references should match source semantics. Do not add dtype casts,
  defaults, or numerical behavior unless the source proves them.

## RMSNorm

Use for SGLang RMSNorm-style module-level kernels such as:

```text
sglang.srt.layers.layernorm.rmsnorm
```

If the source signature is equivalent to:

```python
rmsnorm(x, weight, eps)
```

write:

```json
"capture": {
  "full_args": [0, 1, 2],
  "full_kwargs": [],
  "structural_attr_tokens": []
}
```

The definition should treat `eps` as a Python scalar:

```json
"eps": {"shape": null, "...": "..."}
```

Do not write `shape: []` for `eps`.

## Fused Add RMSNorm

Use for SGLang fused residual-add RMSNorm kernels such as:

```text
sglang.srt.layers.layernorm.fused_add_rmsnorm
```

If the source signature is equivalent to:

```python
fused_add_rmsnorm(x, residual, weight, eps)
```

write:

```json
"capture": {
  "full_args": [0, 1, 2, 3],
  "full_kwargs": [],
  "structural_attr_tokens": []
}
```

The definition should treat `eps` as a Python scalar with `shape: null`.

## SiluAndMul

Use for SGLang SiLU-and-multiply activation kernels such as:

```text
sglang.srt.layers.activation.silu_and_mul
```

If the source semantics split the last dimension in half, write the reference
with the same behavior:

```python
def run(input):
    d = input.shape[-1] // 2
    return F.silu(input[..., :d]) * input[..., d:]
```

Do not add an fp32 cast unless the source implementation does that cast.

For the standard module-level function form, write:

```json
"capture": {
  "full_args": [0],
  "full_kwargs": [],
  "structural_attr_tokens": []
}
```

The matching hints should map:

```json
"inputs": {
  "input": [{"source": "arg", "arg_index": 0}]
}
```
