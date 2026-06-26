# Runtime Config Reference

Use this file when writing or repairing `<run_dir>/config/run_config.json`.

`run_config.json` is a reviewed runtime proposal in the exact place the
deterministic core reads. Do not hide runtime configuration only in markdown.

## Shape

Write a JSON object like:

```json
{
  "model_name": "<hf_repo_or_local_model_path>",
  "image": "lmsysorg/sglang:v0.5.12.post1",
  "gpu": "L40S",
  "tp_size": 1,
  "disable_cuda_graph": true,
  "engine_kwargs": {},
  "timeout": 3600,
  "batch_sizes": [1, 2, 4, 8, 16, 32, 64],
  "max_new_tokens": 96,
  "max_captures_per_target": 128,
  "supplemental_runs": [
    {
      "name": "sampling_supplemental",
      "sampling_params": {"temperature": 0.7, "top_k": 50, "top_p": 0.9},
      "allowed_op_types": ["sampling"]
    },
    {
      "name": "sampling_top_p_1",
      "sampling_params": {"temperature": 0.7, "top_k": 50, "top_p": 1.0},
      "allowed_op_types": ["sampling"]
    },
    {
      "name": "sampling_top_k_disabled",
      "sampling_params": {"temperature": 0.7, "top_k": -1, "top_p": 0.9},
      "allowed_op_types": ["sampling"]
    }
  ]
}
```

Also summarize these choices in `review_checklist.md` with source evidence and
uncertainties.

## Rules

- `model_name`, `image`, `gpu`, and `tp_size` are required for `run`.
- Do not include `definitions_dir`. Standard `run` and `validate` always use
  the selected run root's `output/definitions`, populated from the same-run
  FlashInfer fitrace dump.
- Use HF config and sgl-cookbook only as proposal evidence. The core will not
  fetch or guess these values later.
- If TP/GPU/image are uncertain, keep the target proposal review-only and write
  the uncertainty in `review_checklist.md`; do not hide it behind a heuristic.
- If a runtime image comes from a nightly/dev cookbook or source comment, verify
  the Docker tag exists before writing it into `config/run_config.json`. If the
  tag cannot be verified, keep it in review notes.
- `disable_cuda_graph` is a reviewed runtime choice. Prefer `true` for stable
  collect unless the reviewer explicitly wants CUDA graph warmup/capture
  coverage.
- `engine_kwargs` is an optional reviewed map of extra SGLang `Engine` kwargs.
  Use it only for source-backed runtime knobs required to select a backend path.
  The runtime rejects kwargs unsupported by the selected SGLang image.
- If the prompt includes a runtime config compatibility requirement, copy those
  exact `engine_kwargs` into `config/run_config.json`. These values are reviewed
  startup fixes, not workload-discovery guesses.
- If the HF config or architecture is multimodal-capable but this trace run is
  text-only, propose `"enable_multimodal": false` inside `engine_kwargs`.
- `batch_sizes`, `max_new_tokens`, `supplemental_runs`, and
  `max_captures_per_target` are reviewed collect strategy fields. Propose them
  explicitly when official workload coverage matters. `run` fails if these
  required strategy fields are missing.
- The core uses the checked-in ShareGPT prompt fixture and does not accept
  prompt overrides in `run_config.json`.
- Use `supplemental_runs[*].allowed_op_types` to limit extra generation passes
  to the intended workload family. The runtime core must not infer special
  treatment from target names.
