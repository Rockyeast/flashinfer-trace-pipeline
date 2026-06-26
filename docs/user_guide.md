# User Guide

This document covers the reviewed workflow for onboarding a model and validating a trace run.

## Quick Start

Use this path for a new model. Replace `<hf_model>` and `<model_slug>` with your target model values.

### 1. Prepare Inputs

Prepare the local agent input cache:

```bash
python3 -B -m tools.proposal_tools prepare-agent-inputs \
  --model <hf_model>
```

This downloads `agent_inputs/config/<model_slug>.json` and clones/updates `agent_inputs/sgl-cookbook/`.

```bash
# Required after prepare-agent-inputs.
ls agent_inputs/config/<model_slug>.json

# Optional but recommended for static source checks.
ls agent_inputs/flashinfer/flashinfer
ls agent_inputs/sglang/python/sglang

# Required for final validate.
pip install -e /path/to/flashinfer-bench
```

`prepare-agent-inputs` checks whether the FlashInfer and SGLang source roots exist, but it does not clone them. Prepare those source snapshots separately if you want stronger agent/source checks.

### 2. Generate Proposal

Generate one proposal prompt:

```bash
python3 -B -m tools.proposal_tools spawn-agents \
  --model <hf_model>
```

Run Codex automatically:

```bash
python3 -B -m tools.proposal_tools spawn-agents \
  --model <hf_model> \
  --agent codex
```

For multiple independent agents:

```bash
python3 -B -m tools.proposal_tools spawn-agents \
  --model <hf_model> \
  --count 3 \
  --agent codex
```

`spawn-agents` writes each agent's prompt/run under `runs/<model>/<date>_firstpass...`. With `--count > 1`, it creates sibling agent runs and merges proposals into a review-only merged proposal.

The agent follows `.agents/skills/onboard-model-proposal/SKILL.md` and writes:

```text
runs/<model>/<run_id>/
  proposal/
    architecture.md
    candidate_targets.json
    review_checklist.md
    definitions/
    definition_hints/
  config/
    run_config.json
```

The agent is expected to run `agent-loop` until the proposal is ready for human review.

### 3. Review And Approve

Review:

- `proposal/architecture.md`
- `proposal/candidate_targets.json`
- `proposal/review_checklist.md`
- `config/run_config.json`
- `proposal/definitions/` and `proposal/definition_hints/` for non-FI targets

Approve by writing reviewed artifacts:

```text
runs/<model>/<run_id>/config/
  approved_targets.json
  run_config.json
  definitions/
  definition_hints/
```

Minimum approval checklist:

- Every approved target has explicit `target`, `module`, and `attr`.
- FlashInfer collect targets point to APIs decorated with `@flashinfer_api(trace=...)`.
- Attention wrappers usually target the decorated `.run`; use `.forward` only as a companion when it passes arguments into `.run`.
- `definition_name` for fitrace-backed targets is only a preview; the final name comes from the fitrace dump.
- Known collectable non-FI ops, currently `rmsnorm` and `silu_and_mul`, have review-only definition/hints drafts before approval.
- Non-FI drafts are promoted from `proposal/definitions/` and `proposal/definition_hints/` into `config/definitions/` and `config/definition_hints/`.

Proposal tools do not approve anything. `check-proposal`, `agent-loop`, and `repair-loop` only validate or repair proposal artifacts.

### 4. Run Collect

```bash
python3 -B -m flashinfer_trace.cli run \
  --run <model>/<run_id>
```

`run` executes the reviewed collect path:

```text
remote SGLang probe
-> hook events/captures
-> FlashInfer fitrace dump
-> definition audit/repair
-> workload collect
-> local materialization
```

If the local terminal disconnects, copy the Function call ID from Modal and resume:

```bash
python3 -B -m flashinfer_trace.cli run \
  --run <model>/<run_id> \
  --resume-call-id fc-...
```

### 5. Validate

```bash
python3 -B -m flashinfer_trace.cli validate \
  --run <model>/<run_id>
```

`validate` is the final acceptance command. It runs local consistency checks, official-style layout/export checks, and upstream `flashinfer-bench validate` with GPU disabled. The run is accepted only when it prints:

```text
run accepted: True
```

Review:

```text
runs/<model>/<run_id>/reports/review.md
runs/<model>/<run_id>/reports/run_report.json
```

### 6. Repair If Needed

Do not edit `output/` directly. Use `repair-loop` to generate feedback and optionally invoke an external agent:

```bash
python3 -B -m tools.proposal_tools repair-loop \
  --run <model>/<run_id> \
  --hf-config agent_inputs/config/<model_slug>.json \
  --flashinfer-root agent_inputs/flashinfer/flashinfer
```

Automatic repair with an external agent:

```bash
python3 -B -m tools.proposal_tools repair-loop \
  --run <model>/<run_id> \
  --hf-config agent_inputs/config/<model_slug>.json \
  --flashinfer-root agent_inputs/flashinfer/flashinfer \
  --max-rounds 3 \
  --agent-command "codex exec -C <REPO_ROOT> -s workspace-write --ephemeral" -
```

`repair-loop` only repairs `proposal/`; it does not edit `config/`, does not edit `output/`, and does not run Modal. After repair, human-review/promote again, then rerun collect and validate.

## Reference

### Run Directory

Each run lives under:

```text
runs/<model>/<run_id>/
```

Directory layout:

```text
runs/<model>/<run_id>/
  proposal/
    architecture.md
    candidate_targets.json
    review_checklist.md
    definitions/
    definition_hints/
    proposal_check.json
    agent_feedback.md
    agent_loop.json
    repair_prompt.md
    repair_loop.json
    run_diagnostics.json
  config/
    approved_targets.json
    run_config.json
    definitions/
    definition_hints/
  output/
    definitions/
    workloads/
    blob/
  reports/
    run_report.json
    review.md
```

- `proposal/` is review-only agent output.
- `config/` is human-reviewed input consumed by runtime.
- `output/` is official-style staging data produced by the run.
- `reports/` contains the machine-readable run report and human review digest.

### Run Config

Minimal `config/run_config.json`:

```json
{
  "model_name": "<hf_model>",
  "image": "lmsysorg/sglang:v0.5.12.post1",
  "gpu": "L40S",
  "tp_size": 1,
  "timeout": 3600,
  "disable_cuda_graph": true,
  "batch_sizes": [1, 2, 4, 8, 16, 32, 64],
  "max_new_tokens": 96,
  "max_captures_per_target": 128,
  "supplemental_runs": [
    {
      "name": "sampling_supplemental",
      "sampling_params": {"temperature": 0.7, "top_k": 50, "top_p": 0.9},
      "allowed_op_types": ["sampling"]
    }
  ]
}
```

Runtime choices and collect strategy are reviewed config. The core does not infer GPU, TP, image, or workload coverage from the model name.

Collect uses `sharegpt_100.json` at the repository root as the prompt source. Remote prompt scenarios are derived from `batch_sizes` and `max_new_tokens`.

### Definition Sources

Definition sources are explicit:

- FlashInfer targets use the fitrace dump produced during the same Modal inference run.
- Reviewed non-FI definitions live under `config/definitions/`.
- Accepted definitions are staged under `output/definitions/`.

### Proposal Commands

Manual proposal check:

```bash
python3 -B -m tools.proposal_tools check-proposal \
  --proposal-dir runs/<model>/<run_id>/proposal \
  --hf-config agent_inputs/config/<model_slug>.json \
  --flashinfer-root agent_inputs/flashinfer/flashinfer
```

Manual multi-agent merge:

```bash
python3 -B -m tools.proposal_tools merge-proposals \
  --proposal-dir runs/<model>/<agent_run_a>/proposal \
  --proposal-dir runs/<model>/<agent_run_b>/proposal \
  --proposal-dir runs/<model>/<agent_run_c>/proposal \
  --output-dir runs/<model>/<merged_run>/proposal
```

`merge-proposals` deduplicates candidates and unions evidence. Conflicts are written to `merge_review.md` and `merge_report.json`; they must be resolved by review.

### Result Artifacts

Human review usually starts here:

```text
reports/review.md
```

Machine-readable details live here:

```text
reports/run_report.json
```

Reviewable outputs:

```text
output/definitions/
output/workloads/
output/blob/
```

Captures are raw argument snapshots created when hooks fire. They are intermediate artifacts used by definition audit/repair and sanitization. Standard local results do not retain captures long-term.

## Troubleshooting

### Early Stop

Normal collect stops early when a target fails audit/sanitize. Later targets may show zero events because they were not executed. That does not mean those targets are invalid, and they should not be changed to `collect: false` just because of early stop.

`repair-loop` detects early-stop cases and includes the reason in proposal feedback.

To gather more diagnostics in one run:

```bash
python3 -B -m flashinfer_trace.cli run \
  --run <model>/<run_id> \
  --diagnostic-full-scan
```

Diagnostic full scan keeps running after early failures to expose more issues. Its captures are for debugging, not final collect.

### Incremental Collect

The same run directory can be passed to `run` multiple times. Each round adds or overwrites the specified targets while preserving data for other targets.

- To skip a target and keep existing data: set `collect: false`.
- To re-collect a target and replace existing data: keep `collect: true`.
- The workload manifest is merged automatically.

### Modal Resume

If local execution disconnects after Modal has started, resume with the Modal Function call ID:

```bash
python3 -B -m flashinfer_trace.cli run \
  --run <model>/<run_id> \
  --resume-call-id fc-...
```

Resume only materializes the existing remote result. It does not launch a second SGLang run.
