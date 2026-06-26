---
name: onboard-model-proposal
description: Use an agent/skill workflow to create or refine a review-only onboarding proposal for a new model or kernel. Use when analyzing HF config, SGLang/FlashInfer source, or diagnostics to propose candidate targets without applying them.
---

# Onboard Model Proposal

Create a **review-only** proposal for the FlashInfer trace pipeline.
The skill owns discovery. The Python core only initializes, validates, consumes
reviewed config, and runs deterministic probe/collect/validate.

## Core Contract

The agent proposes. A human approves. The deterministic pipeline consumes
reviewed `config/approved_targets.json`; FlashInfer definitions come from the
same-run fitrace dump unless a reviewed non-FI definition has been promoted into
`config/definitions/`.

Do not turn this skill into a Python rule engine. Do not add discovery rules to
core while onboarding a model.

## References

Load these references only when needed:

- `references/definition_standards.md`: definition naming, tags, axes, TP/EP,
  and FlashInfer API standards.
- `references/candidate_targets.md`: `candidate_targets.json` schema, fitrace
  target rules, capture rules, warmup entries, and non-FI proposal rules.
- `references/runtime_config.md`: `config/run_config.json` fields and reviewed
  collect strategy.
- `references/workload_collection_notes.md`: probe/collect boundary, tensor
  storage, companion capture, and runtime coverage notes.
- `references/non_fi_patterns.md`: common non-FI proposal patterns for
  capture, definition drafts, references, and hints.
- `references/repair_workflow.md`: repair-pass flow, `repair-loop`,
  `agent_feedback.md`, and uncollected definition triage.
- `references/human_apply.md`: what humans promote after review and what the
  agent must not apply.
- `references/official_skill_map.md`: old/official skills as reference material
  only.

The source of truth is current HF config, current SGLang source, and current
FlashInfer `@flashinfer_api(trace=...)` templates.

## Inputs

Use whichever inputs are available, but prefer complete roots over cherry-picked
files:

- HF `config.json`, preferably `agent_inputs/config/<model_slug>.json`
- SGLang source root, e.g. `<root>/sglang/python/sglang`, plus any provided
  model implementation hint such as `srt/models/llama.py`
- FlashInfer source root, e.g. `<root>/flashinfer/flashinfer`
- sgl-cookbook root, e.g. `agent_inputs/sgl-cookbook`
- existing `reports/run_report.json` diagnostics, especially `parse_report`

Follow the call chain from model implementation to SGLang backend to
FlashInfer wrapper/template before proposing a hook target. Do not assume
missing source means a kernel is absent; mark uncertainty explicitly.

## Entry Modes

Choose the mode from the requested run directory.

### First-pass proposal

Use this mode when no proposal exists yet, or when the user asks to onboard a
new model/kernel from source.

Write:

```text
<run_dir>/
  proposal/
    architecture.md
    candidate_targets.json
    review_checklist.md
    definitions/          # review-only non-fitrace drafts, when needed
    definition_hints/     # review-only non-fitrace hints, when needed
  config/
    run_config.json       # reviewed runtime-config starting point
```

Do not write official `approved_targets.json`, official `definitions/`,
`workloads/`, `blob/`, or commits.

### Repair-pass proposal

Use this mode when `<run_dir>/proposal/repair_prompt.md` exists, or when the
user asks to repair a previous run. Read `repair_prompt.md` first, then
`agent_feedback.md`, then the existing proposal files.

Revise only the review-only proposal bundle:

```text
<run_dir>/proposal/candidate_targets.json
<run_dir>/proposal/architecture.md
<run_dir>/proposal/review_checklist.md
<run_dir>/proposal/definitions/
<run_dir>/proposal/definition_hints/
<run_dir>/proposal/ignored_definitions.json
```

Do not edit `config/approved_targets.json`, `output/`, `reports/`, or committed
source code. Do not run Modal. Do not rewrite `config/run_config.json` unless
`repair_prompt.md` explicitly says the runtime config is wrong.

## Workflow

1. Read source, config, diagnostics, and relevant references yourself. If a
   model implementation hint is provided, start there, then use `rg`, imports,
   and call sites to follow the path into backend and wrapper code. Do not stop
   at a high-level model boundary if lower-level runtime targets are available.

2. Fill `proposal/architecture.md` with:

   - model structure
   - attention type and source locations
   - all runtime-relevant kernel/component families found in source
   - source-backed call chains from model code to backend/wrapper functions
   - uncertain or conflicting evidence

3. Fill `proposal/candidate_targets.json`. Read
   `references/candidate_targets.md` before writing target entries. For
   definition draft details, also read `references/definition_standards.md`.

4. Fill `proposal/review_checklist.md` so the reviewer can see confidence,
   uncertainty, missing evidence, warmup candidates, non-FI draft review needs,
   skipped reachable APIs, proposed runtime config, and the next command after
   approval.

5. Write `config/run_config.json` in first-pass mode. Read
   `references/runtime_config.md` before writing it. Runtime config belongs in
   JSON, not only in markdown.

6. Run the deterministic proposal gate before handing work back:

```bash
python3 -B -m tools.proposal_tools agent-loop \
  --proposal-dir <proposal_dir> \
  --hf-config <config.json> \
  --flashinfer-root <flashinfer_source_root>
```

Read `agent_feedback.md`. If it says `FIX_REQUIRED`, edit only the allowed
proposal files and run the same command again. Repeat until it prints
`ready for human review: True`, or until a real source-evidence blocker is
written in `review_checklist.md`.

## Run Diagnostics And Repair

If a previous Modal/run attempt produced `<run_dir>/reports/run_report.json`,
use `repair-loop` as the standard repair entry:

```bash
python3 -B -m tools.proposal_tools repair-loop \
  --run <run_dir> \
  --hf-config <config.json> \
  --flashinfer-root <flashinfer_source_root>
```

Then read `proposal/repair_prompt.md` and follow repair-pass mode. See
`references/repair_workflow.md` for `definition_audit.uncollected`,
`ignored_definitions.json`, and `diagnose-run`.

## Human Review Boundary

The agent must stop at review-only outputs. It must not approve targets,
promote definitions/hints, edit official reviewed config, or apply generated
drafts. See `references/human_apply.md` for the human apply step after explicit
approval.

## What Not To Do

- Do not add an automatic apply/run pipeline inside this skill.
- Do not use keyword scanning as the discovery mechanism.
- Do not run static candidates directly into official definitions.
- Do not use LLM output as automatic approval.
- Do not change core code while onboarding a model unless the user explicitly
  asks.
- Do not approve because a target looks plausible; verify the target in source
  or runtime diagnostics.
- Do not propose plain GEMM, matmul, or linear targets during standard model
  onboarding unless the user explicitly asks for GEMM coverage.

## Useful Checks

The real validation is `agent-loop` on the proposal directory. Use
`agent-loop --help` only to inspect command options. Run `py_compile` only if
the user explicitly asked you to change repository Python code.
