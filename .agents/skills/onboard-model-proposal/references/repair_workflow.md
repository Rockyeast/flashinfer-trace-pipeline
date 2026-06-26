# Repair Workflow Reference

Use this file in repair-pass mode.

## Standard Repair Entry

If a previous Modal/run attempt produced `<run_dir>/reports/run_report.json`,
run:

```bash
python3 -B -m tools.proposal_tools repair-loop \
  --run <run_dir> \
  --hf-config <config.json> \
  --flashinfer-root <flashinfer_source_root>
```

This writes:

```text
<run_dir>/proposal/repair_prompt.md
<run_dir>/proposal/agent_feedback.md
<run_dir>/proposal/agent_loop.json
```

Read `repair_prompt.md` first, then `agent_feedback.md`, then the existing
proposal files. Revise only the review-only proposal bundle. Do not rerun Modal
from inside the agent loop.

## Agent Feedback

`agent_feedback.md` is feedback for the next repair pass. It can come from:

- `agent-loop`: deterministic check of proposal/candidate/definition/hints
  shape.
- `repair-loop`: run-report diagnostics compressed into repair-pass guidance.

It does not fix anything by itself. The agent reads it, edits proposal files,
then reruns the deterministic proposal gate.

## Uncollected Definitions

If feedback lists `definition_audit.uncollected`, each definition must be
triaged.

Add a candidate target when the source call chain shows the definition is part
of the desired workload surface.

If it is an internal/helper definition that should not be collected, write
`<run_dir>/proposal/ignored_definitions.json` as a JSON list:

```json
[
  {
    "name": "definition_name",
    "reason": "source-backed reason this definition is not a workload target"
  }
]
```

Every ignored definition needs a concrete source-backed reason. Do not use this
file to hide unknowns; unknowns stay `FIX_REQUIRED`.

## diagnose-run

`diagnose-run --run <run_dir>` is available when only a compact
`agent_feedback.md` is needed. Prefer `repair-loop` as the standard repair
entry because it emits the fixed prompt and reruns the deterministic proposal
gate.
