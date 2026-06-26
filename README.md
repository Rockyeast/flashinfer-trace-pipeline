# FlashInfer Trace

FlashInfer Trace is a reviewed workflow for discovering model runtime targets,
collecting FlashInfer/non-FI workload artifacts, and validating the generated
dataset layout.

The public workflow is intentionally small:

1. Generate a review-only proposal with the `onboard-model-proposal` skill.
2. Human-review the proposal and write reviewed config under `runs/<model>/<run_id>/config/`.
3. Run collect:

   ```bash
   python3 -B -m flashinfer_trace.cli run --run <model>/<run_id>
   ```

4. Validate the completed run:

   ```bash
   python3 -B -m flashinfer_trace.cli validate --run <model>/<run_id>
   ```

5. If the run exposes proposal-level issues, generate repair feedback:

   ```bash
   python3 -B -m tools.proposal_tools repair-loop --run <model>/<run_id>
   ```

See `docs/user_guide.md` for the full command reference and run directory
layout.

## Repository Layout

- `flashinfer_trace/core/`: reviewed config planning, capture, event/workload
  sanitization, and definition audit.
- `flashinfer_trace/definition_repairs/`: kernel-specific definition repair and
  hint rules.
- `flashinfer_trace/runners/`: Modal/SGLang runner integration.
- `tools/proposal_tools.py`: proposal, multi-agent merge, diagnostics, and
  repair-loop helpers.
- `.agents/skills/onboard-model-proposal/`: agent skill for first-pass and
  repair-pass proposal generation.
- `tests/`: active unit tests.

Runtime inputs are explicit reviewed files. The core does not infer model
runtime settings from model names, hidden defaults, or legacy external
definition directories.
