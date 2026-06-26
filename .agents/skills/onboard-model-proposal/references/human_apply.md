# Human Apply Reference

Use this file to understand the boundary after proposal review.

## Agent Boundary

The agent must stop at review-only outputs. It must not:

- edit `<run_dir>/config/approved_targets.json`
- promote proposal definitions/hints into `config/`
- edit `output/`
- edit `reports/`
- treat LLM output as approval
- run Modal as part of proposal repair

## Human Apply After Review

There is no automatic apply step in the standard workflow. After explicit human
approval, the reviewer edits these reviewed inputs directly:

- `<run_dir>/config/approved_targets.json`
- `<run_dir>/config/run_config.json`
- optional promoted non-FI definitions under `<run_dir>/config/definitions/`
- optional promoted non-FI hints under `<run_dir>/config/definition_hints/`

Every `role: "target"` entry copied into `approved_targets.json` must keep a
reviewed `capture` object. The run fails fast if a target omits it. Warmup
entries must not declare `capture`.

Do not treat proposal drafts as official definitions. For FlashInfer/fitrace
targets, standard collect writes definitions from the same-run fitrace dump. For
non-FI targets, a human must promote the reviewed draft files into `config/`
before collect can consume them.
