# Official Skill Map

The official skills are useful references, not the main control flow for the redesign.

## How to use them

Use them to copy standards, expected artifacts, and validation expectations. Do not resurrect the old automatic end-to-end flow unless the user explicitly asks.

## Relevant official skills

- `extract-kernel-definitions`: definition naming, tags, axes, TP/EP handling, op_type to FlashInfer API hints.
- `collect-workloads`: real SGLang workload collection, FlashInfer logging behavior, selective tensor dump, ragged/paged runtime flags.
- `add-reference-tests`: reference test expectations after a definition is reviewed.
- `validate-dataset`: dataset validation and audit expectations.
- `onboard-model`: old high-level phase map. Treat it as a reference checklist, not an automatic workflow.
- `track-models`: model coverage and discovery ideas.
- `clone-repos`: source setup ideas.

## Redesign boundary

The current `onboard-model-proposal` skill only writes review-only proposal files:

- `architecture.md`
- `candidate_targets.json`
- `review_checklist.md`
- review-only non-FI draft files under `proposal/definitions/` and
  `proposal/definition_hints/`, when needed
- first-pass `config/run_config.json` as the reviewed runtime-config starting
  point

It does not write official definitions, workloads, blobs, traces, PRs, or commits.
