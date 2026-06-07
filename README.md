# FlashInfer Trace Pipeline

This repository contains the automated pipeline for discovering FlashInfer
kernel definitions from serving traces, collecting workloads, generating
baseline solutions/reference tests, and validating the resulting dataset files.

This repository is for the pipeline code itself. Pipeline runs write generated
files into isolated directories under `tmp/run/...`; those files should be
reviewed separately before being promoted into a dataset root.

## What This Pipeline Does

The main flow is:

```text
probe SGLang runtime
  -> parse observed flashinfer.* API evidence
  -> generate definition JSON
  -> collect workloads
  -> generate baseline solutions
  -> generate reference tests
  -> run schema / dataset validation
```

## Quick Start

Install the local CLI dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run a fast model onboarding pass:

```bash
python3 scripts/run_pipeline.py \
  --model-name ModelOrg/ModelName
```

Review the generated files under `tmp/run/...` before promoting them to the dataset.

## Repository Layout

```text
scripts/
  run_pipeline.py                  main pipeline entrypoint
  generate_definitions.py          definition generation
  collect_workloads_modal.py       workload collection
  generate_baseline_solutions.py   baseline solution generation
  generate_tests.py                reference test generation
  adapters/                        per-kernel adapter modules
  artifact_schemas.py              shared artifact schema models
  audit_schemas.py                 local schema audit CLI
  fixtures/                        prompt fixtures used by probe/collect
  parse/                           probe parsing and inventory construction
  pipeline/                        top-level pipeline orchestration helpers
  probe/                           Modal/SGLang runtime probe
  static/                          static candidates from HF config + SGLang source
  test_generators/                 adapter-backed pytest generation

docs/
  README.md                        operational guide and CLI reference
  INTERNALS.md                     parser, adapter, and evidence-boundary internals

tools/
  promote_run_to_dataset.py         review and merge a run into a dataset root
  run_reference_tests_modal.py      run generated reference tests on Modal GPU
  propose_parse_rules.py            build review-only proposals for unknown trace IDs
  apply_llm_kernel_proposal.py      write reviewed proposals as adapter drafts
  clean_local_outputs.py            clean local run/dev-check outputs after review
```

Detailed command examples for these tools are in
[docs/README.md](docs/README.md#tools--post-review-utilities).

## Generated File Boundary

The pipeline writes generated files into isolated run directories by default:

```text
tmp/run/<model>_<timestamp>/
```

Within a run directory, generated files use the same layout as the target
dataset:

```text
definitions/
workloads/
blob/
solutions/
traces/
tests/references/
```

Keep these generated files in the run directory while reviewing them. Promote
only approved files into a separate dataset checkout with `tools/promote_run_to_dataset.py`.

## Promoting Files

After reviewing a run, merge approved files into a separate dataset checkout:

```bash
python3 tools/promote_run_to_dataset.py \
  --run-dir tmp/run/ModelOrg_ModelName_YYYYMMDD_HHMMSS \
  --dataset-dir /path/to/clean/flashinfer-trace \
  --source-model ModelOrg/ModelName \
  --dry-run
```

`--dry-run` prints a report of what would be copied without writing any files.
Once the report looks correct, re-run the same command without `--dry-run` to apply the changes.

## Documentation

- [docs/README.md](docs/README.md): operational guide and CLI reference
- [docs/INTERNALS.md](docs/INTERNALS.md): parser, adapter, and evidence-boundary internals
