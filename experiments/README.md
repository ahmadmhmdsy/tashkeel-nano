# experiments/ - provenance, not pipeline

This directory holds 179 programs (932 KiB) copied **verbatim** from the
originating monorepo (`nano_SLMs`, D-line). They are here so that every number
quoted in `docs/evaluation.md`, `docs/training.md` and the two model cards can be
traced back to the program that produced it. They are historical records, not a
supported interface.

## What this project actually runs

The maintained entry points are `scripts/` (data, training, scoring, serving) and
the configs under `configs/`. See `docs/reproduction.md` for the end-to-end
recipe. Exactly one script here is part of that recipe:

- `e97_build_bed.py` - streaming builder for the `v3full` ctx-256 full-charset bed
  (the bed the 1.08 M model was trained on). `docs/data.md` documents its CLI.

## Why the paths inside them look wrong

The upstream monorepo kept everything under a `diac/` level. This project drops
that level, so a path you read inside these files maps as follows:

| path written in an experiment | real path here |
|---|---|
| `data/diac/raw/...` | `data/raw/...` |
| `data/diac/prepared_rebuilt/...` | `data/prepared_rebuilt/...` |
| `data/diac/v3*/tokens/...` | `data/v3*/tokens/...` |
| `runs/diac/<run>/...` | `runs/<run>/...` |
| `research/experiments.yaml`, `research/EXPERIMENTS.md` | `research/` (verbatim) |
| `research/verdict_audit/` records | `experiments/verdict_audit/` (tools + `evidence/`, `parts/`) |

The same applies to the `runs/diac/...` provenance lines in
`models/nano1p1m/README.md` and `models/nano4p5m/README.md`: they name run records
in the upstream monorepo, which is why they are quoted rather than rewritten.

## Expect most of them not to run

These programs were written against checkpoints, gate caches and intermediate
beds that are deliberately **not** distributed (they are large, and several belong
to abandoned branches). Running one is only meaningful if you have reproduced the
matching artifact first. They are kept, not maintained: they are excluded from the
test suite and the CI job only byte-compiles them.
