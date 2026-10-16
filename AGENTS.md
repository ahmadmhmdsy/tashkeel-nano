# AGENTS.md - tashkeel-nano repository router

Standalone repo for tiny Arabic diacritization (tashkeel) models and the
pipeline that trains them. Carved out of the `nano_SLMs` monorepo (D-line);
`experiments/` and `research/` are verbatim provenance from upstream, not a
supported interface.

## Read first (session bootstrap)

1. `README.md` - what the project is, the two shipped models, quickstart.
2. `docs/` - `architecture.md`, `data.md`, `training.md`, `evaluation.md`,
   `reproduction.md` (end-to-end recipe).
3. `CURRENT_STATE`-equivalent: there is none - this repo ships no live state
   file. Treat `README.md`, the model cards and `research/experiments.yaml`
   as the source of truth for results; `research/EXPERIMENTS.md` holds the
   narrative verdicts.

## Map

| path | role |
|---|---|
| `src/tashkeel_nano/` | the package: `model.py`, `inference.py`, `labels.py`, `tokenizer.py`, `passthrough.py`, eval/metrics, `pipeline/` (monitor, feature registry) |
| `scripts/` | maintained CLI: data, training, scoring, serving, gates |
| `configs/` | `data.yaml` (download catalog), model/stage configs, `lineage/` (the pretrain->fine-tune chain) |
| `models/nano1p1m/`, `models/nano4p5m/` | the two shipped model packages (weights + cards) |
| `demo/` | Gradio app + static browser demo (compares the two, plus third-party baselines) |
| `experiments/` | 179 provenance programs + `verdict_audit/` tools, `evidence/`, `parts/` |
| `research/` | registry (`experiments.yaml`), verdicts (`EXPERIMENTS.md`), design, plans - provenance |
| `tests/` | pytest suite (CI runs it) |
| `data/`, `runs/` | never committed; fetch via `scripts/download_data.py`, regenerate by recorded commands |

## Entry points

- `scripts/selftest.py` - parser / labels / quarantine gates
- `scripts/train.py`, `scripts/train_lm.py` - registered training
- `scripts/prepare_data.py`, `tokenize_data.py`, `eval.py`, `eval_multi_ref.py`,
  `postproc.py`, `dedup_check.py`, `score_gate.py`, `soup.py`, `infer.py`
- `src/tashkeel_nano/` - tokenizer, labels, passthrough codec

## Smallest validation

    python scripts/selftest.py --phase all
    python -m pytest -q

CI (`.github/workflows/ci.yml`) additionally runs the demo helper tests, an
inference smoke test on both shipped packages, `count_params`, and the
markdown/hygiene checks. Reproduce a claim by its artifact (metric file,
gate-bench JSON, tfevents) under the run's own directory - never a console
number.

## Data boundaries

- `data/` holds fetched corpora and derived beds; raw subsets regenerate only
  by their recorded commands and are never deleted or rewritten as cleanup.
- `models/*/weights/`, `models/*/soup_members/`, `models/*/gate_probe/` are
  local-only staging (gitignored); rebuild step probes with `scripts/soup.py`.

## Hazards

- `train.py` / `train_lm.py` are GPU workloads; never run two at once, and
  never beside an inference job on the same device.
- The shipped anchors (1.08 M, 4.49 M) stay the gold models until an experiment
  record changes that; new-gold claims need artifact evidence.
- Postprocessing edits can silently change eval semantics: re-run the
  registered gate on the unchanged split after any output rewrite.

## Evidence and records

- Experiment source of truth: `research/experiments.yaml` (`diacritizer/E-xxx`);
  narrative verdicts: `research/EXPERIMENTS.md`.
- `experiments/` and `research/` are read-only provenance; do not "fix" the
  stale `runs/diac/...` paths inside them - `experiments/README.md` maps them.
