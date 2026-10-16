# research/ - provenance registry, not a supported interface

Verbatim records carried over from the originating monorepo (`nano_SLMs`,
D-line) so that every decision and number in this project can be traced to the
experiment that produced it. These are historical records, kept, not
maintained.

| file / dir | what it is |
|---|---|
| `experiments.yaml` | the machine-readable experiment registry (`diacritizer/E-xxx`); the source of truth for what was run and its verdict |
| `EXPERIMENTS.md` | the narrative ledger - one row per experiment with the reasoning |
| `verdict_audit.md`, `verdict_audit.yaml` | audit hub for the verdict history (reconstructs the record from `experiments/verdict_audit/` tools + evidence) |
| `MODEL_EVALUATION_REPORT.md`, `wn2024_*`, `eval_wn2024_multiref_corrected.json` | the external multiref evaluation evidence for the shipped models |
| `arabic_diacritization/` | design brief (`DESIGN.md`) + three deep-research surveys |
| `archive/diacritizer_failures.md` | the recorded dead ends |
| `plans/` | dated design/experiment plans for the D-line |

Paths inside these files look wrong because they point at the upstream layout:
`data/diac/...` -> `data/...`, `runs/diac/<run>/...` -> `runs/<run>/...`. See
`experiments/README.md` for the full mapping. The verdict-audit *tools* and
their `evidence/` live in `experiments/verdict_audit/`.

Markdown here is excluded from `scripts/lint_markdown.py` (that linter only
guards published pages); these files are quoted as written upstream.
