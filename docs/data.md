# Data

Everything the downloader does is declared in `configs/data.yaml`. The script
contains no source-specific logic: adding a corpus is a config edit.

    python scripts/download_data.py --list                 # show the catalog
    python scripts/download_data.py                        # every non-optional source
    python scripts/download_data.py --source fadel,wikinews
    python scripts/download_data.py --accept-license       # include the gated source

## Layout

    data/raw/<dir>/...        files exactly as published upstream
    data/raw/provenance.json  per source: url, license, commercial_use,
                              acceptance, byte count, sha256, elapsed time

## Sources

| id | kind | what it is | license | commercial |
|---|---|---|---|---|
| `fadel` | http | Fadel train/val/test, one diacritized sentence per line | research use, check upstream | unclear |
| `wikinews` | http | QCRI WikiNews 2014/2024 multi-ref benchmarks + Wikipedia corpus | see upstream | unclear |
| `sadeed_25` | http | SadeedDiac-25 expert-reviewed benchmark | check the dataset card | unclear |
| `sadeed_tashkeela` | hf | Sadeed Tashkeela input/output pairs (Tashkeela-derived) | GPL-2.0-derived, copyleft | **no** |
| `abdou_tashkeel` | hf | Abdou Arabic Tashkeel Dataset, ~1.5 M rows | MIT | yes |
| `qcri_diac_clone` | git | optional clone, only used as a val pseudo-source | see upstream | unclear |

`sadeed_tashkeela` is the largest training source and the reason the model
weights are research-only. It is marked `requires_acceptance: true`, so it is
**skipped** unless you pass `--accept-license`. That flag is an acknowledgement
that you have read the upstream terms — nothing more is implied.

The downloader never silently drops a source: a gated source reports `BLOCKED`,
a network failure reports `FAIL`, and the process exit code reflects both.

## Which splits are used for what

| split | role |
|---|---|
| `fadel/train.txt`, `fadel/val.txt` | training |
| `fadel/test.txt` | **external gate** — downloaded, never enters the prepared shards |
| `abdou_tashkeel` train/valid | training |
| `abdou_tashkeel` **test** | **held-out witness** — never trained on by any model in this line |
| `sadeed_tashkeela` train | training (overlap-flagged against the sadeed25 gate) |
| `sadeed_25` | registered gate |
| `wikinews2014` | registered gate |
| `wikinews2024` | contaminated — reported, excluded from the mean |
| `wikipedia_diac.jsonl` | large unregistered training pool |

`scripts/dedup_check.py` measures the Sadeed overlap and writes
`data/bench/sadeed_overlap_report.json`. It is CPU-only and safe to run beside
a training job.

## Preparing a training corpus

`scripts/prepare_data.py` reads diacritized raw lines and writes an aligned
corpus:

1. parse every line with the passthrough codec, which produces the base characters
   and the 15-class label sequence in one pass;
2. quarantine anything that fails to parse into `quarantine@<source>.txtl` —
   never a silent skip;
3. window to at most `CTX_CHARS = 128` characters, splitting **only** at spaces
   (never mid-word, never inside a number or URL run);
4. deduplicate documents on a normalised text hash;
5. split train/val by document, stratified by `MD5(seed)` rank, so the same
   document never straddles both sides;
6. write `train.jsonl`, `val.jsonl`, a provenance file and
   `prepare_stats.json`.

The seed is `20260911`, so the run is reproducible.

By default the parser drops units that carry no diacritic information. Setting

    PREP_KEEP_ALL=1

keeps **every** character in the alphabet instead. Both published models descend
from the `KEEP_ALL` corpus — it is the recorded source of both token beds — so
use it when you want to reproduce them.

    PREP_KEEP_ALL=1 python scripts/prepare_data.py --out-dir data/prepared_rebuilt

## From a prepared corpus to a token bed

A prepared corpus is text; a bed is fixed-shape integer arrays.

`scripts/tokenize_data.py` is the general packer: it shuffles the prepared rows
with a fixed seed, fills `ctx`-wide rows, and writes
`data/<phase>/tokens/{train,val}_{ids,y}.npy`.

    python scripts/tokenize_data.py --phase v3 --ctx 128 --pad-label 0 \
        --prepared-dir data/prepared_rebuilt
    python scripts/tokenize_data.py --phase v3ctx256 --ctx 256 --pad-label 0 \
        --prepared-dir data/prepared_rebuilt

`experiments/e97_build_bed.py` is the streaming builder for the
**full-character-set ctx-256** bed. It exists because the jsonl intermediate for
that bed is ~8 GB and the bed itself another ~6 GB, so the builder streams rows
instead of materialising them, and it reproduces `pack_rows`'s permutation
exactly so row *j* of the new bed is the same underlying text as row *j* of the
control bed.

    python experiments/e97_build_bed.py --prepared-dir data/prepared_rebuilt \
        --out-dir data/v3full/tokens --ctx 256 --pad-label 0

### The two beds, and why the difference matters

| bed | ctx | alphabet | used by |
|---|---|---|---|
| `v3` | 128 | spaces **stripped** | nano4p5m |
| `v3ctx256` | 256 | spaces **stripped** | the e69/e70 lineage steps |
| `v3full` | 256 | spaces **kept** (real space tokens, 15.28 % of positions) | nano1p1m |

The 4.5 M model was trained on a bed with no space tokens, so its space embedding
row is untrained and ordinary spaced text is a slightly out-of-distribution
input for it. The 1.1 M model was trained on `v3full`, so plain text is exactly
right for it. This single difference is the reason the two models are quoted
under different decode arms.

### PAD convention

`tokenize_data.py` emits label `-1` at padded positions by default, while every
recorded bed in this line uses ` 0`. The two differ **only** at pad positions and
the external gates are unaffected either way, but validation losses are not
comparable across the two conventions. Pass `--pad-label 0` to produce the
lineage convention in one command, which is what every command in this
documentation does.

## Provenance

`data/raw/provenance.json` is written on every downloader run. Each record carries
the source id, kind, homepage, license, commercial-use flag, whether an
acceptance was given, and a sha256 plus byte count for every file fetched. It is
the audit trail for the licensing table in the [README](../README.md), and it is
regenerated rather than edited by hand.
