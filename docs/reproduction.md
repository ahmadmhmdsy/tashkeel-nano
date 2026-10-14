# Reproduction

The exact commands, in order, with the counts each step must produce. If a count
differs, stop there: a later number will not be comparable.

## 0. Environment

    git clone https://github.com/ahmadmhmdsy/tashkeel-nano
    cd tashkeel-nano
    python -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt

No install step is required for the scripts; each one puts `src/` on
`sys.path` itself.

## 1. Raw corpora

    export HF_TOKEN=hf_...          # PowerShell: $env:HF_TOKEN="hf_..."
    python scripts/download_data.py --accept-license

Expect `data/raw` to hold `fadel/`, `wikinews/`, `sadeed_25/`,
`sadeed_tashkeela/` and `abdou_tashkeel/`, and
`data/raw/provenance.json` to record every file with its sha256. Roughly 2.7 GB.

`sadeed_tashkeela` is gated: without `--accept-license` it reports `BLOCKED`
and the exit code is non-zero. That is the intended behaviour, not a failure.

## 2. The full-character-set prepared corpus

    PREP_KEEP_ALL=1 python scripts/prepare_data.py --out-dir data/prepared_rebuilt

Expect `data/prepared_rebuilt/{train,val}.jsonl` and `prepare_stats.json` with

    train_windows  2,433,430
    val_windows      160,214

and **no `wikinews2024` entry** — the reference corpus deliberately excludes it.
This is the single most expensive step: the jsonl intermediate is ~8 GB and it is
CPU-only.

## 3. The token beds

    python experiments/e97_build_bed.py --prepared-dir data/prepared_rebuilt \
        --out-dir data/v3full/tokens --ctx 256 --pad-label 0

    python scripts/tokenize_data.py --phase v3ctx256 --ctx 256 --pad-label 0 \
        --prepared-dir data/prepared_rebuilt

    python scripts/tokenize_data.py --phase v3 --ctx 128 --pad-label 0 \
        --prepared-dir data/prepared_rebuilt

Expected, from `data/v3full/tokens/build_stats.json`:

| bed | ctx | train rows | val rows |
|---|---|---|---|
| `v3full` | 256 | 2,433,430 | 160,214 |
| `v3ctx256` | 256 | 2,433,430 | 160,214 |
| `v3` | 128 | 2,433,430 | 160,214 |

`--pad-label 0` is mandatory for parity: the default is `-1`, which differs only
at padded positions but makes validation losses incomparable with the recorded
runs.

Row **order** is also part of the contract. `e97_build_bed.py` reproduces
`pack_rows`'s shuffle permutation by construction, so row *j* of `v3full` and
row *j* of `v3ctx256` carry the same underlying text.

## 4. Stage-1 character LM

    python scripts/prep_rawlm.py
    python scripts/train_lm.py --config configs/stage1_lm.yaml

Expect `data/stage1/prep_stats.json` with

    train_windows  1,226,329
    val_windows       12,580
    excluded_gate_docs 7,103

The run itself is 12,000 steps at ctx 512, batch 8, accumulation 4.

## 5. The 4.5 M model

    python scripts/train.py --config configs/nano4p5m.yaml

20,000 steps on `v3`, warm-started from `runs/stage1lm/final/model.pt`. The
published artifact is **step 19,500**, selected by best validation loss
(0.175752287773707), not the final step. `runs/e63s_s4_12l160/final/model.pt`
holds it.

## 6. The 1.1 M model

    python scripts/train_lm.py --config configs/lineage/e67_p1_lm64.yaml
    python scripts/train.py    --config configs/lineage/e67_c1_ft64_fullv3.yaml
    python scripts/train.py    --config configs/lineage/e69_c4_ctx256.yaml
    python scripts/train.py    --config configs/nano1p1m.yaml

The chain of initialisations is the part that is easy to get wrong:

| run | init |
|---|---|
| `e67_p1_lm64` | `runs/stage1lm/final/model.pt` (via `tokens_dir: v3s10`) |
| `e67_c1_ft64_fullv3` | `runs/e67_p1_lm64/final/model.pt` |
| `e69_c4_ctx256` | `runs/e67_c1_ft64_fullv3/gate_probe/best_gate_weights.pt` |
| `e98_soupB` | `runs/e69_c4_ctx256/final/model.pt` |

`e67_c1_ft64_fullv3` is configured for 200,000 steps; the run that was actually
carried forward is the gate-probe snapshot named by `e69`'s init, not its
`final/`.

`e67_p1_lm64` reads the `v3s10` phase, a 10 % prefix of the `v3` arrays. It is a
prefix of the *post-shuffle* arrays, so build it with the head flags rather than
by truncating the corpus:

    python scripts/tokenize_data.py --phase v3s10 --ctx 128 --pad-label 0 \
        --prepared-dir data/prepared_rebuilt --head-train 243343 --head-val 9613

## 7. The published 1.1 M artifact

    python scripts/soup.py --run runs/e98_soupB --members 32500:60000:2500 \
        --out runs/e98_soupB/soupB_uniform

This is the step that produces the released weights. Twelve probes
(32,500, 35,000, … 60,000) are averaged elementwise in fp32 and cast back to the
members' dtype. `soup.json` records every member path, its sha256, and the
relative L2 distance from the soup to each member — the basin check. The command
is CPU-only and takes a couple of minutes.

The published `model.safetensors` for 1p1m is this soup, not
`runs/e98_soupB/final/model.pt`. Those two files are **different**; using the
final step instead of the soup loses the whole gain.

## 8. Verify

    python scripts/selftest.py --phase all

    python scripts/score_gate.py --run models/nano1p1m \
        --config training_config.yaml --decode centered --order nfc

    python scripts/score_gate.py --run models/nano4p5m \
        --config training_config.yaml --decode nospace_centered --order shadda_first

The first should report the registered mean of **11.51** and the second **22.90**
(percent). Both write their predictions next to their references so the number
can be traced.

## Parity checks that are cheap and worth doing first

    # the tokenizer, the label schema and the passthrough codec round-trip
    python scripts/selftest.py --phase all

    # one sentence through the published 1p1m package
    python scripts/infer.py --repo ahmadsy/tashkeel-nano-1p1m \
        --text "السلام عليكم ورحمة الله"

    # the same sentence through the local copy of the same package
    python scripts/infer.py --run models/nano1p1m \
        --config training_config.yaml --text "السلام عليكم ورحمة الله"

Those two outputs must be identical, character for character.
