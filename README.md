# tashkeel-nano

**Tiny Arabic diacritization (tashkeel / تشكيل) models — and the pipeline that
trains them, in one repository.**

Two published checkpoints, the exact data recipe behind them, the training code,
the weight-soup step that produced the smaller one, the registered evaluation,
and two runnable demos (a Gradio app and a browser-only ONNX page).

| | [tashkeel-nano-1p1m](https://huggingface.co/ahmadsy/tashkeel-nano-1p1m) | [tashkeel-nano-4p5m](https://huggingface.co/ahmadsy/tashkeel-nano-4p5m) |
|---|---|---|
| parameters | **1,083,328** (1.08 M) | **4,488,160** (4.49 M) |
| fp32 size | 4.15 MiB | 17.2 MB |
| body | 18L · 64 hidden · 256 FFN · 8 heads / 2 KV · ctx 256 | 12L · 160 hidden · 640 FFN · 8 heads / 2 KV · ctx 128 |
| training bed | `v3full` (ctx 256, real space tokens) | `v3` (ctx 128, spaces stripped) |
| published artifact | uniform soup of 12 step probes (32,500 → 60,000) | step 19,500 (best val loss) |
| registered 3-gate DER | **11.51 %** | **22.90 %** (fair decode, output normalised to NFC — see below) |
| decode | `centered` 256/128, NFC output | `nospace_centered`; the package emits shadda-first |

Live demo: **https://huggingface.co/spaces/ahmadsy/tashkeel-compare**

---

## Table of contents

- [What this is](#what-this-is)
- [Results](#results)
- [Install](#install)
- [Inference in 30 seconds](#inference-in-30-seconds)
- [Retrain the published models](#retrain-the-published-models)
- [Train on your own data](#train-on-your-own-data)
- [Evaluate](#evaluate)
- [Demos](#demos)
- [Repository layout](#repository-layout)
- [Licensing](#licensing)
- [Honesty notes and limitations](#honesty-notes-and-limitations)
- [Citation](#citation)

---

## What this is

Arabic text is normally written without its diacritics (harakat). Restoring them
is *tashkeel*, and it is hard because the marks are partly lexical and partly
syntactic. Most published systems are 30 M – 13 B parameters.

These two models are **character-level bidirectional Transformer encoders with a
15-class per-position head**. They are deliberately tiny: the point is the
size/quality trade, and the 1.08 M model beats several systems that are hundreds
of times larger on the same external gates.

The 15 classes are

    bare, fatha, damma, kasra, sukun, fathatan, dammatan, kasratan,
    shadda, shadda_fatha, shadda_damma, shadda_kasra,
    shadda_fathatan, shadda_dammatan, shadda_kasratan

Non-Arabic content (digits, Latin, emoji, tatweel, Quranic annotation,
invisibles) is **passthrough**: it stays visible to the encoder as context, is
never a prediction target, and is re-inserted byte-exact, so
`reconstruct(parse(x)) == x` always holds.

## Results

DER = diacritic error rate, **lower is better**, reported as a percentage. The
**registered metric** is the mean DER over three external corpora —
`fadel_test`, `sadeed25`, `wikinews2014` — under the decode arm that is
fairest for each model. `wikinews2024` is excluded (its text overlaps the
training pool) and `abdou_test` is the never-trained-on witness.

| model | fadel_test | sadeed25 | wikinews2014 | **registered mean** |
|---|---|---|---|---|
| **tashkeel-nano-1p1m** (`centered`, NFC) | 5.20 | 11.71 | 17.64 | **11.51** |
| tashkeel-nano-1p1m (`single`, NFC) | 5.87 | 12.44 | 18.63 | 12.31 |
| **tashkeel-nano-4p5m** (`nospace_centered`, output normalised to NFC) | 14.30 | 22.59 | 31.81 | **22.90** |
| tashkeel-nano-4p5m (`nospace_centered`, shadda-first as emitted) | 27.34 | 38.20 | 45.14 | 36.89 |
| tashkeel-nano-4p5m (`single`, NFC) | 19.70 | 30.30 | 36.31 | 28.77 |
| tashkeel-nano-4p5m (as recorded in its card, pre-NFC renderer) | 31.31 | 43.37 | 47.19 | 40.62 |

Two things in that table need saying out loud, because both have been published
wrongly at some point:

1. **Each model is quoted under its own fair decode.** The 4.5 M model was
   trained on a bed with **no space tokens**, so its space embedding row was
   never trained. Feeding it ordinary spaced text hands the encoder an untrained
   embedding. Stripping whitespace (`nospace`) and using a centered window is
   the fair setting for it, and it moves the registered mean from 40.62 → 22.90.
   The 1.1 M model was trained on a bed **with** real space tokens, so plain text
   is the correct input for it.
2. **Mark order matters.** Combining marks must be emitted in Unicode NFC order
   (fatha, ccc 30, sorts before shadda, ccc 33). A shadda-first renderer disagrees
   with every NFC reference on every geminate. The registered references are NFC,
   so the 22.90 row is the 4.5 M model's own output **normalised to NFC**; the
   package emits shadda-first, which costs 36.89 on the same three gates. The
   other two 4.5 M rows keep the shape those figures were first published in.

`scripts/score_gate.py` reproduces this table from raw corpora:

    python scripts/score_gate.py --run models/nano1p1m \
        --config training_config.yaml --decode centered --order nfc

## Install

The **inference and evaluation path needs no training stack** — the tokenizer,
the label schema and the passthrough codec are stdlib-only, and the metric code
is pure Python.

    git clone https://github.com/ahmadmhmdsy/tashkeel-nano
    cd tashkeel-nano
    python -m venv .venv
    # Windows:  .venv\Scripts\python -m pip install -r requirements.txt
    .venv/bin/python -m pip install -r requirements.txt

Nothing needs to be built. The package is importable straight from `src/`, and
every script inserts it on `sys.path` itself, so you can run the commands below
without installing anything.

## Inference in 30 seconds

### Python

    import sys; sys.path.insert(0, "src")
    from tashkeel_nano import TashkeelNano

    model = TashkeelNano.from_pretrained("ahmadsy/tashkeel-nano-1p1m")
    print(model.diacritize("السلام عليكم ورحمة الله"))

The decode defaults (window, stride, mark order, whitespace handling) come from
`tashkeel_nano.inference.PRESETS` and match each published package exactly, so
you do not have to know them to get the published behaviour.

### Command line

    python scripts/infer.py --repo ahmadsy/tashkeel-nano-1p1m --text "السلام عليكم"
    python scripts/infer.py --repo ahmadsy/tashkeel-nano-4p5m --file in.txt --out out.txt
    python scripts/infer.py --run runs/e63s_s4_12l160/final --text "مرحبا"

One input line → one output line, UTF-8. Text with no Arabic letters is echoed
unchanged. `--decode {single,centered,nospace,nospace_centered}`,
`--order {nfc,shadda_first}`, `--window`, `--stride` and `--nospace` override the
preset when you want to reproduce an experiment.

## Retrain the published models

Everything below is the recorded recipe, not a reconstruction. Shapes and row
counts that must match are listed in [docs/reproduction.md](docs/reproduction.md).

    # 1. raw corpora (config-driven; --accept-license is needed for the GPL-2 source)
    export HF_TOKEN=hf_...                      # PowerShell: $env:HF_TOKEN="hf_..."
    python scripts/download_data.py --accept-license

    # 2. the full-character-set prepared corpus — the reference source of both beds
    PREP_KEEP_ALL=1 python scripts/prepare_data.py --out-dir data/prepared_rebuilt

    # 3. the two token beds
    python experiments/e97_build_bed.py --prepared-dir data/prepared_rebuilt \
        --out-dir data/v3full/tokens --ctx 256 --pad-label 0        # 1p1m bed
    python scripts/tokenize_data.py --phase v3 --ctx 128 --pad-label 0 \
        --prepared-dir data/prepared_rebuilt                        # 4p5m bed
    python scripts/tokenize_data.py --phase v3ctx256 --ctx 256 --pad-label 0 \
        --prepared-dir data/prepared_rebuilt                        # lineage bed

    # 4. the shared stage-1 causal character LM
    python scripts/prep_rawlm.py
    python scripts/train_lm.py --config configs/stage1_lm.yaml

    # 5. the 4.5 M model  (stage1lm -> e63s_s4_12l160)
    python scripts/train.py --config configs/nano4p5m.yaml

    # 6. the 1.1 M model  (stage1lm -> e67_p1_lm64 -> e67_c1_ft64_fullv3
    #                      -> e69_c4_ctx256 -> e98_soupB)
    python scripts/train_lm.py --config configs/lineage/e67_p1_lm64.yaml
    python scripts/train.py    --config configs/lineage/e67_c1_ft64_fullv3.yaml
    python scripts/train.py    --config configs/lineage/e69_c4_ctx256.yaml
    python scripts/train.py    --config configs/nano1p1m.yaml

    # 7. the published 1.1 M artifact is NOT the final step: it is the uniform
    #    mean of 12 step probes. This rebuilds it bit-for-bit from the probes.
    python scripts/soup.py --run runs/e98_soupB --members 32500:60000:2500 \
        --out runs/e98_soupB/soupB_uniform

    # 8. score it
    python scripts/score_gate.py --run runs/e98_soupB/soupB_uniform \
        --decode centered --order nfc

`scripts/train.py` has a **zero-flag auto-resume contract**: after any crash or
kill, re-run the exact same command and it picks up the newest checkpoint under
`runs/<phase>/checkpoint-*` (model + optimizer + step). There is nothing to
remember and nothing to add.

### Disk

The derived beds are much larger than the raw corpora. Budget roughly:

| path | size |
|---|---|
| `data/raw` (all sources) | ~2.7 GB |
| `data/prepared_rebuilt` | ~8.0 GB |
| `data/v3full/tokens` (ctx 256) | ~6.0 GB |
| `data/v3/tokens` (ctx 128) | ~3.0 GB |
| `data/v3ctx256/tokens` | ~6.0 GB |
| `data/stage1/tokens` | ~5.1 GB |

`data/`, `runs/` and `out/` are gitignored — nothing large is ever committed.
Only the two published model packages (~22 MB) and the two ONNX graphs
(~22 MB) live in the repository.

## Train on your own data

The pipeline is config-driven end to end, so adapting it is a config edit rather
than a code edit.

**Bring your own corpus.** Add an entry to `configs/data.yaml` — `kind: http`
for plain files, `kind: hf` for a Hugging Face dataset, `kind: git` for a repo
clone — then fetch it by name:

    python scripts/download_data.py --source my_corpus

`license` and `commercial_use` are recorded into `data/raw/provenance.json`
verbatim. A source marked `requires_acceptance: true` is skipped unless you pass
`--accept-license`; that flag means "I have read the upstream terms and accept
them for my use", nothing more.

**Bring your own text.** `scripts/prepare_data.py` reads diacritized lines from
`data/raw` (edit `SOURCES` or point `--sources` at your own ids) and writes an
aligned, deduplicated, document-stratified corpus. Then tokenize and train:

    PREP_KEEP_ALL=1 python scripts/prepare_data.py --out-dir data/mine
    python scripts/tokenize_data.py --phase mine --ctx 256 --pad-label 0 \
        --prepared-dir data/mine
    cp configs/nano1p1m.yaml configs/mine.yaml     # edit phase/tokens_phase/model
    python scripts/train.py --config configs/mine.yaml

**Bring your own model shape.** `model:` in a config takes `layers`, `hidden`,
`n_heads`, `n_kv`, `ffn`, `ctx`, `dropout`, `causal`, `kv_pattern`,
`shared_layers`. Check a shape before you train it:

    python scripts/count_params.py

## Evaluate

    # the registered gate, with predictions written to disk
    python scripts/score_gate.py --run runs/e98_soupB/soupB_uniform \
        --decode centered --order nfc --out-json out/gate.json

    # score predictions you already have
    python scripts/score_gate.py --pred-dir out/gate

    # one prediction file against one reference file
    python scripts/eval.py compare --pred pred.txt --ref ref.txt

The metric is **word-level**: a word counts as correct when its mark string
matches *any* reference for that line. It is deliberately *not* Sadeed's
letter-level "DER (CE)"; their word-level analogue is called WER (CE) there.
Mixing the two produces numbers that look comparable and are not.
[docs/evaluation.md](docs/evaluation.md) spells out the semantics, the
contamination rules and the decode arms.

## Demos

    # browser-only ONNX demo: no server, no upload, runs in the tab
    python -m http.server 7863 --directory demo/static

    # Gradio app: our models plus published third-party models side by side
    python -m pip install -r demo/gradio/requirements.txt
    python demo/gradio/app.py          # http://127.0.0.1:7862

Both ship in the repository. `demo/static/` is the edition published as the
Hugging Face Space `ahmadsy/tashkeel-compare`, its card included, so uploading
that directory’s contents publishes your own copy; `demo/gradio/` is the
fuller comparison app. The Gradio app’s helper tests live in
`demo/gradio/tests/`: three are pure Python and run in CI, the fourth
(`test_app.py`) needs the `gradio` extra and a model download.

## Repository layout

    configs/            data catalog, model/training configs, lineage configs
    src/tashkeel_nano/  the package: model, tokenizer, labels, passthrough,
                        inference, metrics, pipeline helpers
    scripts/            the curated CLIs (download, prepare, tokenize, train,
                        infer, soup, score, normalize-marks, eval, report,
                        selftest)
    experiments/        every experiment script behind the results, plus
                        verdict_audit/ with the analysis tools
    models/nano1p1m/    the published 1p1m package (weights + card + code copy)
    models/nano4p5m/    the published 4p5m package
    demo/gradio/        Gradio comparison app, its helper tests/ beside it
    demo/static/        browser-only ONNX demo (the published Space, card included)
    tests/              the test suite that CI runs
    docs/               architecture, data, training, evaluation, reproduction
    data/               fetched corpora and derived beds (gitignored)

## Licensing

Three separate layers — do not collapse them:

| layer | terms |
|---|---|
| **source code** | MIT, see [LICENSE](LICENSE) |
| **model weights** | research-use-only, see [LICENSE-MODELS](LICENSE-MODELS) |
| **training corpora** | per-source, see the license and commercial_use fields in [configs/data.yaml](configs/data.yaml) and the provenance.json the downloader writes |

The largest training source is GPL-2.0-derived, and some corpora are
research-only or unclear. **Nothing in this repository grants you rights to any
third-party corpus.** The downloader records what it fetched so you can audit it,
and gated sources are never fetched without an explicit acknowledgement.

## Honesty notes and limitations

- **`sadeed25` is overlap-flagged.** Sadeed_Tashkeela is in the training mix, so
  part of the SadeedDiac-25 gate is close to the training distribution. It is
  kept in the registered mean because it was pre-registered, but read it with
  that caveat. `scripts/dedup_check.py` measures the overlap.
- **`wikinews2024` is contaminated** — its text is in the training pool. It is
  reported for continuity only and is **excluded** from the registered mean.
- **`abdou_test` is the honest witness.** It was never trained on by any model in
  this line. It is reported separately, never folded into the mean.
- **DER here is word-level**, not Sadeed's letter-level DER (CE). See above.
- **The 4.5 M model's published card reports 40.62**, a number produced by a
  renderer that predates the NFC mark-order fix. The corrected fair-decode figure
  under this project's registered scorer is 22.90, and the card carries a
  correction section. Both readings were reproduced here independently; see
  [docs/evaluation.md](docs/evaluation.md#re-running-this-table).
- **Small models fail differently.** Both models occasionally emit a wrong case
  ending on a construct they have not seen; the `DER_nocase` column isolates
  that from genuine diacritic errors.

## Citation

    @software{tashkeel_nano,
      title  = {tashkeel-nano: tiny Arabic diacritization models},
      author = {ahmadmhmdsy},
      year   = {2026},
      url    = {https://github.com/ahmadmhmdsy/tashkeel-nano}
    }

See [CITATION.cff](CITATION.cff) for the machine-readable form.
