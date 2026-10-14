# tashkeel-nano

**Tiny Arabic diacritization (tashkeel / تشكيل) models — and the pipeline that
trains them, in one repository.**

Two published checkpoints, the exact data recipe behind them, the training code,
the weight-soup step that produced the smaller one, the registered evaluation,
and two runnable demos (a Gradio app and a browser-only ONNX page).

**The corpora and the training runs are not in this repository**: it ships the
code, the configs, the two model packages and the two ONNX graphs — about 70 MB
in total. Everything else is rebuilt by the commands in
[Retrain the published models](#retrain-the-published-models); [Disk](#disk)
lists what each command creates and how big it is.

| | [tashkeel-nano-1p1m](https://huggingface.co/ahmadsy/tashkeel-nano-1p1m) | [tashkeel-nano-4p5m](https://huggingface.co/ahmadsy/tashkeel-nano-4p5m) |
|---|---|---|
| parameters | **1,083,328** (1.08 M) | **4,488,160** (4.49 M) |
| fp32 size | 4.15 MiB | 17.2 MiB |
| body | 18L · 64 hidden · 256 FFN · 8 heads / 2 KV · ctx 256 | 12L · 160 hidden · 640 FFN · 8 heads / 2 KV · ctx 128 |
| training bed | `v3full` (ctx 256, real space tokens) | `v3` (ctx 128, spaces stripped) |
| published artifact | uniform soup of 12 step probes (32,500 → 60,000) | step 19,500 (best val loss) |
| registered 3-gate DER | **11.51 %** | **22.90 %** (fair decode, output normalised to NFC — see below) |
| decode | `centered` 256/128, NFC output — preset and fair arm | `nospace_centered`, NFC — fair arm; the package preset is `single`, spaced and shadda-first |

Live demo: **https://huggingface.co/spaces/ahmadsy/tashkeel-compare**

---

## Table of contents

- [What this is](#what-this-is)
- [Results](#results)
- [Install](#install)
- [Inference in 30 seconds](#inference-in-30-seconds)
- [Corpora and data](#corpora-and-data)
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
syntactic. The published systems this project compares against run from 12.5 M
parameters to 13 B.

These two models are **character-level bidirectional Transformer encoders with a
15-class per-position head**. They are deliberately tiny: the point is the
size/quality trade. On the SadeedDiac-25 leaderboard — scored with that
benchmark’s own letter-level DER (CE), which is *not* the word-level metric used
below — the 1.08 M model’s 3.78 is ahead of every quoted system above 1.5 B
parameters (the six 7 B–13 B rows run 26–79) and behind only Claude-3.7-Sonnet and
Gemini-Flash-2.0. The whole table, each row marked *quoted* or *measured here*,
is in the live demo and in `demo/static/data/reference_outputs.json`.

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
   the fair setting for it: scored as emitted (shadda-first) that arm is 36.89,
   and the same predictions normalised to NFC give 22.90. The 1.1 M model was
   trained on a bed **with** real space tokens, so plain text is the correct
   input for it and `centered` is its fair arm.
2. **Mark order matters.** Combining marks must be emitted in Unicode NFC order
   (fatha, ccc 30, sorts before shadda, ccc 33). A shadda-first renderer disagrees
   with every NFC reference on every geminate. The registered references are NFC,
   so the 22.90 row is the 4.5 M model's own output **normalised to NFC**; the
   package emits shadda-first, which costs 36.89 on the same three gates. The
   other two 4.5 M rows keep the shape those figures were first published in.

`scripts/score_gate.py` reproduces this table from raw corpora. Both models are
committed, so the only prerequisite is the corpora — step 1 in
[Retrain](#retrain-the-published-models), `python scripts/download_data.py
--accept-license`, ~2.7 GB:

    # 1.1 M, fair arm (11.51)
    python scripts/score_gate.py --run models/nano1p1m \
        --config training_config.yaml --decode centered --order nfc

    # 4.5 M, fair arm (22.90). --order nfc normalises the emitted shadda-first
    # text to the order the registered references use; omit it and the very
    # same predictions score 36.89.
    python scripts/score_gate.py --run models/nano4p5m \
        --config training_config.yaml --decode nospace_centered --order nfc

## Install

The tokenizer, the label schema and the passthrough codec are stdlib-only, and
the metric code is pure Python, so **scoring needs no torch and no training
stack**; only the model paths do. Python 3.9 or newer.

    git clone https://github.com/ahmadmhmdsy/tashkeel-nano
    cd tashkeel-nano
    python -m venv .venv
    # Windows:  .venv\Scripts\python -m pip install -e ".[hub]"
    .venv/bin/python -m pip install -e ".[hub]"

`.[hub]` is torch + numpy + PyYAML plus `huggingface_hub` and `safetensors`,
which is everything the Python snippet and `scripts/infer.py --repo` need. Add
`.[data]` for the pandas/pyarrow corpora pipeline, `.[demo]` for Gradio, and
`.[onnx]` to re-export the browser graphs;
`requirements.txt` is the flattened equivalent of `.[hub,data,demo]` if you would
rather not install the package.

The commands below say `python`; use the interpreter from the venv you just
created (`.venv/bin/python`, or `.venv\Scripts\python` on Windows). The package
is importable straight from `src/` — the scripts that need it insert `src/` on
`sys.path` themselves — so `pip install` is only strictly required for the
`from tashkeel_nano import ...` snippet.

## Inference in 30 seconds

### Python

    import sys; sys.path.insert(0, "src")
    from tashkeel_nano import TashkeelNano

    model = TashkeelNano.from_pretrained("ahmadsy/tashkeel-nano-1p1m")
    print(model.diacritize("السلام عليكم ورحمة الله"))
    # السَّلَامُ عَلَيْكُمْ وَرَحْمَةُ اللَّهِ

The decode defaults (window, stride, mark order, whitespace handling) come from
`tashkeel_nano.inference.PRESETS` and reproduce exactly what each published
package emits. Two of them matter:

- **1p1m** — `centered` 256/128, NFC output. This is also its fair arm, so
  `from_pretrained` gives you the 11.51 % in the table.
- **4p5m** — `single`, spaced text, shadda-first: the package’s literal
  behaviour, and *not* its fair arm, because that model was trained without
  space tokens. For the registered 22.90 % pass the fair decode:
  `from_pretrained("ahmadsy/tashkeel-nano-4p5m", decode="nospace_centered", order="nfc")`.

Both models share two input behaviours worth knowing:

- **Existing diacritics in the input are stripped and re-predicted.** `strip_marks()`
  drops every combining mark (U+064B-U+0652, U+0670, U+0653-U+065F,
  U+06D6-U+06ED) and NFC-normalises what is left, so fully vocalised text comes
  back re-marked by the model rather than echoed.
- **Text longer than the window is decoded in overlapping windows.** `centered`
  walks the line in windows of `--window` (256 for 1p1m) advancing by `--stride`
  (128), and each character keeps the mark from the window whose centre is
  nearest. `single` and `nospace` decode fixed non-overlapping chunks of `ctx`
  instead. Non-Arabic characters are never marked, and a line with no Arabic
  letters is returned unchanged.

### Command line

    python scripts/infer.py --repo ahmadsy/tashkeel-nano-1p1m --text "السلام عليكم"
    python scripts/infer.py --repo ahmadsy/tashkeel-nano-4p5m --file in.txt --out out.txt

    # the committed local copy of the 1.1 M package, no download needed
    python scripts/infer.py --run models/nano1p1m --config training_config.yaml \
        --text "مرحبا"

    # the 4.5 M package’s fair decode (see above)
    python scripts/infer.py --run models/nano4p5m --config training_config.yaml \
        --decode nospace_centered --order nfc --text "مرحبا"

One input line → one output line, UTF-8. Text with no Arabic letters is echoed
unchanged. For `السلام عليكم ورحمة الله`, the 1.1 M package prints
`السَّلَامُ عَلَيْكُمْ وَرَحْمَةُ اللَّهِ`; the 4.5 M package prints the same letters with the
shadda before the fatha at every geminate - visually identical in most terminals,
different in code-point order, and the reason two rows of the table above differ
by 14 DER points on the same weights. `--decode {single,centered,nospace,nospace_centered}`,
`--order {nfc,shadda_first}`, `--window`, `--stride` and `--nospace` override the
preset when you want to reproduce an experiment.

## Corpora and data

`scripts/download_data.py` reads `configs/data.yaml`, so adding a corpus is a config
edit. Hosts, licences, acceptance flags and which split is used for what are
catalogued in [docs/data.md](docs/data.md) and, machine-readably, in
[configs/data.yaml](configs/data.yaml):

| id | what it is | host | licence recorded | role |
|---|---|---|---|---|
| `fadel` | Fadel Arabic diacritization split | github.com/AliOsm/arabic-text-diacritization | research use, check upstream | training + `fadel_test` gate |
| `wikinews` | QCRI WikiNews 2014/2024 multi-reference benchmarks + Wikipedia corpus | github.com/qcri/advancing-arabic-diacritization | see upstream, research use | `wikinews2014` gate; `wikinews2024` is contaminated |
| `sadeed_25` | SadeedDiac-25 expert-reviewed benchmark | HF `Misraj/SadeedDiac-25` | check the dataset card | `sadeed25` gate |
| `sadeed_tashkeela` | Sadeed Tashkeela input/output pairs (Tashkeela-derived) | HF `Misraj/Sadeed_Tashkeela` | GPL-2.0-derived, copyleft | training; largest source, gated |
| `abdou_tashkeel` | Abdou Arabic Tashkeel Dataset | HF `Abdou/arabic-tashkeel-dataset` | MIT | training + `abdou_test` witness |
| `qcri_diac_clone` | optional repository clone | github.com/qcri/advancing-arabic-diacritization | see upstream | optional val pseudo-source |

    python scripts/download_data.py --list            # print the same catalogue
    python scripts/download_data.py --source fadel,abdou_tashkeel
    python scripts/download_data.py --accept-license  # include the gated source

Fetched files land under `data/raw/<id>/` exactly as published, and
`data/raw/provenance.json` records each one's url, licence, acceptance flag, byte
count and sha256. A gated source without `--accept-license` reports `BLOCKED` and
the exit code is non-zero - intended behaviour, not a failure.

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
    python scripts/tokenize_data.py --phase v3s10 --ctx 128 --pad-label 0 \
        --prepared-dir data/prepared_rebuilt \
        --head-train 243343 --head-val 9613        # 10% prefix, read by e67_p1_lm64

    # 4. the shared stage-1 causal character LM
    python scripts/prep_rawlm.py
    python scripts/train_lm.py --config configs/stage1_lm.yaml

    # 5. the 4.5 M model  (stage1lm -> e63s_s4_12l160)
    #    the run's final/ is step 20,000. The published artifact is step 19,500
    #    (best val loss 0.175752287773707), and that is what the package ships.
    python scripts/train.py --config configs/nano4p5m.yaml

    # 6. the 1.1 M model  (stage1lm -> e67_p1_lm64 -> e67_c1_ft64_fullv3
    #                      -> e69_c4_ctx256 -> e98_soupB)
    python scripts/train_lm.py --config configs/lineage/e67_p1_lm64.yaml
    python scripts/train.py    --config configs/lineage/e67_c1_ft64_fullv3.yaml
    python scripts/train.py    --config configs/lineage/e69_c4_ctx256.yaml
    python scripts/train.py    --config configs/nano1p1m.yaml

    # 7. the published 1.1 M artifact is NOT the final step: it is the uniform
    #    mean of the 12 step probes below. With the same probe set the mean is
    #    deterministic (fp32, cast back to the members' dtype), so this rebuilds
    #    the released weights exactly; a different probe set is a different soup.
    python scripts/soup.py --run runs/e98_soupB --members 32500:60000:2500 \
        --out runs/e98_soupB/soupB_uniform

    # 8. score it
    python scripts/score_gate.py --run runs/e98_soupB/soupB_uniform \
        --decode centered --order nfc

    # 9. optional: rebuild the browser ONNX graphs from the packages
    python -m pip install -e ".[onnx]"
    python scripts/export_onnx.py --run models/nano1p1m \
        --out demo/static/models/nano1p1m.onnx --verify
    python scripts/export_onnx.py --run models/nano4p5m \
        --out demo/static/models/nano4p5m.onnx --verify

The recipe block above is POSIX shell. `VAR=1 cmd` and a trailing `\` are not
PowerShell: on Windows set `$env:HF_TOKEN="hf_..."` and `$env:PREP_KEEP_ALL="1"`
first, put each command on one line, and use `.venv\Scripts\python`; Git Bash or
WSL runs the block unchanged.

`scripts/train.py` has a **zero-flag auto-resume contract**: after any crash or
kill, re-run the exact same command and it picks up the newest checkpoint under
`runs/<phase>/checkpoint-*` (model + optimizer + step). There is nothing to
remember and nothing to add.

The recorded runs, on one 6 GB Quadro RTX 3000 with fp16 autocast: the 4.5 M model
20,000 steps in about 2.2 h, the final 1.1 M run (`e98_soupB`) 60,000 steps in about
9.0 h. Each published package's `training_summary.json` records its step count, best
val loss and wall time.

### Disk

**None of these paths is in this repository.** Each one is created on your own
machine by the step named beside it; the sizes are from the recorded run, so
treat them as a budget rather than a manifest.

| path | created by | approx. on disk | in git? |
|---|---|---|---|
| `data/raw/` | step 1, `download_data.py` (default sources; `qcri_diac_clone` is optional and skipped unless named) | ~2.7 GB | no |
| `data/prepared_rebuilt/` | step 2, `PREP_KEEP_ALL=1 prepare_data.py` | ~8.0 GB | no |
| `data/v3full/tokens/` | step 3, `e97_build_bed.py` — the 1p1m bed, ctx 256 | ~6.0 GB | no |
| `data/v3/tokens/` | step 3, `tokenize_data.py --phase v3` — the 4p5m bed, ctx 128 | ~3.0 GB | no |
| `data/v3ctx256/tokens/` | step 3, `tokenize_data.py --phase v3ctx256` — the lineage bed | ~6.0 GB | no |
| `data/v3s10/tokens/` | step 3, `tokenize_data.py --phase v3s10` — a 10 % prefix of `v3`, read by `e67_p1_lm64` | ~3.0 GB | no |
| `data/stage1/tokens/` | step 4, `prep_rawlm.py` | ~5.1 GB | no |
| `runs/` | steps 4–7, `train.py` / `train_lm.py` / `soup.py` | checkpoints + optimizer state | no |
| `out/` | `score_gate.py --out-json` | small | no |

`.gitignore` keeps `/data/`, `/runs/`, `/out/` and `/scratch/` out of git at the
repository root, together with the local-only model staging directories
(`models/*/weights/`, `models/*/soup_members/`, `models/*/checkpoint-*/`,
`models/*/gate_probe/`).

What **is** committed, and how big it is: the two published model packages are
**≈44.7 MB**, because each ships the same weights twice as `model.pt` and
`model.safetensors` (4.37 + 4.35 MB for 1p1m, 17.99 + 17.96 MB for 4p5m); the
two browser ONNX graphs are **≈23.1 MB** (4.81 + 18.27 MB); everything else is
code, configs and docs. **Repository total ≈69.5 MB.**

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
`shared_layers`. Count the parameters of a config, or of an ad-hoc shape,
before you train it:

    python scripts/count_params.py --config configs/mine.yaml
    python scripts/count_params.py --layers 12 --hidden 160 --ffn 640 --ctx 128

With no arguments it prints the recorded e63s / candidate grid instead.

## Evaluate

The gate commands predict *and* score, so they need the corpora first:
`python scripts/download_data.py --accept-license` (step 1 above, ~2.7 GB).
`--out` is where the `<source>.pred.txt` / `<source>.ref.txt` pairs are written;
`--pred-dir` reads them back.

`--config` is resolved **inside the `--run` directory**, so `--run models/nano4p5m
--config training_config.yaml` reads `models/nano4p5m/training_config.yaml`. Drop
`--config` and it looks for `config.yaml` there, which the packages do not ship under
that name - another reason the commands above pass it.

    # the registered gate, with predictions and a summary written to disk
    python scripts/score_gate.py --run models/nano1p1m \
        --config training_config.yaml --decode centered --order nfc \
        --out out/gate --out-json out/gate_1p1m.json

    # after a retrain, the same command on the published soup instead
    # python scripts/score_gate.py --run runs/e98_soupB/soupB_uniform \
    #     --decode centered --order nfc --out out/gate

    # score predictions you already have
    python scripts/score_gate.py --pred-dir out/gate

    # one prediction file against one reference file
    python scripts/eval.py compare --pred pred.txt --ref ref.txt

The metric is **word-level**: a word counts as correct when its mark string
matches *any* reference for that line. A `/` **inside** a token is not expanded
into alternatives — that is the registered anchor's own behaviour; the
alternative-aware view is `der_anyalt` in `tashkeel_nano.eval_metrics`, and on
the gate corpora the two disagree on ~0.3–0.6 % of tokens. The metric is
deliberately *not* Sadeed's letter-level "DER (CE)"; their word-level analogue
is called WER (CE) there. Mixing the two produces numbers that look comparable
and are not.
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

### The ONNX graphs the browser demo runs

`demo/static/models/nano1p1m.onnx` and `nano4p5m.onnx` are the same weights the
packages ship, exported for `onnxruntime-web`:

| | value |
|---|---|
| input | `input_ids`, int64 `[batch, seq]`, character ids from `tashkeel_nano.tokenizer` |
| output | `logits`, fp32 `[batch, seq, 15]` - raw class scores, no softmax |
| opset / IR | 17 / 8, batch and sequence dimensions dynamic |
| size | 4.81 MB (1p1m), 18.27 MB (4p5m) |

Mark order, whitespace handling and windowing live in the renderer
(`tashkeel_nano.inference` in Python, `demo/static/js/engine.js` in the browser), so
one graph serves every decode arm and the two models differ only in weights. opset 17
has no `RMSNormalization` op, so the export traces each RMSNorm as arithmetic. Rebuild
them with step 9 above, or:

    python -m pip install -e ".[onnx]"
    python scripts/export_onnx.py --run models/nano4p5m \
        --out demo/static/models/nano4p5m.onnx --verify

`--verify` runs the result through `onnxruntime` and compares it with the torch
module on random ids. `tests/test_onnx_parity.py` checks the stronger property the
demo claims, and CI runs it: the committed graphs and the packages must pick the
**same diacritic class at every position** of a real sentence plus random rows (the
logits differ by under 4e-5, so this is a tight check). Re-exporting is allowed to
produce a file that is not byte-identical to the committed one - constant folding
folds slightly differently - as long as that test still passes.

## Repository layout

    configs/            data catalog, model/training configs, lineage configs
    src/tashkeel_nano/  the package: model, tokenizer, labels, passthrough,
                        inference, metrics, pipeline helpers
    scripts/            the curated CLIs (download, prepare, tokenize, train,
                        infer, soup, score, export-onnx, normalize-marks, eval,
                        report, selftest)
    experiments/        every experiment script behind the results, plus
                        verdict_audit/ with the analysis tools
    models/nano1p1m/    the published 1p1m package (weights + card + code copy)
    models/nano4p5m/    the published 4p5m package
    demo/gradio/        Gradio comparison app, its helper tests/ beside it
    demo/static/        browser-only ONNX demo (the published Space, card included)
    tests/              the test suite that CI runs
    docs/               architecture, data, training, evaluation, reproduction
    data/               fetched corpora and derived beds (not in git; see Disk)

## Licensing

Four separate layers — do not collapse them:

| layer | terms |
|---|---|
| **source code** | MIT, see [LICENSE](LICENSE) |
| **model weights** | research-use-only, see [LICENSE-MODELS](LICENSE-MODELS); each package also ships its own copy (`models/nano1p1m/LICENSE`, `models/nano4p5m/LICENSE`) |
| **training corpora** | per-source, see the `license` and `commercial_use` fields in [configs/data.yaml](configs/data.yaml) and the `provenance.json` the downloader writes |
| **`demo/static/` (browser demo)** | Apache-2.0 for its code, as its card states; the ONNX graphs it bundles are redistributions of the research-use-only weights and stay under `LICENSE-MODELS` |

`LICENSE-MODELS` is the authoritative statement for the published weights.
`configs/data.yaml` records each source's own upstream terms — the largest
training source is GPL-2.0-derived — and no field there changes the terms of the
weights. **Nothing in this repository grants you rights to any third-party
corpus.** The downloader records what it fetched so you can audit it, and gated
sources are never fetched without an explicit acknowledgement.

## Honesty notes and limitations

- **`sadeed25` is overlap-flagged.** Sadeed_Tashkeela is in the training mix, so
  part of the SadeedDiac-25 gate is close to the training distribution. It is
  kept in the registered mean because it was pre-registered, but read it with
  that caveat. `scripts/dedup_check.py` measures the overlap.
- **`wikinews2024` is contaminated** — its text is in the training pool. It is
  reported for continuity only and is **excluded** from the registered mean.
- **`abdou_test` is the honest witness.** It was never trained on by any model in
  this line. It is reported separately, never folded into the mean. The 4.5 M run
  (`single` arm) scores DER 39.30 / `DER_nocase` 29.30 / WER 39.65 /
  `text_preservation` 0.9846, and the 1.1 M run in the arm its package ships
  (`centered`, NFC) scores DER 10.23 / `DER_nocase` 7.77 / WER 10.69 /
  `text_preservation` 0.9846 over 15,091 sentences / 1.54 M words
  ([docs/evaluation.md](docs/evaluation.md)). The two arms are not the same, so
  read those as two measurements, not as one head-to-head.
- **DER here is word-level**, not Sadeed's letter-level DER (CE). See above. The
  4.5 M card's `model-index` value **9.91** is Sadeed's DER (CE) on SadeedDiac-25
  under that benchmark's own evaluator and mark-order convention; it is not
  comparable with the 22.90 in the table here.
- **The 4.5 M model's published card reports 40.62**, a number produced by a
  renderer that predates the NFC mark-order fix. The corrected fair-decode figure
  under this project's registered scorer is 22.90, and the card carries a
  correction section. Both readings were reproduced here from the committed
  weights; see [docs/evaluation.md](docs/evaluation.md#re-running-this-table).
- **The 1.1 M mean is quoted as 11.5.** The published figure is 11.51 and a
  re-measurement of the same soup gives 11.50 - a one-word argmax tie-flip between
  batch and single-sequence decode, not a change in the model. The third decimal is
  not stable; [docs/evaluation.md](docs/evaluation.md#re-running-this-table) shows
  the per-gate split.
- **Small models fail differently.** Both models occasionally emit a wrong case
  ending on a construct they have not seen; the `DER_nocase` variant (case
  endings dropped on both sides) isolates that from genuine diacritic errors. It
  is reported per gate in [docs/evaluation.md](docs/evaluation.md#recorded-results);
  the summary table above has no such column.

## Citation

    @software{tashkeel_nano,
      title   = {tashkeel-nano: tiny Arabic diacritization models},
      author  = {ahmadmhmdsy},
      version = {1.0.0},
      year    = {2026},
      url     = {https://github.com/ahmadmhmdsy/tashkeel-nano}
    }

See [CITATION.cff](CITATION.cff) for the machine-readable form.
