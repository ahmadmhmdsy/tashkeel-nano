# Evaluation

## The registered metric

The headline number in both model cards is the **registered 3-gate mean**:

    mean of der_registered over { fadel_test, sadeed25, wikinews2014 }

`der_registered` is a **word-level** error rate. A word counts as correct when its
diacritic string matches **any of the line's references**, where a line's
references are the TAB-separated alternatives written by the benchmark harness
(usually just one).

A `/` **inside** a token is *not* expanded into alternatives under
`der_registered`; that is the registered anchor's own behaviour, inherited from
`eval_der`, and it is what makes the number reproducible against the recorded
reports. The alternative-aware view is `der_anyalt`
(`tashkeel_nano.eval_metrics`), which expands `/`; on the gate corpora the two
disagree on roughly 0.3-0.6 pct of tokens, because a `/`-bearing reference token
can only be matched by reproducing the literal string.

> This is **not** Sadeed's letter-level "DER (CE)". Sadeed's word-level analogue
> is called "WER (CE)". The two are different metrics with similar names, and
> comparing one against the other produces a number that looks meaningful and is
> not.

The four values the scorer reports per file are DER, WER, `DER_nocase` (case
endings dropped on both sides) and `text_preservation` (the share of words whose
base letters are unchanged — a hard gate, ~1.0 by construction in this pipeline).

Aggregation is **word-weighted**: each line's rate is multiplied by that line's
word count before summing, so the result is a corpus-level rate and not a mean of
per-line fractions.

## Which corpora count

| corpus | role |
|---|---|
| `fadel_test` | registered gate. Downloaded, never trained on. |
| `sadeed25` | registered gate, **overlap-flagged** — Sadeed_Tashkeela is in the training mix, so part of this gate is close to the training distribution. Kept because it was pre-registered. |
| `wikinews2014` | registered gate. |
| `wikinews2024` | **contaminated** — its text is in the training pool. Reported for continuity, **excluded** from the mean. |
| `abdou_test` | **held-out witness** — never trained on by any model in this line. Reported separately, never folded into the mean. |

Keeping the contaminated and witness corpora visible but out of the mean is the
whole point: a single number that mixes them would be unfalsifiable.

## Decode arms and the fair-decode rule

Windows have to be chosen before marks can be read out, and the choice is worth
up to ~0.8 DER on the 1.1 M model and far more on the 4.5 M one.

| arm | description |
|---|---|
| `single` | non-overlapping hard chunks of `ctx` |
| `centered` | overlapping windows of `window` at `stride`; each position takes the mark from the nearest-centre window, no voting, ties on code-point order |
| `nospace` | `single` with whitespace stripped before encoding |
| `nospace_centered` | `centered` with whitespace stripped before encoding |

**The rule: every model is quoted under its own fair decode.** The 4.5 M model was
trained on a bed with no space tokens, so feeding it spaced text hands the encoder
an untrained embedding; stripping whitespace and using a centered window is the
fair setting for it. The 1.1 M model was trained with real space tokens, so plain
text is already correct for it and `centered` is its fair arm.

## Mark order

Combining marks must be emitted in **Unicode NFC order**: fatha has canonical
combining class 30 and shadda 33, so fatha sorts first. A shadda-first renderer
disagrees with every NFC reference on every geminate. `tashkeel_nano.inference`
takes `order="nfc"` or `order="shadda_first"` explicitly, and the presets record
which each published package actually does.

## Recorded results

Registered gate (percent, lower is better):

| model | decode | fadel_test | sadeed25 | wikinews2014 | mean |
|---|---|---|---|---|---|
| tashkeel-nano-1p1m | `centered`, NFC | 5.20 | 11.71 | 17.64 | **11.51** |
| tashkeel-nano-1p1m | `single`, NFC | 5.87 | 12.44 | 18.63 | 12.31 |
| tashkeel-nano-4p5m | `nospace_centered`, output normalised to NFC | 14.30 | 22.59 | 31.81 | **22.90** |
| tashkeel-nano-4p5m | `nospace_centered`, shadda-first as emitted | 27.34 | 38.20 | 45.14 | 36.89 |
| tashkeel-nano-4p5m | `single`, NFC | 19.70 | 30.30 | 36.31 | 28.77 |
| tashkeel-nano-4p5m | as recorded in its card (pre-NFC renderer) | 31.31 | 43.37 | 47.19 | 40.62 |

The 4.5 M model's published card reports 40.62 in its headline table. That
figure came from a renderer that predates the NFC fix and penalises the model
for a defect rather than for its quality. The card carries a correction section,
and the fair-decode figure is 22.90. Two of its rows above quote the 4.5 M model
with its output normalised to NFC, because the registered references are NFC;
the package itself emits shadda-first, and that literal rendering is the 36.89
row. Both readings are the same weights, decode arm, window and stride.

### Re-running this table

The 1.08 M row is reproducible here from the shipped weights. Measured
2026-10-14 with `scripts/score_gate.py --run models/nano1p1m --config training_config.yaml --decode centered --order nfc`:

| gate | DER (this run) | DER (table above) |
|---|---|---|
| fadel_test | 0.051967 | 0.0520 |
| sadeed25 | 0.117038 | 0.1171 |
| wikinews2014 | 0.176091 | 0.1764 |
| **registered mean** | **0.115032** | 0.115148 |

The reference files are byte-identical to the ones behind the table, but 64 of
2,501 fadel_test lines (50/1,613 sadeed25, 30/394 wikinews2014) differ from the
recorded predictions by a **single mark** - the same model choosing a different
member of a near-tie vowel pair. Same weights, same decode rule, same window and
stride; the residual is argmax tie-flip between the original batch decode and the
single-sequence decode used here, worth 0.011 percentage points on the mean.
Quote 11.5 %; the third decimal is not stable.

The 4.5 M row reproduces too. Measured 2026-10-14 on the CPU, `--decode
nospace_centered`, scoring the predictions twice - once as emitted and once
after `unicodedata.normalize("NFC", ...)`:

| gate | as emitted (shadda-first) | normalised to NFC | table (NFC) |
|---|---|---|---|
| fadel_test | 0.273410 | 0.142968 | 0.1430 |
| sadeed25 | 0.381979 | 0.225967 | 0.2259 |
| wikinews2014 | 0.451384 | 0.318058 | 0.3181 |
| **registered mean** | 0.368924 | **0.228997** | 0.228984 |

0.0013 percentage points below the recorded mean, from one mark on one sadeed25
line - the same tie-flip as above. The as-emitted column is what the published
package actually produces, and it is why the fair rows have to say which mark
order they were scored in.

Held-out witness, `abdou_test`:

| model | DER | DER_nocase | WER | text_preservation |
|---|---|---|---|---|
| tashkeel-nano-4p5m | 39.30 | 29.30 | 39.65 | 0.9846 |

Per-gate detail for the 4.5 M published checkpoint (step 19,500):

| gate | DER | WER | DER_nocase | text_preservation |
|---|---|---|---|---|
| fadel_test | 0.3131 | 0.3131 | 0.2209 | 1.0000 |
| sadeed25 | 0.4337 | 0.4337 | 0.3070 | 0.9999 |
| wikinews2014 | 0.4719 | 0.4719 | 0.3312 | 1.0000 |
| wikinews2024 | 0.5575 | 0.5575 | 0.4587 | 1.0000 |

## Running the evaluation

    # predict and score the registered gate in one command
    python scripts/score_gate.py --run models/nano1p1m \
        --config training_config.yaml --decode centered --order nfc \
        --out-json out/gate_1p1m.json

    # the 4.5 M model under its fair arm: --order nfc normalises the emitted
    # shadda-first text to the order the registered references use. Omitting it
    # keeps the package default and scores 36.89 against those same references.
    python scripts/score_gate.py --run models/nano4p5m \
        --config training_config.yaml --decode nospace_centered --order nfc

    # score prediction files that already exist
    python scripts/score_gate.py --pred-dir out/gate

    # a single file against a single reference
    python scripts/eval.py compare --pred pred.txt --ref ref.txt

    # the full multi-metric report
    python scripts/report.py --ref data/raw/wikinews/wikinews2014_multi_ref.diac \
        --pred arm_a=out/a.pred.txt --out-json out/metrics.json

`score_gate.py` writes `<out>/<source>.pred.txt` and `<out>/<source>.ref.txt`
beside each other, so a number can always be traced back to the exact files that
produced it. Prediction uses the GPU; scoring is CPU-only and imports no torch.

## Sadeed's own metric

If you compare against the SadeedDiac-25 leaderboard, note that its published
numbers are letter-level. `scripts/eval_multi_ref.py` implements the
`/-separated multi-reference comparison for the WikiNews surface; the caveat it
documents is real — the WikiNews inputs contain `/` alternatives, so models copy
`/` into their predictions. DER is unaffected (it compares marks only, and `/`
is not a mark), but WER and text preservation treat the whole `/-containing token
as the prediction.
