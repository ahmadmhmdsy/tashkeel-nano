---
license: other
license_name: research-only
license_link: LICENSE
language:
- ar
- en
pipeline_tag: text-generation
tags:
- arabic
- diacritization
- tashkeel
- nano
- transformer
- soup
model-index:
- name: tashkeel-nano-1p1m
  results:
  - task:
      type: text-generation
      name: Arabic Diacritization
    dataset:
      name: registered 3-gate mean (fadel_test + sadeed25 + wikinews2014)
      type: der_registered_gate_mean3
    metrics:
    - type: der
      value: 11.51
      name: DER (%)
---

# Tashkeel Nano 1.1M — Arabic Diacritization

**A 1.08M-parameter Arabic diacritization (tashkeel) model that reaches
**11.51% DER** on the registered 3-gate tracker — **4.1× smaller than
tashkeel-nano-4p5m and 5.9× better on the same gate.**

- **Model:** `ahmadsy/tashkeel-nano-1p1m` (internal id `e98_soupB`, artifact `soupB_uniform`)
- **Task:** predict the Arabic diacritic marks (fatha, damma, kasra, shadda,
  tanwin, sukun) for every Arabic letter in a piece of text.
- **Architecture:** bidirectional char-level Transformer encoder + 15-class
  diacritic head (see [Architecture](#architecture)).
- **Parameters:** 1,083,328 (1.08 M) · **Size:** 4.15 MiB (fp32 safetensors)
- **Published checkpoint:** a **uniform soup of 12 step probes** from a
  60,000-step run (steps 32,500 → 60,000) on the `v3full` corpus.

---

## Results

DER = diacritic error rate (lower is better), reported as a percentage. The
**registered 3-gate tracker** is the mean over **fadel_test, sadeed25,
wikinews2014**; wikinews2024 is excluded as contaminated (see
[Limitations](#limitations--honesty)).

All numbers below come from the registered scorer
(`der_registered`) on identical splits.

| Model | Params | fadel_test | sadeed25 | wikinews2014 | **3-gate mean** |
|---|---|---|---|---|---|
| **tashkeel-nano-1p1m (this)** | **1.08 M** | **5.20** | **11.71** | **17.64** | **11.51** |
| E-097 arm B @ 30k (this soup's source) | 1.08 M | 6.38 | 14.39 | 20.63 | 13.80 |
| tashkeel-nano-4p5m (corrected, NFC) | 4.49 M | 14.30 | 22.59 | 31.81 | 22.90 |
| deployed gold `e73_soup/c4b_uniform` | 1.08 M | 13.81 | 20.51 | 25.93 | 20.08 |

Each model is shown under its own **fair decode** (the arm with the lowest
gate mean for that model, per the pre-registered E-100 rule):

- **tashkeel-nano-1p1m** — `centered` sliding window 256 / stride 128 on bare text.
  Under the non-overlapping `single` decode it is 5.87 / 12.44 / 18.63 → **12.31**.
- **tashkeel-nano-4p5m** — `nospace_centered` (its `v3` bed has **zero** space
  tokens, so raw-text decoding feeds an untrained space embedding). Under raw
  `single` it is 19.70 / 30.30 / 36.31 → **28.77**.
- **arm B @ 30k / deployed gold** — `centered` on bare text.

### The soup is the lever, not the extra training

The 12 soup members individually span **12.88 – 14.17** on the gate mean, with
no trend — continuing training from 30k to 60k did **not** produce a monotone
gain. The **uniform average** over them is **11.51**, a **−16.55%** improvement
over the 30k anchor (13.80) and **−10.60%** over the best single member
(12.88). Averaging over a wide, same-basin checkpoint set is the mechanism
that lands the result.

### Correction: the earlier 4.5M comparison was wrong

If you have seen a figure of **−64.01%** for this line against
tashkeel-nano-4p5m, it is **retracted**. That comparison decoded the 4.5M model
with a renderer that predates the NFC mark-order fix, so its output was
penalised by a defect rather than by model quality. The corrected head-to-head,
with NFC on both sides, is:

| comparison | gate mean | delta |
|---|---|---|
| tashkeel-nano-1p1m (this) | **11.51** | — |
| tashkeel-nano-4p5m, fair decode | 22.90 | **−49.71%** |
| arm B @ 30k (this soup's source) | 13.80 | −16.55% |
| deployed gold | 20.08 | −42.67% |

### Unicode NFC — this model emits canonical text

Combining marks are emitted in **NFC order**: fatHa (ccc 30) sorts *before*
shadda (ccc 33), so a geminate renders `دَّ` and not `دّ َ`. The historical
renderer emitted shadda-first, which disagrees with every NFC reference on
every geminate — measured at **4.5% of all units and 42% of the registered
gate** on the 4.5M model. `tashkeel_nano/labels.py` in this repo carries the
fixed `marks_for_label`, and inference additionally normalises its output to
NFC.

---

## Architecture

Char-level bidirectional Transformer encoder with a per-position 15-class
diacritic head. Llama-style blocks (RMSNorm + RoPE + SwiGLU + grouped-query
attention) with the **causal mask off**.

| | |
|---|---|
| layers | 18 |
| hidden | 64 |
| FFN | 256 (SwiGLU) |
| heads | 8 (GQA, 2 KV heads, head_dim 8) |
| context | 256 |
| vocab | 171 (char-level, deterministic) |
| classes | 15 |
| parameters | 1,083,328 |

The 15 classes are `bare, fatha, damma, kasra, sukun, fathatan, dammatan,
kasratan, shadda, shadda_fatha, shadda_damma, shadda_kasra, shadda_fathatan,
shadda_dammatan, shadda_kasratan`.

Non-Arabic content (digits, latin, emoji, tatweel, Quranic annotation,
invisibles) is **passthrough**: it stays visible to the encoder as context but
is never a prediction target and is re-inserted byte-exact, so
`reconstruct(parse(x)) == x` always holds.

### Decoding

`centered` (default): overlapping windows of 256 chars at stride 128 are
scored, and each Arabic letter takes the mark from the window whose **centre**
is nearest to it — no majority vote, ties broken on the mark's codepoint order.
`single` is the non-overlapping hard-chunk fallback (cheaper, ~0.8 DER worse).

---

## Training

| | |
|---|---|
| corpus | `v3full` — the full v3 corpus **with** real space tokens (15.28% of positions) |
| init | `e69_c4_ctx256` final |
| steps | 60,000 |
| batch | 64 |
| LR | 2e-4 constant (AdamW, no schedule) |
| weight decay | 0.1 |
| precision | fp16 autocast |
| seed | 20261012 |
| published artifact | uniform mean of the 12 step probes at 32,500 … 60,000 |

**Why `v3full` matters:** the 4.5M model was trained on `v3`, a bed with **no
space tokens**, so its space embedding row is never trained and raw-text
decoding feeds it an untrained embedding. This model is trained on a bed with
real space tokens, so ordinary raw text is the correct input — no
whitespace-stripping workaround is needed.

---

## Usage

### From the HuggingFace hub (recommended)

```python
from huggingface_hub import snapshot_download
import sys

local = snapshot_download("ahmadsy/tashkeel-nano-1p1m")
sys.path.insert(0, local)

from tashkeel_nano import TashkeelNano

model = TashkeelNano.from_pretrained("ahmadsy/tashkeel-nano-1p1m")
print(model.diacritize("السلام عليكم ورحمة الله"))
# السَّلَامُ عَلَيْكُمْ وَرَحْمَةُ اللَّهِ
```

Only `torch`, `huggingface_hub` and `safetensors` are required; the tokenizer,
label schema and passthrough codec are stdlib-only and ship with the repo.

### CLI

```bash
python diacritize.py --text "مرحبا بالعالم"
python diacritize.py --file input.txt
echo "مرحبا بالعالم" | python diacritize.py
python diacritize.py --decode single --file input.txt   # cheaper, slightly worse
```

### Notes

- Input may be diacritized or bare; marks are stripped and the text is
  NFC-normalised before encoding, so both are accepted.
- Output is always NFC.
- CPU works; a GPU is used automatically when available.
- Long inputs are windowed internally — no manual chunking needed.

---

## Limitations & honesty

1. **The 4.5M comparison in this card is the corrected one.** Any older figure
   near −64% is retracted; see the correction note above. The 4.5M model was
   re-decoded four ways (E-100) and the table quotes its best fair arm.
2. **Each model is compared under its own fair decode**, not under one forced
   decode. That is the pre-registered E-100 rule, and it is the only fair way
   to compare two models trained on different token beds.
3. **wikinews2024 is excluded from the tracker** as contaminated, and
   sadeed25 is overlap-flagged (Sadeed_Tashkeela is in the training mix), so
   the cleanest single reading is **wikinews2014 (17.64%)**.
4. **A soup is not a checkpoint.** The published weights are an average of 12
   checkpoints; the individual members span 12.88 – 14.17. Reproducing 11.51
   requires the same 12 members and the same uniform recipe, in ascending step
   order.
5. **The package's fp32 decode differs slightly from the registered fp16
   measurement.** Running this exact repo end-to-end gives a gate mean of
   **11.50** versus the registered **11.51** (−0.12%), and **12.30** versus
   **12.31** (−0.11%) under `single`. Both are ~20× below the 2% significance
   bar used throughout this line.
6. **Not a Quranic diacritization model.** It predicts the 15 classes above;
   Quranic annotation marks are passthrough and are not predicted.

---

## Reproduction

Registered metrics were produced with the scorer at
`research/verdict_audit/tools/e95_gate_score.py` over identical splits, with
references byte-identical across all runs.

```
soup members    runs/diac/e98_soupB/gate_probe/weights_step{32500..60000}.pt   (12)
soup recipe     scratch/e99_soup.py  — uniform mean, ascending step order
soup artifact   runs/diac/e98_soup/soupB_uniform/model.pt
verdict         runs/diac/e98_centered/soupB_verdict.json
registered      runs/diac/e98_centered/soupB_uniform/<corpus>/score_registered.json
4.5M fair decode runs/diac/e100_published_nospace/e100_verdict.json
NFC fix         diacritizer/src/labels.py:77-84
```

---

## Ancestor models

- `ahmadsy/tashkeel-nano-4p5m` — the 4.49M model. Larger, and **worse** on the
  registered gate (22.90 vs 11.51) once its NFC-corrected fair decode is used.
  Kept published for reference.
- `e73_soup/c4b_uniform` — the previously deployed 1.08M gold (20.08), retained
  as the outgoing anchor.

## License

Research-only; see `LICENSE`.
