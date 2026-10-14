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
model-index:
- name: tashkeel-nano-4p5m
  results:
  - task:
      type: text-generation
      name: Arabic Diacritization
    dataset:
      name: WikiNews-2024 (multi-ref, in-domain)
      type: wikinews2024
    metrics:
    - type: der
      value: 1.77
      name: DER (%)
---

# Tashkeel Nano 4.5M — Arabic Diacritization

**A 4.5M-parameter Arabic diacritization (tashkeel) model that beats the 30M
deployed gold standard on external benchmarks — 6.8× smaller and 6.7× lighter
(17.2 MB).**

- **Model:** `ahmadsy/tashkeel-nano-4p5m` (internal id `e63s_s4_12l160`)
- **Task:** predict the Arabic diacritic marks (fatha, damma, kasra, shadda,
  tanwin, sukun) for every Arabic letter in a piece of text.
- **Architecture:** bidirectional char-level Transformer encoder + 15-class
  diacritic head (see [Architecture](#architecture)).
- **Parameters:** 4,488,160 (4.49 M) · **Size:** 17.2 MB (fp32 safetensors)
- **Published checkpoint:** step 19,500 (best `val_loss` 0.1758) of a
  20,000-step run on the full v3 corpus.

---

## Results

### External gates (unseen data) — vs the 30M gold and the 17.2M champion

DER = diacritic error rate (lower is better). The 3-gate external tracker is
the mean over **fadel_test, sadeed25, wikinews2014** (wikinews2024 is excluded
as contaminated — see [Limitations](#limitations--honesty)).

| Model | Params | fadel_test | sadeed25 † | wikinews2014 | 3-gate mean |
|---|---|---|---|---|---|
| **tashkeel-nano-4p5m (this)** | **4.49 M** | **31.31** | **43.37** | **47.19** | **40.62** |
| champion M1 16L×272 (internal) | 17.2 M | 33.08 | 45.44 | 49.07 | 42.53 |
| stage2b2500 (deployed gold) | 30.1 M | 33.23 | 45.68 | 48.99 | 42.63 |

† sadeed25 is overlap-flagged (Sadeed_Tashkeela is in the training mix) — see
[Limitations](#limitations--honesty).

**The 4.5 M model beats both the 17.2 M internal champion and the 30.1 M
deployed gold on every external gate**, while being 3.8–6.8× smaller. The
smaller model generalizes better: it overfits the training distribution less
than the 30 M gold.

### Held-out witness — abdou_test (never trained on)

The cleanest generalization evidence: the abdou_tashkeel **test** split
(15,091 sentences / 41,378 lines / 1.54 M words) was never trained on by any
model in this line.

| Model | Params | DER | DER_nocase | WER | text_preservation |
|---|---|---|---|---|---|
| **tashkeel-nano-4p5m (this)** | **4.49 M** | **39.30** | **29.30** | **39.65** | 0.9846 |
| stage2b2500 (deployed gold) | 30.1 M | 41.60 | 30.40 | — | — |

The 4.5 M model beats the 30.1 M gold on the fully-unseen held-out set by
**−2.3 pp DER** (and −1.1 pp DER_nocase). (The gold's text_preservation on
this split is 0.9846 as well — 4 gold-reference lines in the split itself
differ, not model-caused.)

Per-gate detail for the published checkpoint (step 19,500), all four registered
gates (multi-metric rows in `runs/diac/e63s_s4_12l160/gate_eval.csv`):

| gate | DER | WER | DER_nocase | text_preservation |
|---|---|---|---|---|
| fadel_test | 0.3131 | 0.3131 | 0.2209 | 1.0000 |
| sadeed25 † | 0.4337 | 0.4337 | 0.3070 | 0.9999 |
| wikinews2014 | 0.4719 | 0.4719 | 0.3312 | 1.0000 |
| wikinews2024 ‡ | 0.5575 | 0.5575 | 0.4587 | 1.0000 |

† overlap-flagged · ‡ in-domain (contaminated — this data is in the training set)

The best-gate checkpoint (step 16,500, `best_gate_weights.pt`) reaches a
3-gate external mean of **39.96**; the published checkpoint is the best
`val_loss` selection (step 19,500), which is the stronger in-domain model.

### WikiNews 2024 multi-reference (in-domain — flagged)

Multi-ref DER on the WikiNews-2024 surface (356 sentences, 10,616 words). This
surface is **in the training set** for this model (contamination stamp), so it
is reported for completeness and is **not** external evidence.

| Model | Params | DER | DER_nocase | WER |
|---|---|---|---|---|
| mishkala (published) | 12.5 M | 1.52 % | 1.34 % | 1.52 % |
| **tashkeel-nano-4p5m (this)** | **4.49 M** | **1.77 %** | **1.45 %** | **1.77 %** |
| stage2b2500 (gold) | 30.1 M | 1.80 % | 1.49 % | 1.80 % |
| e63s_s3_8l272 (internal) | 8.64 M | 1.83 % | 1.44 % | 1.83 % |
| zmahood (published) | ~4.5 M | 1.85 % | 1.63 % | 1.85 % |
| e63s_s2_10l272 (internal) | 10.78 M | 1.90 % | 1.47 % | 1.90 % |
| e63s_s5_10l128 (internal) | 2.40 M | 1.90 % | 1.58 % | 1.90 % |
| e63s_s1_12l272 (internal) | 12.93 M | 1.91 % | 1.49 % | 1.91 % |
| e63s_s6_8l96 (internal) | 1.09 M | 1.96 % | 1.65 % | 1.96 % |

On this surface the 4.5 M model is the best of all compared models except
mishkala (12.5 M, 1.52 %) — and it is 2.8× smaller than mishkala. Note the
published models (mishkala, zmahood) were **not** trained on this surface, so
the comparison is not perfectly fair in our favour either way; the honest
external evidence is the gate table above.

### E-63s smallest-model sweep (context)

This model is arm **S4** of the E-63s sweep — six architectures shrinking
depth/width, all with the same recipe (vanilla CE, warm `stage1lm` init, full
v3 data, 10k-step wall, 4-gate probes every 500 steps). "Tracker mean" is the
3-gate external mean (fadel+sadeed+wn14).

| arm | params | best step | fadel | sadeed † | wn24 ‡ | wn14 | tracker mean |
|---|---|---|---|---|---|---|---|
| S1 12L×272 | 12.93 M | 4,500 | 34.70 | 46.72 | 57.72 | 49.39 | 43.60 |
| S2 10L×272 | 10.78 M | 6,500 | 34.52 | 45.69 | 56.41 | 48.51 | 42.91 |
| S3 8L×272 | 8.64 M | 5,500 | 31.20 | 43.77 | 55.65 | 47.34 | 40.77 |
| **S4 12L×160 (this)** | **4.50 M** | 10,000 | **31.76** | **44.12** | 56.57 | **48.03** | **41.30** |
| S5 10L×128 | 2.40 M | 9,500 | 33.97 | 47.16 | 59.11 | 51.35 | 44.16 |
| S6 8L×96 | 1.11 M | 10,000 | 37.03 | 49.32 | 60.46 | 52.99 | 46.45 |

S4 is the smallest arm that matches the champion class; below 4.5 M (S5, S6)
the embedding capacity becomes the cost driver (−3 to −6 DER). S4 was still
improving at the 10k wall, so it was auto-resumed to 20k steps; the published
checkpoint is the best `val_loss` of that run.

---

## Architecture

Bidirectional char-level Transformer encoder with a 15-class diacritic head
(Llama-style blocks with the causal mask **off**):

| component | value |
|---|---|
| char vocabulary | 171 (deterministic: Arabic block, digits, punctuation, ASCII, 4 specials) |
| embedding | 171 × 160 |
| layers | 12 |
| hidden | 160 |
| attention | 8 heads / 2 KV heads (GQA), head_dim 20, RoPE, bidirectional |
| FFN | SwiGLU, 640 |
| norm | RMSNorm (pre-norm), eps 1e-6 |
| head | Linear(160 → 15), no bias |
| max context | 128 chars |
| total params | 4,488,160 |

The 15 classes are the diacritic labels: bare, fatha, damma, kasra, sukun,
fathatan, dammatan, kasratan, shadda, shadda+fatha, shadda+damma, shadda+kasra,
shadda+fathatan, shadda+dammatan, shadda+kasratan. The head is applied only at
Arabic-base positions; every other character (digits, Latin, punctuation,
tatweel, Quranic marks, invisibles) is a passthrough context token and is
reinserted byte-exact, so non-Arabic text survives verbatim.

---

## Training data (v3 corpus)

The model was trained on the **v3** token pack = the v2b corpus +
**Sadeed_Tashkeela** (classical books), packed into ctx-128 char windows.

| source | role | windows |
|---|---|---|
| fadel train (AliOsm/arabic-text-diacritization) | train | 50,000 |
| abdou_tashkeel train | train | 1,443,972 |
| qcri wiki (QCRI diacritized Wikipedia) | train | 32,818 |
| wikinews2024 (QCRI WikiNews-2024) ‡ | train | 356 |
| sadeed_tashkeela (Misraj/Sadeed_Tashkeela) | train | 990,544 |
| abdou_tashkeel valid | val | 29,633 |
| fadel val | val | 2,500 |
| sadeed_tashkeela val | val | 52,134 |
| **total** | | **2,433,430 train / 160,214 val** |

Pack stats: 12,702 windows quarantined (invalid mark combinations), 7,581
duplicates removed. ‡ wikinews2024 is in the training set — this is the
contamination that makes the wn2024 gate in-domain.

### Sources & citations

- **fadel** — AliOsm/arabic-text-diacritization (GitHub), Fadel train/val/test split.
- **wikinews / qcri wiki** — QCRI advancing-arabic-diacritization (GitHub):
  WikiNews-2024/2014 multi-ref benchmarks, Wikipedia diacritized corpus.
- **sadeed_tashkeela** — HuggingFace `Misraj/Sadeed_Tashkeela` (Tashkeela-derived).
- **sadeed_25** (gate only) — HuggingFace `Misraj/SadeedDiac-25` benchmark.
- **abdou_tashkeel** — HuggingFace `abdou_tashkeel` (compilation; see license note).

### License & use

**Research use only.** The training mix includes web-scraped franken-datasets:
abdou_tashkeel's card claims MIT for the compilation but ~45 % of its content
is Tashkeela (commonly GPL-2 mirrored) plus shamela pages and
GPT-4o-mini-diacritized Wikipedia; fadel derives from Tashkeela;
sadeed_tashkeela is Tashkeela-derived (GPL-2 noted); qcri is paper-bundled
shared data (citation expected). A commercial superset would need a real
license audit. Not for commercial use without one.

---

## Strategy — how this result was achieved

1. **Stage-1 LM pretraining** (`stage1lm`, the ancestor): a 28L×272 char-LM
   (30.1 M) trained on raw Arabic (abdou strip-marks, sadeed input, qcri wiki,
   fadel train) for 12k steps, `val_loss` 2.4075 → 1.2806. This is the
   warm-start source.
2. **Warm transfer, not from-scratch:** the nano model initializes its
   embeddings + encoder stack from `stage1lm` (shape-filtered partial transfer
   for hidden < 272) and trains the 15-class head fresh. Warm init, not width,
   is the dominant factor at fixed budget.
3. **Smallest-model sweep (E-63s):** six arms shrinking depth/width on the full
   v3 corpus with an identical recipe, so the size effect is isolated.
4. **External gate probing every 500 steps** (DER, WER, DER_nocase,
   text_preservation) with a `best_gate_weights.pt` tracker. This matters: the
   in-domain `val_loss` improved monotonically to the wall everywhere and
   would have mis-selected every optimum — the external probes picked the real
   peaks.
5. **The nano model generalizes better than the gold:** at 4.5 M params it
   overfits the training distribution less than the 30 M gold, so it wins on
   the fully-unseen gates (fadel_test, wikinews2014) while staying competitive
   on the overlap-flagged ones.
6. **Extended schedule:** S4 was still improving at the 10k wall, so it was
   auto-resumed to 20k steps; the published checkpoint is the best `val_loss`
   (step 19,500).

### Training config

| field | value |
|---|---|
| loss | vanilla cross-entropy (label_smoothing 0.0) |
| optimizer | AdamW, lr 2e-4, weight_decay 0.1 |
| batch | 32 (accum 1), ctx 128 |
| steps | 20,000 (10k + 10k auto-resume) |
| precision | fp16 autocast + GradScaler, fp32 master weights |
| seed | 777 |
| selection | best val_loss → step 19,500 |

---

## Ancestor models

- **Direct ancestor — `stage1lm`** (`runs/diac/stage1lm/final/model.pt`):
  28L×272, 30.1 M, char-LM pretrain on raw Arabic, ctx 512, 12k steps,
  `val_loss` 2.4075 → 1.2806. This model warm-starts from it (embeddings +
  encoder stack; the 15-class head is fresh). It is the most meaningful
  ancestor because it is the actual initialization.
- **D-line lineage (context, not direct ancestors):** `stage1lm` →
  `stage2b2500` (deployed gold, 30.1 M) → E-63s sweep → `e63s_s4_12l160`.
  The gold is the reference this model beats, not an ancestor — the nano arm
  warm-starts from `stage1lm`, not from the gold.
- **Ablation family:** the E-63s arms S1–S6 (above) are the size ablation this
  model sits in.

---

## Usage

### From the HuggingFace hub (recommended)

```python
import sys
from huggingface_hub import snapshot_download

# download the repo and put its bundled `tashkeel_nano` package on the path
sys.path.insert(0, snapshot_download("ahmadsy/tashkeel-nano-4p5m"))

from tashkeel_nano import TashkeelNano
model = TashkeelNano.from_pretrained("ahmadsy/tashkeel-nano-4p5m")
print(model.diacritize("السلام عليكم ورحمة الله"))
# السَّلَامُ عَلَيْكُمْ وَرحْمَةُ اللَّهِ
```

The repo bundles the `tashkeel_nano` Python package (model, tokenizer, labels,
passthrough codec, inference). Adding the snapshot directory to `sys.path` makes
it importable; `from_pretrained` then loads the weights from the same snapshot.
If you already have the repo cloned locally, point `sys.path` at the repo root
instead of calling `snapshot_download`.

### CLI

```bash
python diacritize.py --text "السلام عليكم ورحمة الله"
python diacritize.py --file input.txt
echo "مرحبا بالعالم" | python diacritize.py
```

### Notes

- Runs on CPU (4.5 M params) or GPU; `device="cuda"` to force GPU.
- Input is diacritized or bare Arabic; marks are stripped internally and
  re-predicted. Non-Arabic text passes through byte-exact.
- The model predicts one 15-class label per Arabic letter; the passthrough
  codec reinserts the combining marks.

---

## Limitations & honesty

- **wikinews2024 is in the training set** (contamination stamp). The wn2024
  gate and the WikiNews-2024 multi-ref table are in-domain for this model and
  are **not** external evidence. The external tracker excludes wn2024.
- **sadeed25 is overlap-flagged:** Sadeed_Tashkeela is in the training mix, so
  the sadeed number is honest w.r.t. itself but not fully external.
- **mishkala (12.5 M) still leads on the WikiNews-2024 multi-ref surface**
  (1.52 % vs our 1.77 %) — but that surface is in-domain for us and
  out-of-domain for mishkala, so it is not a clean comparison. On the honest
  external gates (fadel_test, wikinews2024's sibling wikinews2014) this model
  beats the gold.
- **License:** research-only (see above). Not for commercial use without a
  license audit.
- The published checkpoint is the best `val_loss` selection; the
  best-external-gate checkpoint (step 16,500, 3-gate mean 39.96) is a
  different trade-off and is not the published artifact.

---

## Reproduction

Internal provenance: phase `e63s_s4_12l160`, config
`configs/e63s_s4_12l160.yaml`, run log `runs/e63s_s4.log`, gate metrics
`runs/diac/e63s_s4_12l160/gate_eval.csv`, sweep verdict
`research/e63_overfit_report_2026-09-23.md` (E-63s appendix), comparison
`research/MODEL_EVALUATION_REPORT.md`. Training used one Quadro RTX 3000
(6,144 MiB, fp16-only Turing); a bigger GPU would be faster, not different.
