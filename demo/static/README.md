---
title: Tashkeel Compare
emoji: "✨"
colorFrom: green
colorTo: blue
sdk: static
pinned: false
license: apache-2.0
---

# Tashkeel Compare

Arabic diacritization (tashkeel), side by side, in the browser. Our **1.08 M
parameter** model runs next to published third-party models on the same text,
with parameter count and latency shown for every one.

## Languages

The interface ships in **English and Arabic**. The page starts in the visitor's
language when the browser asks for Arabic and otherwise in English, and the
switch in the masthead overrides that at any time; the choice is remembered.

The two editions are the same page computing the same numbers. Only the prose,
the sample names and the writing direction change - and in Arabic the display
and UI faces become Amiri and Noto Sans Arabic. Model names stay as their
repository identifiers in both languages, because that is what you would type
into a search box.

## Why this Space exists

A model card can claim a small model is competitive. A Space lets anyone check
it in about ten seconds, on their own text. The point is the size/quality trade:
**1.08 M parameters** against models tens to hundreds of times larger.

## How it runs

This is a **static** Space. There is no server process and nothing is uploaded:
both of our models ship as ONNX graphs and are executed by ONNX Runtime Web
inside your own browser tab. Your text never leaves the page.

The two third-party models cannot run that way, so their columns on the
**Compare** tab are **precomputed** outputs for the five built-in samples, and on
the **Score** tab for the one pre-filled reference. They are labelled
`precomputed` wherever they appear. Nothing pretends to be live that is not.

## What is here

| Model | Params | How | Notes |
|---|---|---|---|
| `ahmadsy/tashkeel-nano-1p1m` | 1.08 M | live in browser | ours. Centered sliding-window decode, emits Unicode NFC |
| `ahmadsy/tashkeel-nano-4p5m` | 4.49 M | live in browser | ours. Superseded by 1p1m; ships the pre-NFC renderer |
| `flokymind/mishkala` | 12.5 M | precomputed | Mamba + Transformer with a CRF decoder |
| `Abdou/arabic-tashkeel-flan-t5-small` | 77 M | precomputed | greedy decoding |

Every model here is a downloadable checkpoint. Nothing unpublished is compared
anywhere in this Space.

## The four tabs

1. **Compare** - one text, every selected model, with parameters, latency and a
   mark-order badge per model. Letters whose marks disagree with the first
   column are tinted, so the disagreement is the thing you see.
2. **Bulk** - paste or upload a large document. It is split on paragraph and
   sentence boundaries, every chunk is diacritized, and the result is
   reassembled with your original whitespace and line structure intact. A
   splitting report states exactly what happened, and the finished text can be
   downloaded or copied.
3. **Score** - paste a diacritized reference and get DER and WER per model. The
   reference is stripped to its bare letters, every model re-vocalises that bare
   text, and the outputs are scored against what you pasted.
4. **Board** - the published SadeedDiac-25 table, a cost-vs-quality scatter, and
   our own rows scored literally (mark order *not* normalised) so the artefact
   below is visible rather than merely described.

## Mark order matters, and this Space says so

SadeedDiac-25 gold writes the shadda *before* the vowel (`U+0651 U+064E`),
which is not NFC. A naive literal comparison against NFC output
(`U+064E U+0651`) marks every geminate letter wrong. Measured on the same
predictions, that artefact alone moves DER from 3.78 to 8.78.

Scoring here therefore canonicalises the mark order of each letter on **both**
sides before comparing, so a difference in the metric means a difference in
reading, not a difference in spelling convention. The toggle is on screen, so
you can turn it off and watch the artefact appear.

## Honesty notes

- Benchmark rows marked **quoted** come from the published SadeedDiac-25
  leaderboard and use that benchmark's own evaluator, which is
  diacritic/letter-level. Rows marked **measured here** were produced locally
  with that same evaluator and mark order normalised.
- The 8 B-13 B systems in the quoted table cannot run on free CPU, so their
  numbers are quoted rather than re-measured.
- The benchmark does not state whether its own leaderboard systems were
  order-normalised, so the ranking is approximate.
- `wikinews2024` is excluded from the registered tracker as contaminated.
- The **Score** tab's pre-filled reference is the opening excerpt of the
  SadeedDiac-25 reference, taken by position rather than picked to favour
  anyone. It is one passage; the **Board** tab carries the full 1200-row result.
- The in-browser models are a JavaScript port of the published PyTorch
  packages, checked to agree with them exactly on a fixed set of texts.
- ONNX Runtime Web is loaded from jsDelivr, so the page needs network access
  the first time it runs.

## Local run

```bash
python -m http.server 7863
# then open http://127.0.0.1:7863/
```

## License

Apache-2.0 for this Space's code. Each model keeps its own license.
