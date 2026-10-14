# Architecture

## The model

A **character-level bidirectional Transformer encoder** with a per-position
15-class classification head. Nothing about it is exotic; the whole design is
"small, dense, and fully parallel over the sequence".

| | 1p1m | 4p5m |
|---|---|---|
| layers | 18 | 12 |
| hidden | 64 | 160 |
| FFN | 256 (SwiGLU) | 640 (SwiGLU) |
| heads | 8 | 8 |
| KV heads | 2 (GQA) | 2 (GQA) |
| head dim | 8 | 20 |
| context | 256 | 128 |
| vocab | 171 | 171 |
| classes | 15 | 15 |
| parameters | 1,083,328 | 4,488,160 |

Each block is Llama-style: RMSNorm, rotary position embeddings (RoPE), SwiGLU
feed-forward, grouped-query attention. The one deviation that matters is that
**the causal mask is off** — this is an encoder, so every position sees the whole
window.

`src/tashkeel_nano/model.py` is the only implementation. The forward signature is

    DiacritizerModel.forward(input_ids, labels=None, attn_mask=None)
        -> {"logits": [B, T, 15], "loss"?: scalar, "acc"?: scalar}

and the shape is built by `build_from_config(cfg, vocab_size)`, which reads
`cfg["model"]`: `layers`, `hidden`, `n_heads`, `n_kv`, `ffn`, `ctx`,
`dropout`, `causal`, `kv_pattern`, `shared_layers`.

Both published packages ship a copy of this file, so a model downloaded on its own
is self-contained. Those copies are byte-identical to each other and to the
training copy used for the release.

## Tokenizer

`src/tashkeel_nano/tokenizer.py` — deterministic, stdlib-only, vocab size 171.

The alphabet is built from a fixed base-character range plus the specials
`<pad>`, `<bos>`, `<eos>`, `<unk>`. Encoding is **by code point**, not by
grapheme cluster: one token per character. There is no trained vocabulary file and
no `sentencepiece` model, which is why inference needs no extra dependency and
why the JavaScript port in `demo/static/js/engine.js` can be an exact match.

## Labels and mark order

`src/tashkeel_nano/labels.py` defines the 15 classes:

    bare, fatha, damma, kasra, sukun, fathatan, dammatan, kasratan,
    shadda, shadda_fatha, shadda_damma, shadda_kasra,
    shadda_fathatan, shadda_dammatan, shadda_kasratan

A label is rendered to marks by `marks_for_label`. **There are two variants in the
wild, and this repository contains both, on purpose:**

| where | behaviour |
|---|---|
| `src/tashkeel_nano/labels.py` and `models/nano1p1m/tashkeel_nano/labels.py` | emits **NFC order** — `unicodedata.normalize("NFC", SHADDA + vowel)`, so fatha (ccc 30) precedes shadda (ccc 33) |
| `models/nano4p5m/tashkeel_nano/labels.py` | emits **shadda-first** with no normalisation |

That is not a bug in the copy; it is a faithful record of what each published
package actually does. A geminate written shadda-first disagrees with every NFC
reference, which is why the 4.5 M model's originally published DER was penalised
by roughly 4.5 % of units.

`tashkeel_nano.inference` therefore makes the order an explicit parameter
(`order="nfc"` or `order="shadda_first"`) and records the right value for each
model in `PRESETS`. When you load a model through `TashkeelNano` you get the
published behaviour without having to know any of this.

## Passthrough

`src/tashkeel_nano/passthrough.py` splits text into **units**. A unit is either a
target (a bare Arabic letter that can carry a mark) or a run of non-target
characters (digits, Latin, emoji, tatweel, Quranic annotation marks,
invisibles). Non-target runs are kept verbatim, are visible to the encoder as
context, and are never predicted.

The invariant is

    reconstruct(parse(x)) == x     for every input x

`scripts/selftest.py` asserts it on a corpus of adversarial strings, including
emoji, bidi controls and mixed scripts.

## Decoding

Marks are predicted per position, but a long document does not fit in one
window, so the windows have to be chosen. Four arms exist, and they are not
equivalent:

| arm | what it does |
|---|---|
| `single` | non-overlapping hard chunks of `ctx`. Cheapest. |
| `centered` | overlapping windows of `window` at `stride`; each position takes the mark from the window whose **centre** is nearest to it. No majority vote; ties break on mark code-point order. |
| `nospace` | `single`, but whitespace is stripped before encoding (for beds with no space tokens). |
| `nospace_centered` | `centered`, but whitespace is stripped before encoding. |

Published defaults, from `tashkeel_nano.inference.PRESETS`:

    tashkeel-nano-1p1m  ->  decode="centered", order="nfc",         window=256, stride=128
    tashkeel-nano-4p5m  ->  decode="single",   order="shadda_first"

The 4.5 M model's *fair* arm is `nospace_centered`; see
[evaluation.md](evaluation.md) for why the preset keeps the historically recorded
arm and how to select the fair one.

## Inference API

`src/tashkeel_nano/inference.py` is the single entry point.

    from tashkeel_nano import TashkeelNano

    TashkeelNano.from_pretrained(repo_id, *, token, device, cache_dir,
                                 decode, order, nospace, window, stride, preset)
    TashkeelNano.from_run(run_dir, *, ckpt, config, device, vocab_size,
                          decode, order, nospace, window, stride, preset)
    TashkeelNano.diacritize(text) -> str
    TashkeelNano.diacritize_lines(lines) -> list[str]
    TashkeelNano.predict_marks(piece, skip_ws=False)

`from_pretrained` reads the model's `config.json` (which carries the decode
settings for 1p1m) and falls back to `PRESETS` keyed by the repository basename.
`from_run` reads a training run's `config.yaml` so the shape is rebuilt exactly.

## Import cost

`tashkeel_nano/__init__.py` imports only stdlib for `eval_der`, `eval_metrics`,
`labels`, `passthrough` and `tokenizer`. `model`, `inference` and `pipeline`
are loaded lazily through `__getattr__`, so scoring a prediction file does not
import torch at all.
