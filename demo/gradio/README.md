# Gradio demo — Tashkeel Compare

A side-by-side Arabic diacritization app: our models next to published
third-party models, on the same text, with parameter count and latency shown for
every column.

    python -m pip install -r demo/gradio/requirements.txt
    python demo/gradio/app.py            # http://127.0.0.1:7862

Environment overrides: `GRADIO_SERVER_NAME` (default `127.0.0.1`) and
`GRADIO_SERVER_PORT` (default `7862`).

## Tabs

| tab | what it does |
|---|---|
| **Compare** | run every selected model on the same text and show the results side by side |
| **Bulk** | split a large paste or upload, diacritize all of it, download the results |
| **Score** | score every model against a reference you supply |
| **Leaderboard** | the published SadeedDiac-25 table plus a cost-vs-quality scatter |

## Roster rules

Roster membership is **result-based**. Every model in `registry.py` was probed
locally before being admitted, and models that failed are not listed anywhere in
the UI — a model that cannot run must not appear as a broken button. The recorded
verdicts, including the excluded models and why, are in the `registry.py`
module docstring.

Two further honesty rules are enforced in the UI:

- quoted benchmark numbers are labelled **quoted**, never "measured here";
- scoring is order-insensitive by default, and the screen says so.

## Models and loading

Our two models are fetched from the Hub (`ahmadsy/tashkeel-nano-1p1m`,
`ahmadsy/tashkeel-nano-4p5m`). Third-party models run code published in their own
Hugging Face repositories, so their weights and code are outside this project's
control; the app says as much on screen.

Loading is lazy with LRU eviction, so the process does not hold every model in
memory at once.

## Requirements

`requirements.txt` pins `gradio==6.26.0` and pulls `torch`, `transformers`,
`pytorch-crf`, `sentencepiece` and `pandas` for the third-party adapters. This is a
much heavier environment than the inference and evaluation path needs — if you
only want our models, use `scripts/infer.py` or the static demo in
`demo/static/` instead.

## A note on validation

Gradio surfaces most component-validation errors at **interaction time**, not at
boot. A clean start-up therefore proves little: exercise each tab once after any
change.
