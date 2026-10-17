"""R-001: greedy-decode gate sources with base + LoRA adapter, then score with score_gate.

Inputs are split at spaces into chunks of at most PLAIN_CHUNK characters (the
size the training chunks have once marks are stripped), decoded, and re-joined
with single spaces, so <source>.pred.txt lines align 1:1 with <source>.ref.txt
and score_gate.py --pred-dir scores them unchanged.

    python r001/predict.py --model smollm2-135m --adapter runs/r001/pilot_a1_small/adapter \
        --out runs/r001/pilot_a1_small/preds --sources fadel_test --limit 100
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import cap_vram, MODELS, PLAIN_CHUNK, REPO, prompt

sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "r001"))
import score_gate  # noqa: E402
from data import chunks  # noqa: E402

REGISTERED = ("fadel_test", "sadeed25", "wikinews2014")


@torch.no_grad()
def decode(model, tok, texts, bs, max_new_factor=3.0):
    tok.padding_side = "left"
    out = []
    for i in range(0, len(texts), bs):
        group = texts[i:i + bs]
        enc = tok([prompt(t) for t in group], return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        cap = int(max(len(tok(t, add_special_tokens=False)["input_ids"]) for t in group) * max_new_factor) + 16
        with torch.autocast("cuda", dtype=torch.float16):
            gen = model.generate(**enc, max_new_tokens=cap, do_sample=False,
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        for row in gen[:, enc["input_ids"].shape[1]:]:
            text = tok.decode(row, skip_special_tokens=True)
            out.append(unicodedata.normalize("NFC", text.split("\n")[0].strip()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", default=",".join(REGISTERED))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()
    cap_vram()

    tok = AutoTokenizer.from_pretrained(MODELS[a.model])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODELS[a.model], dtype=torch.float16).cuda()
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    model.eval()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for src in a.sources.split(","):
        pairs = score_gate.load_pairs(src, a.limit)
        pieces, owner = [], []
        for k, (bare, _) in enumerate(pairs):
            for c in chunks(bare, PLAIN_CHUNK):
                pieces.append(c)
                owner.append(k)
        t0 = time.time()
        order = sorted(range(len(pieces)), key=lambda i: len(pieces[i]))
        dec = decode(model, tok, [pieces[i] for i in order], a.bs)
        got = [None] * len(pieces)
        for i, d in zip(order, dec):
            got[i] = d
        lines = [[] for _ in pairs]
        for k, d in zip(owner, got):
            lines[k].append(d)
        (out / f"{src}.pred.txt").write_text("\n".join(" ".join(x) for x in lines) + "\n", encoding="utf-8")
        (out / f"{src}.ref.txt").write_text("\n".join("\t".join(r) for _, r in pairs) + "\n", encoding="utf-8")
        m = score_gate.score_file(out / f"{src}.pred.txt", out / f"{src}.ref.txt")
        m["lines"], m["seconds"] = len(pairs), round(time.time() - t0, 1)
        summary[src] = m
        print(src, {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}, flush=True)
    reg = [summary[s]["DER"] for s in REGISTERED if s in summary]
    if len(reg) == len(REGISTERED):
        summary["registered_mean_DER_pct"] = round(100 * sum(reg) / len(reg), 2)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, dict)}))


if __name__ == "__main__":
    main()
