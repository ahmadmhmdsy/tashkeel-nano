"""TashkeelNano - inference for the tashkeel-nano D-line diacritizer.

Self-contained: model class, char tokenizer, 15-class label schema and the
passthrough codec are the exact modules that produced the published metrics.
Only torch + the Python stdlib are required.

DECODING: the published checkpoint is a uniform soup of 12 step probes and was
SELECTED on the CENTERED sliding-window decode, so that is the default here.
Overlapping windows of ctx chars at stride ctx // 2 are scored and each Arabic
base letter takes the mark from the window whose CENTER is nearest to it (no
majority vote; ties break on the mark's codepoint order). decode='single'
falls back to the non-overlapping hard-chunk decode.

Output is Unicode NFC. Combining marks are emitted in NFC order (fatHa ccc 30
before shadda ccc 33); the historical shadda-first render disagreed with every
NFC reference on every geminate - 4.5 pct of all units and 42 pct of the
registered gate on the 4.5M model.

Usage:
    from tashkeel_nano import TashkeelNano
    model = TashkeelNano.from_pretrained("ahmadsy/tashkeel-nano-1p1m")
    print(model.diacritize("\u0627\u0644\u0633\u0644\u0627\u0645 \u0639\u0644\u064a\u0643\u0645"))
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import torch

from .model import build_from_config
from .tokenizer import VOCAB_SIZE, encode
from .labels import marks_for_label
from .passthrough import is_arabic_base

# Combining-mark ranges stripped before inference (NFC-normalized bare text).
MARK_RANGES = ((0x064B, 0x0652), (0x0670, 0x0670), (0x0653, 0x065F),
                (0x06D6, 0x06ED))


def strip_marks(text: str) -> str:
    """Bare text: drop every combining mark; NFC-normalize."""
    out = []
    for ch in text:
        o = ord(ch)
        if any(lo <= o <= hi for lo, hi in MARK_RANGES):
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def _window_starts(n: int, window: int, stride: int):
    """Start offsets for overlapping windows covering [0, n)."""
    if n <= window:
        return [0]
    starts = list(range(0, n - window + 1, stride))
    last = n - window
    if starts[-1] != last:
        starts.append(last)
    return starts


class TashkeelNano:
    """1.1M-param Arabic diacritizer (bidirectional char encoder + 15-class head)."""

    def __init__(self, model, ctx: int = 256, device: str = "cpu"):
        self.model = model
        self.ctx = ctx
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    # -- loading ------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, repo_id: str, *, token: str | None = None,
                        device: str | None = None, cache_dir=None) -> "TashkeelNano":
        """Download (or reuse a local snapshot of) the HF repo and load it."""
        from huggingface_hub import snapshot_download
        local = Path(snapshot_download(repo_id=repo_id, token=token,
                                       cache_dir=cache_dir,
                                       allow_patterns=["*.safetensors", "*.json",
                                                      "*.py", "*.md"]))
        cfg = json.loads((local / "config.json").read_text(encoding="utf-8"))
        m = cfg
        model_cfg = {
            "model": {
                "hidden": m["hidden_size"], "layers": m["num_hidden_layers"],
                "n_heads": m["num_attention_heads"], "n_kv": m["num_key_value_heads"],
                "ffn": m["intermediate_size"], "ctx": m["max_position_embeddings"],
                "dropout": m.get("dropout", 0.0), "causal": m.get("causal", False),
                "kv_pattern": m.get("kv_pattern", "new"),
                "shared_layers": m.get("shared_layers", 0),
            }
        }
        model = build_from_config(model_cfg, vocab_size=VOCAB_SIZE)
        from safetensors.torch import load_file
        sd = load_file(str(local / "model.safetensors"))
        model.load_state_dict(sd, strict=True)
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        return cls(model, ctx=m["max_position_embeddings"], device=dev)

    # -- inference ----------------------------------------------------------
    @torch.no_grad()
    def diacritize(self, text: str, decode: str = "centered") -> str:
        """Diacritize Arabic text (marks are predicted per Arabic letter).

        decode='centered' (default, the published decode) or 'single'.
        """
        bare = strip_marks(text)
        if not bare.strip():
            return text
        if decode == "single":
            return self._single(bare)
        n = len(bare)
        stride = max(1, self.ctx // 2)
        votes = [[] for _ in range(n)]
        for s in _window_starts(n, self.ctx, stride):
            e = min(s + self.ctx, n)
            piece = bare[s:e]
            ids = torch.tensor([encode(piece)], dtype=torch.long, device=self.device)
            logits = self.model(ids)["logits"]
            center = (s + e - 1) / 2.0
            pred = logits.argmax(-1)[0].tolist()
            for wi in range(len(piece)):
                votes[s + wi].append((marks_for_label(pred[wi]), center))
        out = []
        for pos, ch in enumerate(bare):
            out.append(ch)
            if not is_arabic_base(ch):
                continue
            cands = votes[pos]
            if not cands:
                out.append("")
                continue
            out.append(min(cands, key=lambda cm: (abs(pos - cm[1]), cm[0]))[0])
        return unicodedata.normalize("NFC", "".join(out))

    def _single(self, bare: str) -> str:
        out = []
        for s in range(0, len(bare), self.ctx):
            piece = bare[s:s + self.ctx]
            ids = torch.tensor([encode(piece)], dtype=torch.long, device=self.device)
            logits = self.model(ids)["logits"]
            pred = logits.argmax(-1)[0].tolist()
            for wi, ch in enumerate(piece):
                out.append(ch)
                if is_arabic_base(ch):
                    out.append(marks_for_label(pred[wi]))
        return unicodedata.normalize("NFC", "".join(out))
