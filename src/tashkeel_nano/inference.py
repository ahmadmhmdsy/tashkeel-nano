"""Unified inference for the tashkeel-nano D-line models.

ONE entry point that reproduces every decode arm this project measured, so a
reported number can always be traced to the exact decode that produced it.

DECODE ARMS (a 2x2 factorial; E-092..E-100)
-------------------------------------------
                       spaces fed          spaces stripped
  hard chunk           single              nospace
  centered             centered            nospace_centered

  single      non-overlapping pieces of ctx chars (the historical default).
  centered    overlapping windows of 'window' chars at 'stride'; every Arabic
              base letter takes the mark from the window whose CENTER is
              nearest (no majority vote; ties break on mark codepoint order).
              This is the decode tashkeel-nano-1p1m was SELECTED on.
  nospace     as above, but whitespace is stripped from what the encoder sees
              and re-inserted at its original positions. The training corpus
              contains no space tokens, so the deployed decode feeds an
              untrained embedding row at every word boundary (E-094).
  nospace_centered  both.

MARK ORDER
----------
  nfc           emit the label's mark string and NFC-normalize the result.
                Canonical; agrees with every NFC reference. This is what
                tashkeel-nano-1p1m emits.
  shadda_first  emit U+0651 before its vowel and DO NOT normalize. This is the
                FROZEN render of the tashkeel-nano-4p5m published package,
                which was uploaded before the NFC fix. Keep it only to
                re-derive that model's published numbers; prefer 'nfc' for
                new work.

WHAT THIS IS NOT
----------------
The two published checkpoints do not share a mark-order convention. That is a
property of the artifacts, not a choice made here: nano4p5m's package ships a
pre-NFC labels.py, nano1p1m's ships the NFC one. PRESETS below records the
published setting per model so from_pretrained() is faithful by default.

Usage:
    from tashkeel_nano import TashkeelNano
    model = TashkeelNano.from_pretrained("ahmadsy/tashkeel-nano-1p1m")
    print(model.diacritize("السلام عليكم"))

    # local training run (runs/<phase>/final/{model.pt,config.yaml})
    model = TashkeelNano.from_run("runs/e98_soup/soupB_uniform")
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import torch

from .labels import marks_for_label
from .model import build_from_config
from .passthrough import is_arabic_base
from .tokenizer import VOCAB_SIZE, encode

#: Combining-mark ranges dropped before inference (the "bare" text).
MARK_RANGES = ((0x064B, 0x0652), (0x0670, 0x0670), (0x0653, 0x065F),
               (0x06D6, 0x06ED))

#: Whitespace the 'nospace' arms strip before encoding.
WS = set(" \t\n\r\x0b\x0c")

#: U+0651 ARABIC SHADDA.
SHADDA = "\u0651"

#: Decode settings of the PUBLISHED packages, keyed by HF repo basename.
#: These are read from the artifacts, not invented: nano1p1m was selected on the
#: centered decode and emits NFC; nano4p5m's published inference.py does a
#: non-overlapping chunked decode and its frozen labels.py emits shadda-first.
PRESETS = {
    "tashkeel-nano-1p1m": {"decode": "centered", "order": "nfc",
                           "window": 256, "stride": 128},
    "tashkeel-nano-4p5m": {"decode": "single", "order": "shadda_first"},
    # the same settings under the local package directory names, so
    # TashkeelNano.from_run("models/nano4p5m") is faithful by default instead of
    # silently falling back to the centered/nfc defaults.
    "nano1p1m": {"decode": "centered", "order": "nfc",
                 "window": 256, "stride": 128},
    "nano4p5m": {"decode": "single", "order": "shadda_first"},
}

DECODE_ARMS = ("single", "centered", "nospace", "nospace_centered")
ORDERS = ("nfc", "shadda_first")


def strip_marks(text: str) -> str:
    """Bare text: drop every combining mark, then NFC-normalize."""
    out = []
    for ch in text:
        o = ord(ch)
        if any(lo <= o <= hi for lo, hi in MARK_RANGES):
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def _is_mark(ch: str) -> bool:
    return 0x064B <= ord(ch) <= 0x0652


def reorder_shadda_first(text: str) -> str:
    """Move U+0651 to the front of every combining-mark run, without NFC.

    Reproduces the pre-NFC renderer: shadda-first was the raw Tashkeela order,
    and NFC (fatha ccc 30 before shadda ccc 33) is what every reference uses.
    """
    out, i, n = [], 0, len(text)
    while i < n:
        if _is_mark(text[i]):
            j = i
            while j < n and _is_mark(text[j]):
                j += 1
            run = text[i:j]
            if SHADDA in run:
                run = SHADDA + "".join(c for c in run if c != SHADDA)
            out.append(run)
            i = j
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def window_starts(n: int, window: int, stride: int):
    """Start offsets of overlapping windows covering [0, n)."""
    if n <= window:
        return [0]
    starts = list(range(0, n - window + 1, stride))
    last = n - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def _model_cfg_from_hf(cfg: dict) -> dict:
    """HF config.json -> the repo's diac_* config family."""
    return {"model": {
        "hidden": cfg["hidden_size"],
        "layers": cfg["num_hidden_layers"],
        "n_heads": cfg["num_attention_heads"],
        "n_kv": cfg["num_key_value_heads"],
        "ffn": cfg["intermediate_size"],
        "ctx": cfg["max_position_embeddings"],
        "dropout": cfg.get("dropout", 0.0),
        "causal": cfg.get("causal", False),
        "kv_pattern": cfg.get("kv_pattern", "new"),
        "shared_layers": cfg.get("shared_layers", 0),
    }}


def resolve_decode(repo_id, cfg=None, preset=None) -> dict:
    """Published preset for a model name, overridden by its config.json."""
    name = preset or str(repo_id).rstrip("/").split("/")[-1]
    out = dict(PRESETS.get(name, {}))
    d = (cfg or {}).get("decoding") or {}
    for src, dst in (("arm", "decode"), ("order", "order"),
                     ("window", "window"), ("stride", "stride")):
        if d.get(src) is not None:
            out[dst] = d[src]
    out.setdefault("decode", "centered")
    out.setdefault("order", "nfc")
    return out


class TashkeelNano:
    """Arabic diacritizer: bidirectional char encoder + 15-class head.

    One class covers both published checkpoints and every experimental arm.
    """

    def __init__(self, model, ctx: int = 256, device: str = "cpu", *,
                 decode: str = "centered", order: str = "nfc",
                 nospace: bool = False, window: int | None = None,
                 stride: int | None = None):
        if decode not in DECODE_ARMS:
            raise ValueError("decode must be one of %r, got %r" % (DECODE_ARMS, decode))
        if order not in ORDERS:
            raise ValueError("order must be one of %r, got %r" % (ORDERS, order))
        self.model = model
        self.ctx = int(ctx)
        self.device = torch.device(device)
        self.decode = decode
        self.order = order
        self.nospace = bool(nospace) or decode.startswith("nospace")
        self.window = int(window or self.ctx)
        self.stride = int(stride or max(1, self.window // 2))
        self.model.to(self.device)
        self.model.eval()

    # -- constructors -------------------------------------------------------
    @classmethod
    def from_pretrained(cls, repo_id: str, *, token: str | None = None,
                        device: str | None = None, cache_dir=None,
                        decode=None, order=None, nospace=None,
                        window=None, stride=None, preset=None) -> "TashkeelNano":
        """Download (or reuse a local snapshot of) a HF repo and load it."""
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        local = Path(snapshot_download(
            repo_id=repo_id, token=token, cache_dir=cache_dir,
            allow_patterns=["*.safetensors", "*.json", "*.py", "*.md"]))
        cfg = json.loads((local / "config.json").read_text(encoding="utf-8"))
        model = build_from_config(_model_cfg_from_hf(cfg), vocab_size=VOCAB_SIZE)
        model.load_state_dict(load_file(str(local / "model.safetensors")), strict=True)
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kw = resolve_decode(repo_id, cfg, preset)
        kw.update({k: v for k, v in dict(decode=decode, order=order,
                                         nospace=nospace, window=window,
                                         stride=stride).items() if v is not None})
        return cls(model, ctx=cfg["max_position_embeddings"], device=dev, **kw)

    @classmethod
    def from_run(cls, run_dir, *, ckpt: str = "model.pt", config: str = "config.yaml",
                 device: str | None = None, vocab_size: int | None = None,
                 decode=None, order=None, nospace=None,
                 window=None, stride=None, preset=None) -> "TashkeelNano":
        """Load a training run directory (or any dir with a ckpt + yaml config).

        If the directory has no config file, the config embedded in the
        checkpoint is used instead, so a soup artifact - scripts/soup.py writes
        its config into model.pt - loads on its own.
        """
        import yaml
        d = Path(run_dir)
        snap = torch.load(d / ckpt, map_location="cpu", weights_only=False)
        cfg_path = d / config
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        elif isinstance(snap, dict) and isinstance(snap.get("config"), dict):
            cfg = snap["config"]
        else:
            raise FileNotFoundError(
                "%s has no %s and %s carries no embedded config; pass config="
                % (d, config, ckpt))
        vs = vocab_size or snap.get("vocab_size") or VOCAB_SIZE
        model = build_from_config(cfg, vocab_size=vs)
        model.load_state_dict(snap["model"] if "model" in snap else snap, strict=True)
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kw = resolve_decode(preset or d.name, cfg, preset)
        kw.update({k: v for k, v in dict(decode=decode, order=order,
                                         nospace=nospace, window=window,
                                         stride=stride).items() if v is not None})
        return cls(model, ctx=cfg.get("model", cfg).get("ctx", 256), device=dev, **kw)

    # -- inference ----------------------------------------------------------
    @torch.no_grad()
    def predict_marks(self, piece: str, skip_ws: bool = False):
        """Per-position mark strings for one piece (aligned to piece indices).

        Whitespace positions get "" when skip_ws is set, so callers index the
        result exactly as they index the piece.
        """
        kept = "".join(c for c in piece if c not in WS) if skip_ws else piece
        if not kept:
            return [""] * len(piece)
        ids = encode(kept)
        if not ids:
            return [""] * len(piece)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        logits = self.model(x)["logits"]
        marks = [marks_for_label(a) for a in logits.argmax(-1)[0].tolist()]
        if len(marks) != len(kept):                      # defensive: never misalign
            raise RuntimeError("tokenizer produced %d ids for %d chars"
                               % (len(marks), len(kept)))
        if not skip_ws:
            return marks
        out, k = [], 0
        for ch in piece:
            if ch in WS:
                out.append("")
            else:
                out.append(marks[k])
                k += 1
        return out

    def _assemble(self, piece: str, marks) -> str:
        out = []
        for i, ch in enumerate(piece):
            out.append(ch)
            if is_arabic_base(ch):
                out.append(marks[i])
        return "".join(out)

    def _single(self, bare: str) -> str:
        parts = []
        for s in range(0, len(bare), self.ctx):
            piece = bare[s:s + self.ctx]
            parts.append(self._assemble(piece, self.predict_marks(piece, self.nospace)))
        return "".join(parts)

    def _centered(self, bare: str) -> str:
        n = len(bare)
        votes = [[] for _ in range(n)]
        for s in window_starts(n, self.window, self.stride):
            e = min(s + self.window, n)
            piece = bare[s:e]
            marks = self.predict_marks(piece, self.nospace)
            center = (s + e - 1) / 2.0
            for wi in range(len(piece)):
                votes[s + wi].append((marks[wi], center))
        out = []
        for pos, ch in enumerate(bare):
            out.append(ch)
            if not is_arabic_base(ch):
                continue
            cands = votes[pos]
            if not cands:
                continue
            out.append(min(cands, key=lambda cm: (abs(pos - cm[1]), cm[0]))[0])
        return "".join(out)

    def diacritize(self, text: str) -> str:
        """Diacritize one line of Arabic text (marks predicted per letter)."""
        bare = strip_marks(text)
        if not bare.strip():
            return text
        core = self._centered(bare) if self.decode.endswith("centered") else self._single(bare)
        if self.order == "shadda_first":
            return reorder_shadda_first(core)
        return unicodedata.normalize("NFC", core)

    def diacritize_lines(self, lines, batch_note: bool = False):
        """Generator: diacritize an iterable of lines."""
        for line in lines:
            yield self.diacritize(line)

    def __repr__(self):
        return ("TashkeelNano(params=%d, ctx=%d, decode=%r, order=%r, nospace=%s, device=%r)"
                % (sum(p.numel() for p in self.model.parameters()), self.ctx,
                   self.decode, self.order, self.nospace, str(self.device)))
