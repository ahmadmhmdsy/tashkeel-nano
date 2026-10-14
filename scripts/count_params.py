"""Count parameters for a model shape using the actual model code.

    python scripts/count_params.py --config configs/nano1p1m.yaml
    python scripts/count_params.py --layers 12 --hidden 160 --ffn 640 --ctx 128

With no arguments it prints the recorded e63s / candidate grid, which is what CI
exercises.
"""
import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tashkeel_nano.model import build_from_config
from tashkeel_nano import tokenizer as TK

VOCAB_SIZE = TK.VOCAB_SIZE

# Existing e63s models
EXISTING = [
    ("e63s_s1_12l272", 12, 272, 1088),
    ("e63s_s2_10l272", 10, 272, 1088),
    ("e63s_s3_8l272", 8, 272, 1088),
    ("e63s_s4_12l160", 12, 160, 640),
    ("e63s_s5_10l128", 10, 128, 512),
    ("e63s_s6_8l96", 8, 96, 384),
]

# Candidate ~1M param configurations (hidden must be divisible by 8)
CANDIDATES = [
    ("cand_10l88h352", 10, 88, 352),
    ("cand_12l80h320", 12, 80, 320),
    ("cand_14l72h288", 14, 72, 288),
    ("cand_16l64h256", 16, 64, 256),
    ("cand_18l56h224", 18, 56, 224),
    ("cand_20l48h192", 20, 48, 192),
    ("cand_22l40h160", 22, 40, 160),
    ("cand_24l32h128", 24, 32, 128),
    ("cand_8l100h400", 8, 100, 400),
    ("cand_6l108h432", 6, 108, 432),
    ("cand_5l120h480", 5, 120, 480),
    ("cand_4l140h560", 4, 140, 560),
    ("cand_6l112h448", 6, 112, 448),
    ("cand_4l128h512", 4, 128, 512),
]


def shape_cfg(layers, hidden, ffn, n_heads=8, n_kv=2, ctx=128):
    return {
        "layers": layers,
        "hidden": hidden,
        "ffn": ffn,
        "n_heads": n_heads,
        "n_kv": n_kv,
        "ctx": ctx,
    }


def count_params(cfg, vocab_size=VOCAB_SIZE):
    """Build a model and count its parameters.

    The dict is handed straight to build_from_config, so every model: key is
    honoured: layers, hidden, ffn, n_heads, n_kv, ctx, dropout, casual, causal,
    kv_pattern and shared_layers.
    """
    m = cfg.get("model", cfg)
    model = build_from_config(m, vocab_size)
    return sum(p.numel() for p in model.parameters())


def describe(name, cfg, vocab_size=VOCAB_SIZE):
    params = count_params(cfg, vocab_size)
    m = cfg.get("model", cfg)
    print("%s: %sL, %sh, %sffn, ctx %s -> %s params (%sM)"
          % (name, m.get("layers"), m.get("hidden"), m.get("ffn", "?"),
             m.get("ctx", "?"), format(params, ","), round(params / 1e6, 2)))
    return params


def grid():
    print("=== Existing e63s models (verified) ===")
    for name, layers, hidden, ffn in EXISTING:
        describe(name, shape_cfg(layers, hidden, ffn))

    print("")
    print("=== Candidate ~1M param configurations ===")
    for name, layers, hidden, ffn in CANDIDATES:
        try:
            describe(name, shape_cfg(layers, hidden, ffn))
        except Exception as e:
            print("%s: ERROR: %s" % (name, e))

    print("")
    print("=== Configs closest to 1M params ===")
    rows = []
    for name, layers, hidden, ffn in CANDIDATES:
        try:
            cfg = shape_cfg(layers, hidden, ffn)
            rows.append((name, cfg, count_params(cfg)))
        except Exception:
            pass
    rows.sort(key=lambda r: abs(r[2] - 1_000_000))
    for name, cfg, params in rows[:8]:
        print("%s: %dL, %dh, %dffn -> %s params (%sM)"
              % (name, cfg["layers"], cfg["hidden"], cfg["ffn"],
                 format(params, ","), round(params / 1e6, 2)))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="a training config yaml; its model: block is counted")
    ap.add_argument("--layers", type=int)
    ap.add_argument("--hidden", type=int)
    ap.add_argument("--ffn", type=int)
    ap.add_argument("--n-heads", dest="n_heads", type=int, default=8)
    ap.add_argument("--n-kv", dest="n_kv", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=128)
    args = ap.parse_args(argv)

    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            raise SystemExit("%s is not a mapping" % args.config)
        describe(Path(args.config).stem, cfg)
        return 0

    given = {"layers": args.layers, "hidden": args.hidden, "ffn": args.ffn}
    if any(v is not None for v in given.values()):
        missing = [k for k, v in given.items() if v is None]
        if missing:
            raise SystemExit("shape needs all of --layers, --hidden, --ffn (missing %s)"
                             % ", ".join("--" + m for m in missing))
        describe("shape", shape_cfg(args.layers, args.hidden, args.ffn,
                                    args.n_heads, args.n_kv, args.ctx))
        return 0

    grid()
    return 0


if __name__ == "__main__":
    sys.exit(main())
