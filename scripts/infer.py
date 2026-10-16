"""Diacritize Arabic text with a published checkpoint or a local training run.

    # published models (decode defaults come from tashkeel_nano.inference.PRESETS)
    python scripts/infer.py --repo ahmadsy/tashkeel-nano-1p1m --text "السلام عليكم"
    python scripts/infer.py --repo ahmadsy/tashkeel-nano-4p5m --file in.txt --out out.txt

    # a committed package, no download needed (models/nano1p1m/)
    python scripts/infer.py --run models/nano1p1m --config training_config.yaml \
        --text "مرحبا"

    # pick a decode arm explicitly
    python scripts/infer.py --repo ahmadsy/tashkeel-nano-1p1m --decode nospace_centered

One input line -> one output line. Text that carries no Arabic letters is echoed
unchanged. Output encoding is UTF-8.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tashkeel_nano.inference import DECODE_ARMS, ORDERS, TashkeelNano  # noqa: E402


def read_lines(args):
    if args.text is not None:
        return args.text.splitlines()
    if args.file:
        return Path(args.file).read_text(encoding="utf-8").splitlines()
    return sys.stdin.read().splitlines()


def main():
    ap = argparse.ArgumentParser(description="Arabic diacritization - tashkeel-nano")
    src = ap.add_argument_group("model")
    src.add_argument("--repo", default=None, help="Hugging Face repo id")
    src.add_argument("--run", default=None, help="local run dir (model.pt + config.yaml)")
    src.add_argument("--ckpt", default=None, help="explicit checkpoint file")
    src.add_argument("--config", default=None, help="yaml model shape (with --ckpt)")
    src.add_argument("--preset", default=None,
                     help="decode preset name (default: the repo basename)")
    src.add_argument("--device", default=None, help="cpu / cuda (default: auto)")
    src.add_argument("--hf-token", default=None)
    dec = ap.add_argument_group("decode")
    dec.add_argument("--decode", default=None, choices=list(DECODE_ARMS))
    dec.add_argument("--order", default=None, choices=list(ORDERS),
                     help="nfc (canonical) or shadda_first (nano4p5m's published render)")
    dec.add_argument("--nospace", action="store_true",
                     help="strip whitespace from what the encoder sees (E-094)")
    dec.add_argument("--window", type=int, default=None)
    dec.add_argument("--stride", type=int, default=None)
    io = ap.add_argument_group("input / output")
    io.add_argument("--text", default=None)
    io.add_argument("--file", default=None)
    io.add_argument("--out", default=None, help="write here instead of stdout")
    args = ap.parse_args()

    if not (args.repo or args.run or args.ckpt):
        ap.error("give one of --repo, --run or --ckpt")

    if args.repo:
        model = TashkeelNano.from_pretrained(
            args.repo, token=args.hf_token, device=args.device, preset=args.preset,
            decode=args.decode, order=args.order,
            nospace=(True if args.nospace else None),
            window=args.window, stride=args.stride)
    elif args.run:
        d = Path(args.run)
        model = TashkeelNano.from_run(
            d, config=args.config or "config.yaml", device=args.device,
            preset=args.preset, decode=args.decode, order=args.order,
            nospace=(True if args.nospace else None),
            window=args.window, stride=args.stride)
    else:
        d = Path(args.ckpt).parent
        model = TashkeelNano.from_run(
            d, ckpt=Path(args.ckpt).name, config=args.config or "config.yaml",
            device=args.device, preset=args.preset, decode=args.decode,
            order=args.order, nospace=(True if args.nospace else None),
            window=args.window, stride=args.stride)

    print("[loaded] %s" % model, file=sys.stderr)
    lines = read_lines(args)
    if not lines:
        print("[warn] no input lines", file=sys.stderr)
    out_lines = [model.diacritize(line) for line in lines]
    if args.out:
        Path(args.out).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        print("[wrote] %s (%d lines)" % (args.out, len(out_lines)), file=sys.stderr)
    else:
        for line in out_lines:
            print(line)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    sys.exit(main())
