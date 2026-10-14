"""CLI for tashkeel-nano-1p1m.

  python diacritize.py --text "السلام عليكم"
  python diacritize.py --file input.txt
  echo "مرحبا بالعالم" | python diacritize.py

Loads ahmadsy/tashkeel-nano-1p1m from the HF hub (or a local snapshot).
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser(description="Arabic diacritization - tashkeel-nano-1p1m")
    ap.add_argument("--text", default=None)
    ap.add_argument("--file", default=None)
    ap.add_argument("--repo", default="ahmadsy/tashkeel-nano-1p1m")
    ap.add_argument("--device", default=None, help="cpu / cuda (default: auto)")
    ap.add_argument("--decode", default="centered", choices=["centered", "single"],
                    help="centered (published, default) or single hard-chunk")
    args = ap.parse_args()

    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tashkeel_nano import TashkeelNano

    model = TashkeelNano.from_pretrained(args.repo, device=args.device)
    print(f"[loaded] {args.repo}  params="
          f"{sum(p.numel() for p in model.model.parameters()):,}", file=sys.stderr)

    if args.text:
        lines = args.text.splitlines()
    elif args.file:
        lines = open(args.file, encoding="utf-8").read().splitlines()
    else:
        lines = sys.stdin.read().splitlines()
    for line in lines:
        print(model.diacritize(line, decode=args.decode))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    main()
