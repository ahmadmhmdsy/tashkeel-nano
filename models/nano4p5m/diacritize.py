"""CLI for tashkeel-nano-4p5m.

  python diacritize.py --text "السلام عليكم"
  python diacritize.py --file input.txt
  echo "مرحبا بالعالم" | python diacritize.py

Loads ahmadsy/tashkeel-nano-4p5m from the HF hub (or a local snapshot).
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser(description="Arabic diacritization — tashkeel-nano-4p5m")
    ap.add_argument("--text", default=None)
    ap.add_argument("--file", default=None)
    ap.add_argument("--repo", default="ahmadsy/tashkeel-nano-4p5m")
    ap.add_argument("--device", default=None, help="cpu / cuda (default: auto)")
    args = ap.parse_args()

    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tashkeel_nano import TashkeelNano

    model = TashkeelNano.from_pretrained(args.repo, device=args.device)
    print(f"[loaded] {args.repo}  params="
          f"{sum(p.numel() for p in model.model.parameters()):,}", file=sys.stderr)

    if args.text:
        for line in args.text.splitlines():
            print(model.diacritize(line))
    elif args.file:
        for line in open(args.file, encoding="utf-8").read().splitlines():
            print(model.diacritize(line))
    else:
        for line in sys.stdin.read().splitlines():
            print(model.diacritize(line))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    main()
