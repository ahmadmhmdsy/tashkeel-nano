"""Normalise the mark order in prediction files to Unicode NFC.

    python scripts/normalize_marks.py --dir runs/gate/nano4p5m
    python scripts/normalize_marks.py --dir out/4p5m --out-dir out/4p5m_nfc

The registered references are Unicode NFC. A renderer that writes the shadda
before its vowel disagrees with them on every geminate, so the same weights
score differently depending on the order they emitted their marks in. NFC
canonical ordering sorts combining marks by combining class (fatha 30,
damma 31, kasra 32, shadda 33), so normalise("NFC", text) restores the order
the references use and touches nothing else - not the letters, the vowels or
the whitespace.

Measured on the published 4.5M package (nospace_centered, registered gate):
36.89 % exactly as emitted, 22.90 % normalised. docs/evaluation.md re-runs it.
"""
from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path


def normalize(text: str) -> str:
    """Return text with its combining marks in Unicode canonical order."""
    return unicodedata.normalize("NFC", text)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Normalise diacritic mark order to Unicode NFC.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--dir", required=True, help="directory of <name>.pred.txt files")
    ap.add_argument("--out-dir", default=None,
                    help="write normalised copies here instead of in place")
    ap.add_argument("--suffix", default=".pred.txt", help="file suffix to process")
    args = ap.parse_args(argv)

    src = Path(args.dir)
    if not src.is_dir():
        raise SystemExit("%s is not a directory" % src)
    files = sorted(p for p in src.glob("*" + args.suffix) if p.is_file())
    if not files:
        raise SystemExit("no *%s files under %s" % (args.suffix, src))
    dst = Path(args.out_dir) if args.out_dir else src
    dst.mkdir(parents=True, exist_ok=True)

    for path in files:
        with open(path, encoding="utf-8", newline="") as fh:
            text = fh.read()
        out = normalize(text)
        with open(dst / path.name, "w", encoding="utf-8", newline="") as fh:
            fh.write(out)
        print("%-44s %s" % (path.name, "changed" if out != text else "already NFC"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
