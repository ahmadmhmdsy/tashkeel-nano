"""Fail if a published markdown page shows a literal ** to the reader.

A line that opens with ** while the previous line is still inside the same
paragraph toggles emphasis at a line boundary: the page renders as alternating
bold and plain fragments and leaves a stray ** on screen. Both model cards
shipped that way once (the summary sentence at the top of each card). This
renders every published page the same way a reader sees it and looks for a **
that survived into the HTML outside a code block.

    python -m pip install markdown-it-py
    python scripts/lint_markdown.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    from markdown_it import MarkdownIt
except ImportError:                                   # pragma: no cover
    print("markdown-it-py is required: python -m pip install markdown-it-py")
    raise SystemExit(2)

REPO = Path(__file__).resolve().parent.parent
#: working directories whose markdown is not a published page
SKIP = {".git", ".venv", "__pycache__", "data", "runs", "out", "scratch",
        "experiments", "research"}


def pages():
    for p in sorted(REPO.rglob("*.md")):
        if SKIP & set(p.relative_to(REPO).parts):
            continue
        yield p


def body(text: str) -> str:
    if text.startswith("---"):                       # drop YAML front matter
        parts = text.split("---", 2)
        return parts[2] if len(parts) > 2 else text
    return text


def main() -> int:
    md = MarkdownIt("commonmark")
    files, leaks = list(pages()), []
    for p in files:
        html = md.render(body(p.read_text(encoding="utf-8")))
        html = re.sub(r"<pre>.*?</pre>", "", html, flags=re.S)
        html = re.sub(r"<code>.*?</code>", "", html, flags=re.S)
        n = len(re.findall(r"\*\*", html))
        if n:
            leaks.append((p.relative_to(REPO).as_posix(), n))
    for path, n in leaks:
        print("LEAK %s: %d literal ** in the rendered page" % (path, n))
    print("%d markdown pages rendered, %d leaking" % (len(files), len(leaks)))
    return 1 if leaks else 0


if __name__ == "__main__":
    sys.exit(main())
