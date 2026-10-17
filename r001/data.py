"""R-001 data: a fixed, seeded chunk subset of the training pool, plus the dev slice.

Pool = fadel/train.txt + abdou_tashkeel train + sadeed_tashkeela train (the
training splits in docs/data.md). Gate sets, abdou test and fadel/val.txt are
never sampled; fadel/val.txt supplies only the 500-line dev slice.

Lines are split at spaces into chunks of at most MAX_CHARS diacritized
characters. The sample is bottom-k by md5(SEED:chunk): the N chunks with the
smallest keys. That is a uniform seeded sample that does not depend on reading
order, so each source file is processed by its own worker process and only the
local bottom-k sets are merged. Mark stripping runs on the N winners only.

    python r001/data.py                      # -> data/r001/{train,dev}.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import os
import re
from concurrent.futures import ProcessPoolExecutor
import json
import random
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tashkeel_nano.inference import strip_marks  # noqa: E402

RAW = REPO / "data" / "raw"
MAX_CHARS = 200
SEED = 0


def chunks(line, max_chars=MAX_CHARS):
    """Split at spaces into runs of at most max_chars; a longer word stands alone."""
    out, cur = [], ""
    for word in line.split():
        cand = word if not cur else cur + " " + word
        if len(cand) <= max_chars or not cur:
            cur = cand
        else:
            out.append(cur)
            cur = word
    if cur:
        out.append(cur)
    return out


MARK = re.compile("[ً-ْ]")


def pool_files():
    files = [("txt", str(RAW / "fadel" / "train.txt"), None)]
    files += [("pq", str(f), "vocalized") for f in sorted((RAW / "abdou_tashkeel" / "data").glob("train-*.parquet"))]
    files += [("pq", str(f), "output") for f in sorted((RAW / "sadeed_tashkeela" / "data").glob("train-*.parquet"))]
    return files


def file_lines(kind, path, col):
    if kind == "txt":
        yield from Path(path).read_text(encoding="utf-8").splitlines()
        return
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(path).iter_batches(columns=[col], batch_size=20_000):
        for text in batch.column(0).to_pylist():
            yield from str(text).split("\n")


def bottom_k(job):
    """(count, [(key, chunk)]) - this file's k smallest-key chunks that carry marks."""
    (kind, path, col), k = job
    heap, n = [], 0  # max-heap via negated keys
    for line in file_lines(kind, path, col):
        for c in chunks(str(line)):
            if not MARK.search(c):
                continue
            n += 1
            key = int.from_bytes(hashlib.md5(f"{SEED}:{c}".encode()).digest()[:8], "big")
            if len(heap) < k:
                heapq.heappush(heap, (-key, c))
            elif key < -heap[0][0]:
                heapq.heapreplace(heap, (-key, c))
    return n, [(-nk, c) for nk, c in heap]


def pair(diac):
    diac = unicodedata.normalize("NFC", diac.strip())
    plain = strip_marks(diac).strip()
    return {"plain": plain, "diac": diac} if plain and plain != diac else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--dev", type=int, default=500)
    ap.add_argument("--out", default=str(REPO / "data" / "r001"))
    a = ap.parse_args()
    files = pool_files()
    k = a.n + 2_000  # margin for winners that fail pair()
    with ProcessPoolExecutor(max_workers=min(len(files), max(1, (os.cpu_count() or 2) - 2))) as ex:
        parts = list(ex.map(bottom_k, [(f, k) for f in files]))
    seen = sum(n for n, _ in parts)
    cand = sorted(x for _, h in parts for x in h)
    res = []
    for _, c in cand:
        p = pair(c)
        if p:
            res.append(p)
        if len(res) == a.n:
            break
    dev = []
    for line in (RAW / "fadel" / "val.txt").read_text(encoding="utf-8").splitlines():
        for c in chunks(line):
            p = pair(c)
            if p:
                dev.append(p)
    dev = random.Random(SEED).sample(dev, a.dev)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", res), ("dev", dev)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"pool_chunks": seen, "train": len(res), "dev": len(dev),
             "max_chars": MAX_CHARS, "seed": SEED,
             "sampling": "bottom-k md5(seed:chunk)", "files": len(files)}
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(stats)


if __name__ == "__main__":
    main()
