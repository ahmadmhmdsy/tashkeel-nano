"""Uniform weight soup over a probe trajectory - the published nano1p1m recipe.

The released tashkeel-nano-1p1m weights are NOT the final training step: they are
the elementwise fp32 mean of 12 step probes spanning steps 32500..60000 of run
e98_soupB, cast back to the members' dtype. This script rebuilds exactly that
from a run directory's gate_probe/weights_step*.pt files.

    # the published 1p1m soup
    python scripts/soup.py --run runs/e98_soupB --members 32500:60000:2500 \
        --out runs/e98_soupB/soupB_uniform

    # any explicit member list
    python scripts/soup.py --probes runs/e98_soupB/gate_probe \
        --members 32500,35000,37500 --out /tmp/soup3 --name my_soup

Members are averaged in ASCENDING STEP ORDER. The output keeps the members'
wrapper shape ({model, step, config}) so the existing loaders read it unchanged,
and a soup.json record is written beside it with every member path, its sha256,
and the relative L2 distance from the soup to each member (the basin check the
E-073 work used).

CPU only. No GPU is touched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parent.parent


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def probe_step(path):
    m = re.search(r"weights_step(\d+)", path.name)
    return int(m.group(1)) if m else -1


def parse_members(spec, probes_dir):
    """'32500:60000:2500' or '32500,35000' or 'all' -> ordered member paths."""
    available = sorted(probes_dir.glob("weights_step*.pt"), key=probe_step)
    if not available:
        raise SystemExit("no weights_step*.pt under %s" % probes_dir)
    by_step = {probe_step(p): p for p in available}
    if spec in (None, "", "all"):
        return available
    if ":" in spec:
        parts = [int(x) for x in spec.split(":")]
        if len(parts) != 3:
            raise SystemExit("range spec must be start:stop:step, got %r" % spec)
        lo, hi, st = parts
        want = list(range(lo, hi + 1, st))
    else:
        want = [int(x) for x in spec.split(",") if x.strip()]
    missing = [s for s in want if s not in by_step]
    if missing:
        raise SystemExit("no probe for step(s) %r; available: %s"
                         % (missing, sorted(by_step)))
    return [by_step[s] for s in want]


def soup(paths, out_path, tag):
    first = torch.load(paths[0], map_location="cpu", weights_only=False)
    dtypes = {k: v.dtype for k, v in first["model"].items()}
    acc = None
    for p in paths:
        sd = torch.load(p, map_location="cpu", weights_only=False)["model"]
        if acc is None:
            acc = {k: v.to(torch.float32).clone() for k, v in sd.items()}
        else:
            for k, v in sd.items():
                acc[k] += v.to(torch.float32)
    n = len(paths)
    out_sd = {k: (acc[k] / n).to(dtypes[k]) for k in acc}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": out_sd, "step": "soup_%s_%d" % (tag, n),
                "config": first.get("config")}, out_path)
    # A config.yaml beside model.pt, so score_gate.py, infer.py and
    # TashkeelNano.from_run find the architecture without being told: the soup
    # keeps the first member's config verbatim. from_run also falls back to the
    # copy embedded in model.pt, so an older soup directory still loads.
    member_cfg = first.get("config") if isinstance(first, dict) else None
    if isinstance(member_cfg, str):
        try:
            member_cfg = json.loads(member_cfg)   # train.py stores json.dumps(cfg)
        except ValueError:
            member_cfg = None
    if isinstance(member_cfg, dict):
        (out_path.parent / "config.yaml").write_text(
            yaml.safe_dump(member_cfg, sort_keys=False, allow_unicode=True),
            encoding="utf-8")

    dists = []
    for p in paths:
        sd = torch.load(p, map_location="cpu", weights_only=False)["model"]
        num = sum(float(((out_sd[k].float() - sd[k].float()) ** 2).sum()) for k in out_sd)
        den = sum(float((sd[k].float() ** 2).sum()) for k in out_sd)
        dists.append(math.sqrt(num / max(den, 1e-12)))
    stats = {"tag": tag, "members": len(paths), "out": str(out_path),
             "rel_l2": {"min": min(dists), "mean": sum(dists) / len(dists), "max": max(dists)}}
    print("%-22s n=%2d  rel_L2 to members min=%.4f mean=%.4f max=%.4f  -> %s"
          % (tag, n, stats["rel_l2"]["min"], stats["rel_l2"]["mean"],
             stats["rel_l2"]["max"], out_path))
    return out_path, stats, dists


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None, help="run dir; probes are read from <run>/gate_probe")
    ap.add_argument("--probes", default=None, help="explicit probe dir (overrides --run)")
    ap.add_argument("--members", default="all",
                    help="'start:stop:step', a comma list of steps, or 'all'")
    ap.add_argument("--out", default=None, help="output dir (default <run>/soup_uniform)")
    ap.add_argument("--name", default=None, help="soup tag (default: the output dir name)")
    args = ap.parse_args()

    if not (args.run or args.probes):
        ap.error("give --run or --probes")
    probes_dir = Path(args.probes) if args.probes else Path(args.run) / "gate_probe"
    if not probes_dir.is_absolute():
        probes_dir = REPO / probes_dir
    out_dir = Path(args.out) if args.out else (Path(args.run) / "soup_uniform")
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    tag = args.name or out_dir.name

    members = parse_members(args.members, probes_dir)
    steps = [probe_step(p) for p in members]
    print("=== tashkeel-nano soup ===")
    print("probes  : %s" % probes_dir)
    print("members : %d  steps %s" % (len(members), steps))
    print("out     : %s" % out_dir)

    out_path, stats, dists = soup(members, out_dir / "model.pt", tag)
    record = {
        "generator": "scripts/soup.py",
        "tag": tag,
        "method": "uniform (elementwise fp32 mean), cast back to member dtype",
        "members": [
            {"step": probe_step(p), "path": str(p.relative_to(REPO)).replace("\\", "/"),
             "sha256": sha256_file(p), "rel_l2_to_soup": round(d, 6)}
            for p, d in zip(members, dists)
        ],
        "out": str(out_path.relative_to(REPO)).replace("\\", "/"),
        "out_sha256": sha256_file(out_path),
        "rel_l2": {k: round(v, 6) for k, v in stats["rel_l2"].items()},
    }
    (out_dir / "soup.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("record  : %s" % (out_dir / "soup.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
