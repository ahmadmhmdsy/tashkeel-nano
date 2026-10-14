"""Config-driven corpus and benchmark downloader (R51).

Everything is declared in configs/data.yaml: which sources exist, where each one
lives upstream, its license, and whether it may be fetched without an explicit
acknowledgement. Nothing about a source is hard-coded here, so adding a corpus is
a config edit, not a code change.

    python scripts/download_data.py --list
    python scripts/download_data.py                          # non-gated sources
    python scripts/download_data.py --source fadel,wikinews
    python scripts/download_data.py --accept-license         # include GPL-2 source
    python scripts/download_data.py --hf-token $HF_TOKEN     # or env HF_TOKEN

Output:
    data/raw/<dir>/...        files exactly as published upstream
    data/raw/provenance.json  per-source record: url, license, commercial-use
                              flag, acceptance, byte count, sha256

CPU + network only, so it is safe beside a training run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO / "configs" / "data.yaml"
UA = "tashkeel-nano/1.0 (+https://github.com/ahmadmhmdsy/tashkeel-nano)"


def load_config(path):
    import yaml
    p = Path(path)
    if not p.exists():
        raise SystemExit("config not found: %s" % p)
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg.get("catalog"), dict) or not cfg["catalog"]:
        raise SystemExit("config %s has no non-empty 'catalog' section" % p)
    return cfg


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def http_download(url, dest, token=None, force=False):
    """Single-file download. Skips when the file already exists and is non-empty."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print("  [skip] %s (%d bytes)" % (dest.name, dest.stat().st_size))
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": UA}
    if token:
        headers["Authorization"] = "Bearer " + token
    tmp = dest.with_name(dest.name + ".part")
    req = Request(url, headers=headers)
    with urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)
    tmp.replace(dest)
    print("  [ok]   %s (%d bytes)" % (dest.name, dest.stat().st_size))
    return dest


def hf_download(repo, dest, repo_type="dataset", allow_patterns=None,
                token=None, force=False):
    """Fetch a HF repo into dest/ (the published file layout is preserved)."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("this source needs huggingface_hub: pip install huggingface_hub")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    local = snapshot_download(repo_id=repo, repo_type=repo_type,
                              local_dir=str(dest), allow_patterns=allow_patterns,
                              token=token, force_download=force)
    n = sum(1 for p in Path(local).rglob("*") if p.is_file())
    print("  [ok]   %s -> %s (%d files)" % (repo, dest, n))
    return Path(local)


def git_clone(repo, dest, force=False):
    dest = Path(dest)
    if (dest / ".git").exists() and not force:
        print("  [skip] %s already cloned" % dest.name)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1", repo, str(dest)]
    print("  $ " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return dest


def fetch_one(sid, spec, raw_dir, token, accept_license, force):
    """Fetch one catalog entry. Returns (status, provenance record)."""
    kind = spec.get("kind")
    dest = raw_dir / spec.get("dir", sid)
    rec = {
        "id": sid,
        "kind": kind,
        "title": spec.get("title"),
        "homepage": spec.get("homepage"),
        "license": spec.get("license"),
        "commercial_use": spec.get("commercial_use"),
        "requires_acceptance": bool(spec.get("requires_acceptance")),
        "accepted": bool(accept_license),
        "note": (spec.get("note") or "").strip() or None,
    }
    if rec["requires_acceptance"] and not accept_license:
        print("  [BLOCKED] %s needs --accept-license (%s)" % (sid, rec["license"]))
        rec["status"] = "BLOCKED"
        return "BLOCKED", rec

    before = time.time()
    if kind == "http":
        for name, url in (spec.get("files") or {}).items():
            http_download(url, dest / name, token=token, force=force)
        rec["files"] = sorted(
            {"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in dest.glob("*") if p.is_file())
    elif kind == "hf":
        hf_download(spec["repo"], dest, repo_type=spec.get("repo_type", "dataset"),
                    allow_patterns=spec.get("allow_patterns"), token=token, force=force)
        rec["repo"] = spec["repo"]
        rec["files"] = sorted(
            {"name": str(p.relative_to(dest)).replace("\\", "/"),
             "bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in dest.rglob("*") if p.is_file())
    elif kind == "git":
        git_clone(spec["repo"], dest, force=force)
        rec["repo"] = spec["repo"]
        rec["files"] = []
    else:
        raise SystemExit("source %s has unknown kind %r" % (sid, kind))

    rec["status"] = "PASS"
    rec["dir"] = str(dest.relative_to(raw_dir.parent.parent)).replace("\\", "/")
    rec["total_bytes"] = sum(f["bytes"] for f in rec["files"])
    rec["elapsed_s"] = round(time.time() - before, 1)
    print("  [PASS] %s  %d files  %.1f MB  %.1fs"
          % (sid, len(rec["files"]), rec["total_bytes"] / 1e6, rec["elapsed_s"]))
    return "PASS", rec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--source", default=None,
                    help="comma-separated ids, or 'all' (default: config default_sources)")
    ap.add_argument("--raw-dir", default=None, help="override the config raw_dir")
    ap.add_argument("--hf-token", default=None,
                    help="HF token; falls back to $HF_TOKEN / $HUGGING_FACE_HUB_TOKEN")
    ap.add_argument("--accept-license", action="store_true",
                    help="acknowledge the terms of sources that require it")
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    ap.add_argument("--list", action="store_true", help="show the catalog and exit")
    args = ap.parse_args()

    cfg = load_config(args.config)
    catalog = cfg["catalog"]
    raw_dir = Path(args.raw_dir or cfg.get("raw_dir") or "data/raw")
    if not raw_dir.is_absolute():
        raw_dir = REPO / raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if args.list:
        print("catalog: %s" % args.config)
        print("%-18s %-5s %-9s %-14s %s" % ("id", "kind", "optional", "commercial", "license"))
        for sid, spec in catalog.items():
            print("%-18s %-5s %-9s %-14s %s"
                  % (sid, spec.get("kind"), bool(spec.get("optional")),
                     str(spec.get("commercial_use")), spec.get("license")))
        return 0

    want = args.source or cfg.get("default_sources") or "all"
    if want == "all":
        selected = [s for s, spec in catalog.items() if not spec.get("optional")]
    else:
        selected = [s.strip() for s in str(want).split(",") if s.strip()]
    unknown = [s for s in selected if s not in catalog]
    if unknown:
        raise SystemExit("unknown source(s) %r; known: %s"
                         % (unknown, ", ".join(catalog)))

    print("=== tashkeel-nano download_data ===")
    print("config  : %s" % args.config)
    print("raw dir : %s" % raw_dir)
    print("sources : %s" % ", ".join(selected))
    print("hf token: %s" % ("present" if token else "absent"))
    print("license : %s" % ("accepted" if args.accept_license else "NOT accepted (gated sources will be skipped)"))
    records, blocked = [], 0
    for sid in selected:
        print("[%s]" % sid)
        try:
            status, rec = fetch_one(sid, catalog[sid], raw_dir, token,
                                    args.accept_license, args.force)
        except Exception as exc:                      # keep going: partial fetch is useful
            status = "FAIL"
            rec = {"id": sid, "status": "FAIL", "error": repr(exc)[:400]}
            print("  [FAIL] %s: %r" % (sid, exc))
        records.append(rec)
        blocked += status == "BLOCKED"

    prov = {
        "generator": "scripts/download_data.py",
        "config": str(args.config),
        "accept_license": bool(args.accept_license),
        "sources": records,
    }
    out = raw_dir / "provenance.json"
    out.write_text(json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(1 for r in records if r["status"] == "PASS")
    print("\nprovenance -> %s" % out)
    print("PASS %d / %d   BLOCKED %d   FAIL %d"
          % (ok, len(records), blocked, len(records) - ok - blocked))
    return 0 if ok and not blocked else (2 if blocked and not ok else 1)


if __name__ == "__main__":
    sys.exit(main())
