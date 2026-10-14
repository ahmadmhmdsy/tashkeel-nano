"""Score a model on the D-line gate corpora - the registered metric, in one command.

Registered metric (the number in both model cards):
    mean of der_registered over fadel_test, sadeed25, wikinews2014

der_registered is a WORD-level error rate: a word is correct when its mark string
matches ANY of that line's references (references are TAB-separated, exactly as
bench.py wrote them). NOTE that a "/" INSIDE a token is not expanded into
alternatives here -- that is the registered anchor's own behaviour, inherited
from eval_der, and it is what makes the number reproducible. Use
eval_metrics.der_anyalt when the alternative-aware view is wanted; on the gate
corpora the two disagree on ~0.3-0.6 pct of tokens.
It is NOT Sadeed's letter-level "DER (CE)"; their word-level analogue is called
WER (CE) there. Never mix the two.

wikinews2024 is CONTAMINATED (its text overlaps the training pool) and is reported
for continuity only - it is excluded from the registered mean. abdou_test is the
held-out witness: it was never trained on, so it is the honest generalisation check.

    python scripts/score_gate.py --repo ahmadsy/tashkeel-nano-1p1m
    python scripts/score_gate.py --run models/nano1p1m \
        --config training_config.yaml --decode centered --order nfc
    python scripts/score_gate.py --run models/nano4p5m \
        --config training_config.yaml --decode nospace_centered --order nfc
    #   ^ the 4p5m package emits shadda-first; --order nfc normalises the text
    #     to the order the registered references use (36.89 -> 22.90).
    python scripts/score_gate.py --pred-dir out/preds          # score existing files
    python scripts/score_gate.py --pred-dir out/preds --sources fadel_test

Prediction and scoring are separated: with a model the script writes
<out>/<source>.pred.txt and <out>/<source>.ref.txt, then scores them; with
--pred-dir it only scores. GPU is used for prediction, CPU for scoring.
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tashkeel_nano import eval_der  # noqa: E402  (stdlib-only)

RAW = REPO / "data" / "raw"


def _inference():
    """Import the torch-backed inference module on first use.

    Keeps --pred-dir a CPU-only, torch-free scoring path.
    """
    from tashkeel_nano import inference
    return inference


def strip_marks(text):
    return _inference().strip_marks(text)


def _choice(name):
    """An argparse type that validates against inference.<name> lazily."""
    def parse(value):
        allowed = getattr(_inference(), name)
        if value not in allowed:
            raise argparse.ArgumentTypeError("%r not in %r" % (value, tuple(allowed)))
        return value
    return parse

#: the three corpora whose der_registered is averaged into the registered metric
REGISTERED = ("fadel_test", "sadeed25", "wikinews2014")
#: reported for continuity, never in the mean (training-pool overlap)
UNREGISTERED = ("wikinews2024",)
#: never trained on: the honest generalisation witness
WITNESS = ("abdou_test",)
ALL_SOURCES = REGISTERED + UNREGISTERED + WITNESS


def load_pairs(src, limit=None):
    """(bare_input, [reference, ...]) per sentence. Mirrors the gate harness."""
    def ok(text):
        return bool(strip_marks(text).strip())

    pairs = []
    if src == "fadel_test":
        p = RAW / "fadel" / "test.txt"
        if not p.exists():
            raise SystemExit("missing %s - run scripts/download_data.py --source fadel" % p)
        for line in p.read_text(encoding="utf-8").splitlines():
            ref = unicodedata.normalize("NFC", line.strip())
            bare = strip_marks(line)
            if ok(bare):
                pairs.append((bare, [ref]))
    elif src == "abdou_test":
        import pandas as pd
        p = RAW / "abdou_tashkeel" / "data" / "test-00000-of-00001.parquet"
        if not p.exists():
            raise SystemExit("missing %s - run scripts/download_data.py --source abdou_tashkeel" % p)
        for r in pd.read_parquet(p).itertuples():
            gold = unicodedata.normalize("NFC", str(r.vocalized))
            if ok(gold):
                pairs.append((strip_marks(str(r.non_vocalized)) or strip_marks(gold), [gold]))
    elif src == "sadeed25":
        import pandas as pd
        p = RAW / "sadeed_25" / "sadeed25.parquet"
        if not p.exists():
            raise SystemExit("missing %s - run scripts/download_data.py --source sadeed_25" % p)
        for r in pd.read_parquet(p).itertuples():
            gold = unicodedata.normalize("NFC", str(r.output))
            if not ok(gold):
                continue
            g_lines, b_lines = gold.split("\n"), str(r.input).split("\n")
            if len(g_lines) > 1 and len(g_lines) == len(b_lines):
                for g_line, b_line in zip(g_lines, b_lines):
                    bare_l = strip_marks(b_line)
                    if ok(bare_l):
                        pairs.append((bare_l, [g_line]))
            elif ok(gold):
                pairs.append((strip_marks(str(r.input)) or strip_marks(gold), [gold]))
    elif src in ("wikinews2014", "wikinews2024"):
        p = RAW / "wikinews" / ("wikinews%s_multi_ref.diac" % src[-4:])
        if not p.exists():
            raise SystemExit("missing %s - run scripts/download_data.py --source wikinews" % p)
        for line in p.read_text(encoding="utf-8").splitlines():
            sx = line.strip()
            if src == "wikinews2014" and sx.startswith("#"):
                continue
            bare = strip_marks(sx)
            if ok(bare):
                pairs.append((bare, [unicodedata.normalize("NFC", sx)]))
    else:
        raise SystemExit("unknown source %r" % src)

    return pairs[:limit] if limit else pairs


def score_file(pred_path, ref_path):
    preds = Path(pred_path).read_text(encoding="utf-8").splitlines()
    refs = Path(ref_path).read_text(encoding="utf-8").splitlines()
    n = max(len(preds), len(refs))
    per_line = []
    for i in range(n):
        p = preds[i] if i < len(preds) else ""
        alts = refs[i].split("\t") if i < len(refs) else [""]
        per_line.append(eval_der.score_line(p, alts))
    return eval_der.aggregate(per_line)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("model")
    src.add_argument("--repo", default=None)
    src.add_argument("--run", default=None)
    src.add_argument("--config", default="config.yaml",
                     help="yaml beside the checkpoint (published packages ship "
                          "training_config.yaml)")
    src.add_argument("--preset", default=None)
    src.add_argument("--device", default=None)
    src.add_argument("--hf-token", default=None)
    dec = ap.add_argument_group("decode")
    dec.add_argument("--decode", default=None, type=_choice("DECODE_ARMS"))
    dec.add_argument("--order", default=None, type=_choice("ORDERS"))
    dec.add_argument("--nospace", action="store_true")
    dec.add_argument("--window", type=int, default=None)
    dec.add_argument("--stride", type=int, default=None)
    run = ap.add_argument_group("run")
    run.add_argument("--sources", default=",".join(ALL_SOURCES))
    run.add_argument("--limit", type=int, default=None, help="sentences per source")
    run.add_argument("--pred-dir", default=None,
                     help="score <dir>/<source>.pred.txt against <dir>/<source>.ref.txt")
    run.add_argument("--out", default=None, help="where predictions are written")
    run.add_argument("--out-json", default=None)
    args = ap.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = [s for s in sources if s not in ALL_SOURCES]
    if unknown:
        raise SystemExit("unknown source(s) %r; known: %s" % (unknown, ", ".join(ALL_SOURCES)))

    model = None
    if args.pred_dir is None:
        if not (args.repo or args.run):
            ap.error("give --repo or --run, or --pred-dir to score existing files")
        inf = _inference()
        if args.repo:
            model = inf.TashkeelNano.from_pretrained(
                args.repo, token=args.hf_token, device=args.device, preset=args.preset,
                decode=args.decode, order=args.order,
                nospace=(True if args.nospace else None),
                window=args.window, stride=args.stride)
        else:
            model = inf.TashkeelNano.from_run(
                Path(args.run), config=args.config, device=args.device,
                preset=args.preset, decode=args.decode, order=args.order,
                nospace=(True if args.nospace else None),
                window=args.window, stride=args.stride)
        print("[loaded] %s" % model)

    pred_dir = Path(args.pred_dir) if args.pred_dir else Path(
        args.out or (REPO / "runs" / "gate" / "predictions"))
    pred_dir.mkdir(parents=True, exist_ok=True)

    results, registered_vals = {}, []
    for s in sources:
        pred_p, ref_p = pred_dir / ("%s.pred.txt" % s), pred_dir / ("%s.ref.txt" % s)
        if model is not None:
            pairs = load_pairs(s, args.limit)
            if not pairs:
                print("[SKIPPED] %s - no usable sentences" % s)
                continue
            pred_p.write_text("\n".join(model.diacritize(b) for b, _ in pairs) + "\n",
                              encoding="utf-8")
            ref_p.write_text("\n".join("\t".join(r) for _, r in pairs) + "\n",
                             encoding="utf-8")
            print("[pred] %s -> %s (%d lines)" % (s, pred_p, len(pairs)))
        if not (pred_p.exists() and ref_p.exists()):
            print("[SKIPPED] %s - no predictions at %s" % (s, pred_p))
            continue
        agg = score_file(pred_p, ref_p)
        results[s] = agg
        der = agg.get("DER")
        if s in REGISTERED and der is not None:
            registered_vals.append(der)
        print("%-13s DER=%.6f  WER=%.6f  DER_nocase=%.6f  preserve=%.6f  lines=%s"
              % (s, der or 0.0, agg.get("WER", 0.0), agg.get("DER_nocase", 0.0),
                 agg.get("text_preservation", 0.0), agg.get("lines", agg.get("n_lines", "?"))))

    report = {
        "generator": "scripts/score_gate.py",
        "registered_metric": "mean of DER over " + ", ".join(REGISTERED),
        "metric_semantics": ("DER here is WORD-level (a word is correct when its mark "
                             "string matches any reference). It is not Sadeed's "
                             "letter-level DER (CE)."),
        "sources": results,
        "registered_mean_DER": (sum(registered_vals) / len(registered_vals)
                                if registered_vals else None),
        "registered_n": len(registered_vals),
        "excluded_from_mean": list(UNREGISTERED),
        "witness": list(WITNESS),
    }
    print("")
    if registered_vals:
        print("REGISTERED mean DER = %.6f  (n=%d)" % (report["registered_mean_DER"],
                                                      report["registered_n"]))
    else:
        print("REGISTERED mean DER = n/a (no registered source scored)")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("report -> %s" % args.out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
