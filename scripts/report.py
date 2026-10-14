"""Full-metric evaluation report (delegates to tashkeel_nano.eval_report).

    python scripts/report.py --ref data/raw/wikinews/wikinews2024_multi_ref.diac \
        --manifest models.json --out-json out/metrics.json --out-md out/metrics.md

    python scripts/report.py --ref REF --pred arm_a=pred_a.txt --pred arm_b=pred_b.txt

This is a thin wrapper so the driver can also be imported as a library:
    from tashkeel_nano import eval_report
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tashkeel_nano import eval_report  # noqa: E402


def main():
    fn = getattr(eval_report, "main", None)
    if fn is None:
        raise SystemExit("tashkeel_nano.eval_report exposes no main(); use its API directly")
    return fn()


if __name__ == "__main__":
    sys.exit(main())
