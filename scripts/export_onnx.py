"""Export a published model package to ONNX for the browser demo.

Usage:

    python scripts/export_onnx.py --run models/nano1p1m --out demo/static/models/nano1p1m.onnx
    python scripts/export_onnx.py --run models/nano4p5m --out demo/static/models/nano4p5m.onnx

The graph has one input, input_ids (int64, [batch, seq] over the character
vocabulary of tashkeel_nano.tokenizer), and one output, logits (fp32,
[batch, seq, 15]): raw per-position class scores, no softmax. Mark order,
whitespace handling and windowing are applied by the renderer
(src/tashkeel_nano/inference.py for Python, demo/static/js/engine.js for the
browser), so one exported graph serves every decode arm and the two models
differ only in their weights.

opset 17 has no RMSNormalization op, so each RMSNorm is traced as its own
arithmetic (Pow, ReduceMean, Sqrt, Div, Mul) - the same shape the committed
graphs under demo/static/models/ have. Pass --verify to re-check the file
against the torch module on random ids before shipping it, and run
tests/test_onnx_parity.py (CI does) to prove the graph still picks the same
class as the package at every position. The serialized bytes need not match an
earlier export - constant folding is not stable - only the predictions.
"""
from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def _rms_norm_decomposed(x, normalized_shape, weight=None, eps=None):
    """opset-17 RMSNorm: x * rsqrt(mean(x^2) + eps) * weight, computed in fp32."""
    import torch
    if eps is None:
        eps = torch.finfo(x.dtype).eps
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        out = out * weight.float()
    return out.to(x.dtype)


@contextlib.contextmanager
def _decomposed_norm():
    """Trace RMSNorm as arithmetic for the duration of the export."""
    import torch.nn.functional as F
    original = F.rms_norm
    F.rms_norm = _rms_norm_decomposed
    try:
        yield
    finally:
        F.rms_norm = original


def _wrap(model):
    """Wrap the module so the traced output is the bare logits tensor."""
    import torch

    class _W(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids):
            return self.m(input_ids)["logits"]

    return _W(model).eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="package or run directory")
    ap.add_argument("--out", required=True, help="destination .onnx file")
    ap.add_argument("--config", default="training_config.yaml")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--verify", action="store_true",
                    help="re-run the file against torch on random ids")
    args = ap.parse_args()

    import torch
    from tashkeel_nano.inference import TashkeelNano
    from tashkeel_nano.tokenizer import VOCAB_SIZE

    nano = TashkeelNano.from_run(args.run, config=args.config, device="cpu")
    seq = min(int(nano.ctx), 32)
    dummy = torch.randint(0, VOCAB_SIZE, (1, seq), dtype=torch.long)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with _decomposed_norm():
        torch.onnx.export(
            _wrap(nano.model), (dummy,), str(out),
            input_names=["input_ids"], output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                          "logits": {0: "batch", 1: "seq"}},
            opset_version=args.opset, do_constant_folding=True, dynamo=False)
    print("[exported] %s (%d bytes, opset %d)" % (out, out.stat().st_size, args.opset))

    if args.verify:
        import numpy as np
        import onnxruntime as ort
        ids = torch.randint(0, VOCAB_SIZE, (2, 48), dtype=torch.long)
        with torch.no_grad():
            want = nano.model(ids)["logits"].numpy()
        got = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"]).run(
            None, {"input_ids": ids.numpy()})[0]
        diff = float(np.abs(got - want).max())
        print("[verify] max |onnx - torch| = %.3e" % diff)
        if diff > 1e-4:
            raise SystemExit("exported graph does not match torch")
        print("[verify] PASS")


if __name__ == "__main__":
    main()
