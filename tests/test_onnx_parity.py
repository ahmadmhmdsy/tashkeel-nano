"""The shipped browser graphs must predict what the shipped packages predict.

demo/static/models/*.onnx is executed by ONNX Runtime Web in the visitor's tab,
while models/nano1p1m and models/nano4p5m are the published python packages. The
demo footer claims their output matches. This test pins that: on one real
sentence and four random rows per model, both paths must choose the same
diacritic class at every position. Requires onnxruntime (pip install -e ".[onnx]").
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tashkeel_nano.inference import TashkeelNano           # noqa: E402
from tashkeel_nano.tokenizer import VOCAB_SIZE, encode     # noqa: E402

SAMPLE = "السلام عليكم ورحمة الله وبركاته، هذا نص للاختبار."

CASES = [
    ("models/nano1p1m", "demo/static/models/nano1p1m.onnx"),
    ("models/nano4p5m", "demo/static/models/nano4p5m.onnx"),
]


@pytest.mark.parametrize("run,graph", CASES)
def test_shipped_graph_predicts_the_same_classes_as_the_package(run, graph):
    nano = TashkeelNano.from_run(ROOT / run, config="training_config.yaml", device="cpu")
    session = ort.InferenceSession(str(ROOT / graph), providers=["CPUExecutionProvider"])
    ctx = int(nano.ctx)

    ids = encode(SAMPLE)[:ctx]
    assert ids, "the sample sentence must tokenize"
    rows = [[ids]]
    torch.manual_seed(7)
    rows += torch.randint(0, VOCAB_SIZE, (4, ctx)).unsqueeze(1).tolist()

    positions = 0
    for row in rows:
        x = torch.tensor(row, dtype=torch.long)
        with torch.no_grad():
            want = nano.model(x)["logits"].numpy()
        got = session.run(None, {"input_ids": np.asarray(row, dtype=np.int64)})[0]
        assert got.shape == want.shape
        # fp32 accumulation order differs slightly; the prediction must not.
        assert float(np.abs(want - got).max()) < 1e-3
        bad = int((want.argmax(-1) != got.argmax(-1)).sum())
        assert bad == 0, "%s: %d of %d positions disagree" % (run, bad, want.shape[1])
        positions += int(want.shape[1])

    assert positions > ctx
