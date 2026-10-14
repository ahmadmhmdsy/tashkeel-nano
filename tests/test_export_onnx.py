"""The export-time RMSNorm decomposition must match torch's own op."""
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "export_onnx.py"
_SPEC = importlib.util.spec_from_file_location("export_onnx", _PATH)
export_onnx = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export_onnx)


def test_decomposed_rms_norm_matches_torch():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)
    w = torch.randn(16)
    got = export_onnx._rms_norm_decomposed(x, (16,), w, 1e-5)
    want = torch.nn.functional.rms_norm(x, (16,), w, 1e-5)
    assert torch.allclose(got, want, atol=1e-6)


def test_decomposed_rms_norm_handles_no_weight():
    torch.manual_seed(1)
    x = torch.randn(3, 4, 8)
    got = export_onnx._rms_norm_decomposed(x, (8,), None, 1e-5)
    want = torch.nn.functional.rms_norm(x, (8,), None, 1e-5)
    assert torch.allclose(got, want, atol=1e-6)
