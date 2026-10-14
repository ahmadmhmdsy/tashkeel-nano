"""Inference: decode helpers, synthesis checks, and a real end-to-end pass."""
import json
from pathlib import Path

import pytest
import torch

from tashkeel_nano import inference as INF
from tashkeel_nano.model import build_from_config

REPO = Path(__file__).resolve().parent.parent
MARK_CHARS = {chr(cp) for lo, hi in INF.MARK_RANGES for cp in range(lo, hi + 1)}
TINY = {"model": {"layers": 2, "hidden": 16, "n_heads": 2, "n_kv": 1,
                  "ffn": 32, "ctx": 64, "causal": False}}


def tiny_model():
    return build_from_config(TINY, vocab_size=INF.VOCAB_SIZE)


# --- decode helpers --------------------------------------------------------

def test_window_starts_single_window_when_it_fits():
    assert INF.window_starts(10, 10, 5) == [0]
    assert INF.window_starts(1, 10, 5) == [0]


def test_window_starts_always_covers_the_tail():
    for n, w, s in ((26, 10, 5), (30, 10, 5), (31, 10, 7), (100, 16, 8)):
        starts = INF.window_starts(n, w, s)
        assert starts[0] == 0
        assert starts[-1] == n - w
        assert all(start <= n - w for start in starts)
        covered = set()
        for start in starts:
            covered |= set(range(start, min(start + w, n)))
        assert covered == set(range(n)), (n, w, s)


def test_strip_marks_drops_marks_and_normalizes():
    assert INF.strip_marks("بِسْمِ") == "بسم"
    assert INF.strip_marks("abc 123") == "abc 123"


def test_reorder_shadda_first_moves_shadda_to_the_front_of_the_run():
    vowel_first = "\u0628\u064e\u0651"
    assert INF.reorder_shadda_first(vowel_first) == "\u0628\u0651\u064e"
    # already shadda-first: unchanged
    assert INF.reorder_shadda_first("\u0628\u0651\u064e") == "\u0628\u0651\u064e"
    # no shadda: unchanged
    assert INF.reorder_shadda_first("\u0628\u064e") == "\u0628\u064e"
    # stray shadda is still hoisted
    assert INF.reorder_shadda_first("\u0628\u0651") == "\u0628\u0651"


def test_presets_match_the_published_packages():
    assert INF.resolve_decode("ahmadsy/tashkeel-nano-1p1m") == {
        "decode": "centered", "order": "nfc", "window": 256, "stride": 128}
    assert INF.resolve_decode("ahmadsy/tashkeel-nano-4p5m") == {
        "decode": "single", "order": "shadda_first"}


def test_local_package_directories_resolve_to_the_published_decode():
    assert INF.resolve_decode("models/nano1p1m") == {
        "decode": "centered", "order": "nfc", "window": 256, "stride": 128}
    assert INF.resolve_decode("models/nano4p5m") == {
        "decode": "single", "order": "shadda_first"}


def test_config_json_overrides_a_preset():
    cfg = {"decoding": {"arm": "nospace_centered", "window": 512}}
    out = INF.resolve_decode("ahmadsy/tashkeel-nano-1p1m", cfg)
    assert out["decode"] == "nospace_centered"
    assert out["window"] == 512
    assert out["stride"] == 128          # untouched


def test_unknown_model_gets_safe_defaults():
    assert INF.resolve_decode("someone/unknown") == {"decode": "centered", "order": "nfc"}


def test_bad_decode_and_order_are_rejected():
    with pytest.raises(ValueError):
        INF.TashkeelNano(tiny_model(), decode="nope")
    with pytest.raises(ValueError):
        INF.TashkeelNano(tiny_model(), order="nope")


# --- end to end on a synthetic model ---------------------------------------

@pytest.mark.parametrize("decode", INF.DECODE_ARMS)
@pytest.mark.parametrize("order", INF.ORDERS)
def test_output_preserves_base_letters_exactly(decode, order):
    model = INF.TashkeelNano(tiny_model(), ctx=64, decode=decode, order=order,
                             window=32, stride=16)
    text = "مرحبا بكم في هذا الاختبار 42"
    out = model.diacritize(text)
    assert INF.strip_marks(out) == INF.strip_marks(text)


@pytest.mark.parametrize("decode", INF.DECODE_ARMS)
def test_output_only_adds_marks_after_arabic_base_letters(decode):
    from tashkeel_nano.passthrough import is_arabic_base
    model = INF.TashkeelNano(tiny_model(), ctx=64, decode=decode, window=32, stride=16)
    text = "السلام عليكم 42 abc"
    out = model.diacritize(text)
    for i, ch in enumerate(out):
        if ch not in MARK_CHARS:
            continue
        j = i
        while j > 0 and out[j - 1] in MARK_CHARS:
            j -= 1
        assert j > 0 and is_arabic_base(out[j - 1]), (i, out)


def test_blank_input_is_returned_unchanged():
    model = INF.TashkeelNano(tiny_model(), ctx=64)
    assert model.diacritize("   ") == "   "
    assert model.diacritize("") == ""


def test_nfc_output_is_canonical():
    import unicodedata
    model = INF.TashkeelNano(tiny_model(), ctx=64, order="nfc", window=32, stride=16)
    out = model.diacritize("الله محمد")
    assert unicodedata.normalize("NFC", out) == out


def test_shadda_first_output_is_not_nfc_but_round_trips_through_the_helper():
    model = INF.TashkeelNano(tiny_model(), ctx=64, order="shadda_first",
                             window=32, stride=16)
    out = model.diacritize("الله محمد")
    assert INF.reorder_shadda_first(INF.reorder_shadda_first(out)) == out


def test_nospace_arm_marks_the_same_letters_as_the_spaced_arm():
    from tashkeel_nano.passthrough import is_arabic_base
    text = "مرحبا بكم"
    a = INF.TashkeelNano(tiny_model(), ctx=64, decode="single", window=32, stride=16)
    b = INF.TashkeelNano(tiny_model(), ctx=64, decode="nospace", window=32, stride=16)
    assert [c for c in a.diacritize(text) if is_arabic_base(c)] == \
           [c for c in b.diacritize(text) if is_arabic_base(c)]


@pytest.mark.skipif(not (REPO / "models" / "nano1p1m" / "model.pt").exists(),
                    reason="published model packages are not present")
def test_published_packages_load_and_agree_with_their_golden_file():
    golden_p = REPO / "tests" / "fixtures" / "published_outputs.json"
    if not golden_p.exists():
        pytest.skip("tests/fixtures/published_outputs.json has not been recorded")
    golden = json.loads(golden_p.read_text(encoding="utf-8"))
    for name, spec in golden.items():
        model = INF.TashkeelNano.from_run(
            REPO / "models" / name, config="training_config.yaml",
            decode=spec["decode"], order=spec["order"])
        assert sum(p.numel() for p in model.model.parameters()) == spec["params"]
        for case in spec["cases"]:
            got = model.diacritize(case["input"])
            assert got == case["output"], (name, case["input"], got, case["output"])
