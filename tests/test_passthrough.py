"""Passthrough codec: unit segmentation, bare rendering, quarantine surface.

The contract (DESIGN.md section 4, decision D6) is that parse() yields one
target unit per Arabic base letter (carrying that letter's 15-class label) and
passthrough units for everything else; the class marks themselves are consumed
by their target unit. reconstruct() therefore re-renders the input with exactly
the class marks removed and EVERY other code point verbatim, and the passthrough
stream is byte-exact.
"""
import unicodedata

import pytest

from tashkeel_nano import labels as L
from tashkeel_nano import passthrough as PT

CASES = [
    "",
    "السلام عليكم",
    "مرحبا 😀 café عليكم ✌️",
    "IPv6 في https://example.com/a?b=1&c=2 و 3.14",
    "العام 1446 || 2025 م",
    "أ\u200fب\u200dج\u200bد \ufeffهنا",
    "ســـلام",
    "بِسْمِ ٱللَّهِ",
    "WiFiجهاز GPT-4o نموذج",
    "عليىٰ",
    ". , ! ? ; : ... مفتاح",
    "أهلا 你好",
    "المدى \u200eRock\u200e سلام",
    "\u064eالسلام",
    "بِسْم\u06d6للّه\u06d0",
]


def bare_render(text):
    """What parse() + reconstruct() must produce: input minus the class marks."""
    out, i, n = [], 0, len(text)
    while i < n:
        ch = text[i]
        out.append(ch)
        i += 1
        if PT.is_arabic_base(ch):
            while i < n and text[i] in L._CONSUMABLE:
                i += 1
    return "".join(out)


def passthrough_stream(text):
    return "".join(c for c in bare_render(text) if not PT.is_arabic_base(c))


@pytest.mark.parametrize("text", CASES)
def test_reconstruct_is_the_input_minus_the_class_marks(text):
    assert PT.reconstruct(PT.parse(text)) == bare_render(text)


@pytest.mark.parametrize("text", CASES)
def test_units_cover_exactly_the_bare_text(text):
    assert sum(len(u.raw) for u in PT.parse(text)) == len(bare_render(text))


@pytest.mark.parametrize("text", CASES)
def test_passthrough_units_are_byte_exact(text):
    units = PT.parse(text)
    assert "".join(u.raw for u in units if not u.is_target) == passthrough_stream(text)


@pytest.mark.parametrize("text", CASES)
def test_target_units_are_exactly_the_arabic_base_letters(text):
    units = PT.parse(text)
    targets = [u for u in units if u.is_target]
    assert len(targets) == sum(1 for ch in text if PT.is_arabic_base(ch))
    assert all(len(u.raw) == 1 and PT.is_arabic_base(u.raw) for u in targets)


@pytest.mark.parametrize("text", CASES)
def test_every_target_unit_carries_a_valid_class_id(text):
    for u in PT.parse(text):
        if u.is_target:
            assert 0 <= u.label < L.N_CLASSES
        else:
            assert u.label is None


def test_parse_is_idempotent_over_its_own_reconstruction():
    text = "بِسْمِ اللهِ الرحمنِ 42 😀"
    once = PT.reconstruct(PT.parse(text))
    assert PT.reconstruct(PT.parse(once)) == once


def test_strip_then_relabel_round_trips_up_to_unicode_normalization():
    # the decoder emits canonical NFC while a raw Tashkeela fixture is
    # shadda-first, so the round trip is exact up to NFC -- the contract that
    # matters, since every benchmark reference is NFC.
    sent = "بِسْمِّهِ \u200fاللَّهِ الرَّحْمَنِ"
    units = PT.parse(sent)
    rebuilt = "".join(u.raw if not u.is_target else u.raw + L.marks_for_label(u.label)
                      for u in units)
    assert unicodedata.normalize("NFC", rebuilt) == unicodedata.normalize("NFC", sent)


def test_the_target_stream_carries_no_class_marks():
    bases = "".join(u.raw for u in PT.parse("بِسْمِّهِ الرَّحْمَنِ") if u.is_target)
    assert not any(c in L._CONSUMABLE for c in bases)


@pytest.mark.parametrize("bad", ["بّْ", "بََ", "بٌٌ", "بِِّ"])
def test_illegal_mark_runs_are_quarantined(bad):
    with pytest.raises(PT.QuarantineError):
        PT.parse(bad)


def test_quarantine_reports_the_offending_index():
    with pytest.raises(PT.QuarantineError) as exc:
        PT.parse("ابََ")
    assert exc.value.index == 1


def test_labels_helper_marks_passthrough_units_as_none():
    units = PT.parse("بِسْمِ x")
    assert PT.labels(units) == [L.C_KASRA, L.C_SUKUN, L.C_KASRA, None]
