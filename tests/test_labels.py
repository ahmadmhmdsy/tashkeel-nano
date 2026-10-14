"""Label schema: 15 classes, lossless marks round trip, NFC-canonical output."""
import unicodedata

import pytest

from tashkeel_nano import labels as L


def test_fifteen_classes_with_stable_names():
    assert L.N_CLASSES == 15
    assert len(L.CLASS_NAMES) == 15
    assert L.CLASS_NAMES[0] == "bare"
    assert L.CLASS_NAMES[-1] == "shadda_kasratan"


def test_class_ids_are_the_canonical_order():
    ids = [L.C_BARE, L.C_FATHA, L.C_DAMMA, L.C_KASRA, L.C_SUKUN,
           L.C_FATHATAN, L.C_DAMMATAN, L.C_KASRATAN,
           L.C_SHADDA, L.C_SHADDA_FATHA, L.C_SHADDA_DAMMA, L.C_SHADDA_KASRA,
           L.C_SHADDA_FATHATAN, L.C_SHADDA_DAMMATAN, L.C_SHADDA_KASRATAN]
    assert ids == list(range(15))


def test_bare_has_no_marks():
    assert L.marks_for_label(L.C_BARE) == ""


@pytest.mark.parametrize("label", [l for l in range(15) if l != L.C_BARE])
def test_marks_round_trip_through_label_for_marks(label):
    marks = L.marks_for_label(label)
    assert marks
    assert L.label_for_marks(tuple(marks)) == label


def test_output_is_always_nfc_normalized():
    for label in range(15):
        m = L.marks_for_label(label)
        assert unicodedata.normalize("NFC", m) == m, label


def test_nfc_puts_fatha_before_shadda():
    # fatha has combining class 30, shadda 33, so canonical order is fatha first.
    m = L.marks_for_label(L.C_SHADDA_FATHA)
    assert m.index(L.FATHA) < m.index(L.SHADDA)


def test_both_source_mark_orders_map_to_the_same_class():
    # real corpora contain shadda+vowel and vowel+shadda; both are gemination
    assert L.label_for_marks((L.SHADDA, L.FATHA)) == L.C_SHADDA_FATHA
    assert L.label_for_marks((L.FATHA, L.SHADDA)) == L.C_SHADDA_FATHA


def test_illegal_mark_sequences_are_rejected():
    with pytest.raises(ValueError):
        L.label_for_marks((L.FATHA, L.FATHA))
    with pytest.raises(ValueError):
        L.label_for_marks((L.SHADDA, L.SUKUN))
