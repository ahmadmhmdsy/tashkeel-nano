r'''Order-insensitivity and correctness tests for scoring.py.

Run:  python demo/gradio/tests/test_scoring.py
'''
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')

from scoring import (  # noqa: E402
    OUR_ROWS, QUOTED_COLUMNS, QUOTED_ROWS, analyze_word, canonicalize,
    pareto_points, score_pair,
)

FATHA = '\u064E'
SHADDA = '\u0651'
DAMMA = '\u064F'
REH = '\u0631'
SEEN = '\u0633'
KAF = '\u0643'
TEH = '\u062A'
BEH = '\u0628'

# Same word, two spelling conventions for the geminate.
NFC_WORD = REH + FATHA + SEEN + FATHA + SHADDA       # fatha then shadda
LEGACY_WORD = REH + FATHA + SEEN + SHADDA + FATHA    # shadda then fatha

KATABA = KAF + FATHA + TEH + FATHA + BEH + FATHA


def test_perfect_match():
    r = score_pair(KATABA, KATABA)
    assert r['der'] == 0.0 and r['wer'] == 0.0, r
    assert r['der_errors'] == 0 and r['wer_errors'] == 0
    print('perfect match: PASS')


def test_mark_order_is_ignored():
    r = score_pair(NFC_WORD, LEGACY_WORD, canonical=True)
    assert r['der'] == 0.0, 'canonical scoring must ignore mark order: %r' % r
    assert r['wer'] == 0.0
    print('canonical order-insensitivity: PASS (shadda-first == fatha-first)')


def test_mark_order_matters_when_asked():
    r = score_pair(NFC_WORD, LEGACY_WORD, canonical=False)
    assert r['der'] > 0.0, 'literal scoring should flag the reordering'
    assert r['wer'] > 0.0
    print('literal mode flags reordering: PASS (der=%.2f)' % r['der'])


def test_single_wrong_vowel():
    hyp = KAF + DAMMA + TEH + FATHA + BEH + FATHA
    r = score_pair(KATABA, hyp)
    assert r['der_errors'] == 1, r
    assert r['der_total'] == 3, r
    assert abs(r['der'] - 1 / 3) < 1e-9
    assert r['wer_errors'] == 1
    print('single wrong vowel: PASS (1/3 letters, 1/1 word)')


def test_missing_diacritics():
    r = score_pair(KATABA, KAF + TEH + BEH)
    assert r['der_errors'] == 3, r
    assert r['wer_errors'] == 1
    print('missing diacritics: PASS (3 letter errors, 1 word error)')


def test_letter_mismatch():
    r = score_pair(KATABA, 'xyz')
    assert r['wer_errors'] == 1
    assert r['details'][0]['letter_mismatch'] is True
    print('letter mismatch: PASS (flagged, not silently scored)')


def test_word_count_mismatch():
    r = score_pair('one two three', 'one two')
    assert r['word_count_mismatch'] is True
    assert r['wer_errors'] == 1
    print('word count mismatch: PASS (missing word counted)')


def test_canonicalize_is_shadda_first():
    got = canonicalize(NFC_WORD)
    assert got == LEGACY_WORD, 'expected shadda-first, got %r' % got
    assert canonicalize(got) == got, 'canonicalize must be idempotent'
    print('canonicalize: PASS (idempotent, shadda-first)')


def test_analyze_word():
    letters, marks = analyze_word(KATABA)
    assert len(letters) == 3, letters
    assert marks[0] == (FATHA,), marks
    bare_letters, bare_marks = analyze_word(KAF + TEH + BEH)
    assert bare_letters == letters
    assert bare_marks == [(), (), ()], bare_marks
    print('analyze_word: PASS')


def test_empty_inputs():
    r = score_pair('', '')
    assert r['der'] == 0.0 and r['wer'] == 0.0
    print('empty: PASS')


def test_leaderboard_shape():
    assert len(QUOTED_COLUMNS) == 4
    for row in QUOTED_ROWS + OUR_ROWS:
        assert len(row['values']) == 4, row
        assert row['model']
    pts = pareto_points()
    assert any(p['ours'] for p in pts), 'our models must appear in the scatter'
    assert all(p['params_m'] > 0 for p in pts)
    assert all(p['source'] in ('measured here', 'quoted') for p in pts)
    print('leaderboard: PASS (%d quoted, %d ours, %d scatter points)'
          % (len(QUOTED_ROWS), len(OUR_ROWS), len(pts)))


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
    print('ALL SCORING TESTS PASS')