'''Order-insensitive diacritization scoring, plus the published leaderboard.

Why order-insensitive: SadeedDiac-25 gold writes the shadda *before* the vowel
(U+0651 U+064E), which is not NFC. A naive literal comparison against NFC
output (U+064E U+0651) marks every geminate letter wrong. E-104 measured the
size of that artefact: 8.78 vs 3.78 DER on the same predictions.

Every metric here canonicalises the mark order of each letter on BOTH sides
before comparing, so a difference in metric means a difference in reading,
not a difference in spelling convention.

Definitions (this module's own, stated exactly so nobody has to guess):

  DER = letters whose canonical mark set differs from the reference
        / reference letters that carry at least one mark
  WER = words whose canonical diacritized form differs from the reference
        / reference words

These are NOT a re-implementation of the Sadeed benchmark's evaluator; the
published leaderboard numbers come from that evaluator and are labelled
``quoted`` wherever they appear.
'''
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Tuple

DIACRITIC_CHARS = set(chr(c) for c in list(range(0x064B, 0x0656)) + [0x0670])

# Canonical mark order within one letter: shadda first, then the short vowels,
# then the tanwin forms, then sukun and the superscript alef.
_CANON_ORDER = {
    chr(0x0651): 0,
    chr(0x064E): 1,
    chr(0x064F): 2,
    chr(0x0650): 3,
    chr(0x064B): 4,
    chr(0x064C): 5,
    chr(0x064D): 6,
    chr(0x0652): 7,
    chr(0x0670): 8,
    chr(0x0653): 9,
    chr(0x0654): 10,
    chr(0x0655): 11,
}

_LETTER_RE = re.compile('[\u0621-\u064A\u0671-\u06D3]')
_WORD_RE = re.compile(r'\S+')


def _sort_marks(marks) -> Tuple[str, ...]:
    return tuple(sorted(marks, key=lambda c: _CANON_ORDER.get(c, 99)))


def analyze_word(word: str, canonical: bool = True):
    '''Return (letters, marks_per_letter) for one word.

    ``marks_per_letter[i]`` is the tuple of diacritics attached to
    ``letters[i]``. With ``canonical=True`` the marks are sorted into a fixed
    order so that shadda-first and fatha-first spellings compare equal.
    '''
    # Deliberately no NFC here: canonical mode sorts the marks itself, and
    # literal mode must see the marks in the order they were written.
    letters: List[str] = []
    marks: List[List[str]] = []
    for ch in word:
        if ch in DIACRITIC_CHARS:
            if letters:
                marks[-1].append(ch)
        elif _LETTER_RE.match(ch):
            letters.append(ch)
            marks.append([])
    if canonical:
        return letters, [_sort_marks(m) for m in marks]
    return letters, [tuple(m) for m in marks]


def strip_marks(text: str) -> str:
    '''Bare text: drop every combining mark, then NFC-normalise.'''
    return unicodedata.normalize(
        'NFC', ''.join(ch for ch in text if ch not in DIACRITIC_CHARS)
    )


def canonicalize(text: str) -> str:
    '''Rewrite every letter's marks into canonical order.'''
    out: List[str] = []
    pending: List[str] = []
    for ch in unicodedata.normalize('NFC', text):
        if ch in DIACRITIC_CHARS:
            pending.append(ch)
            continue
        out.extend(_sort_marks(pending))
        pending = []
        out.append(ch)
    out.extend(_sort_marks(pending))
    return ''.join(out)


def _diff_word(ref_word: str, hyp_word: str, canonical: bool = True) -> Dict:
    r_letters, r_marks = analyze_word(ref_word, canonical)
    h_letters, h_marks = analyze_word(hyp_word, canonical)
    ref_marked = sum(1 for m in r_marks if m)
    if r_letters != h_letters:
        return {
            'letter_mismatch': True,
            'errors': max(len(r_letters), 1),
            'ref_marked': ref_marked,
            'ok': False,
        }
    errors = sum(1 for a, b in zip(r_marks, h_marks) if a != b)
    return {
        'letter_mismatch': False,
        'errors': errors,
        'ref_marked': ref_marked,
        'ok': errors == 0,
    }


def score_pair(reference: str, hypothesis: str, canonical: bool = True) -> Dict:
    '''Score one reference/hypothesis pair. Lower is better for der and wer.'''
    ref_words = _WORD_RE.findall(reference)
    hyp_words = _WORD_RE.findall(hypothesis)
    n = min(len(ref_words), len(hyp_words))

    der_errors = 0
    der_total = 0
    wer_errors = 0
    wer_total = 0
    details: List[Dict] = []

    for i in range(n):
        d = _diff_word(ref_words[i], hyp_words[i], canonical)
        der_errors += d['errors']
        der_total += d['ref_marked']
        wer_total += 1
        if not d['ok']:
            wer_errors += 1
        details.append({
            'ref': ref_words[i],
            'hyp': hyp_words[i],
            'ok': d['ok'],
            'letter_mismatch': d['letter_mismatch'],
        })

    extra = abs(len(ref_words) - len(hyp_words))
    if extra:
        wer_errors += extra
        wer_total += extra
        der_errors += extra
        der_total += extra

    return {
        'der': (der_errors / der_total) if der_total else 0.0,
        'wer': (wer_errors / wer_total) if wer_total else 0.0,
        'der_errors': der_errors,
        'der_total': der_total,
        'wer_errors': wer_errors,
        'words': wer_total,
        'word_count_mismatch': len(ref_words) != len(hyp_words),
        'details': details,
    }


# ------------------------------------------------------------------ leaderboard

# Quoted from the published SadeedDiac-25 benchmark table (their evaluator,
# letter-level, with their own CE correction). Never presented as measured here.
QUOTED_COLUMNS = ['DER (CE)', 'WER (CE)', 'DER (w/o CE)', 'WER (w/o CE)']

QUOTED_ROWS: List[Dict] = [
    {'model': 'Claude-3.7-Sonnet', 'params_m': None, 'values': [1.3941, 4.6718, 0.7693, 2.3098]},
    {'model': 'Gemini-Flash-2.0', 'params_m': None, 'values': [3.1926, 7.9942, 2.3783, 5.5044]},
    {'model': 'GPT-4', 'params_m': None, 'values': [3.8645, 5.2719, 3.8645, 10.9274]},
    {'model': 'Sadeed (Kuwain 1.5B)', 'params_m': 1500.0, 'values': [7.2915, 13.7425, 5.2625, 9.9245]},
    {'model': 'Aya-23-8B', 'params_m': 8000.0, 'values': [25.6274, 47.4908, 19.7584, 40.2478]},
    {'model': 'ALLaM-7B', 'params_m': 7000.0, 'values': [50.3586, 70.3369, 39.4100, 67.0920]},
    {'model': 'Yehia-7B', 'params_m': 7000.0, 'values': [50.8801, 70.2323, 39.7677, 67.1520]},
    {'model': 'SILMA-9B', 'params_m': 9000.0, 'values': [78.6567, 99.7367, 60.7106, 99.5586]},
    {'model': 'Jais-13B', 'params_m': 13000.0, 'values': [78.6820, 99.7541, 60.7271, 99.5702]},
    {'model': 'Gemma-2-9B', 'params_m': 9000.0, 'values': [78.8560, 99.7928, 60.9188, 99.5895]},
]

# Measured locally with the benchmark's own evaluator on all 1200 SadeedDiac-25
# rows, mark order normalised. Artifact: scratch/e104_sadeed_results.json
OUR_ROWS: List[Dict] = [
    {'model': 'tashkeel-nano-1p1m (ours)', 'params_m': 1.08, 'values': [3.78, 12.73, 2.25, 7.41]},
    {'model': 'tashkeel-nano-4p5m (ours)', 'params_m': 4.49, 'values': [9.91, 31.03, 5.01, 15.73]},
]

OURS_AS_EMITTED = {
    'tashkeel-nano-1p1m (ours)': [8.78, 31.14, 6.19, 22.73],
    'tashkeel-nano-4p5m (ours)': [9.91, 31.03, 5.01, 15.73],
}


def pareto_points() -> List[Dict]:
    '''Rows with a known parameter count, for the cost-vs-quality scatter.'''
    pts = []
    for row in QUOTED_ROWS + OUR_ROWS:
        if row['params_m']:
            pts.append({
                'model': row['model'],
                'params_m': row['params_m'],
                'der': row['values'][0],
                'ours': row['model'].endswith('(ours)'),
                'source': 'measured here' if row['model'].endswith('(ours)') else 'quoted',
            })
    return pts