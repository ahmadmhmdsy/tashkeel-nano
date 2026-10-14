r'''Functional tests for the app tab callbacks.

These call the underlying functions directly rather than clicking, because
Gradio validation errors surface only at interaction time and a clean boot
proves nothing about behaviour.

Run:  python demo/gradio/tests/test_app.py
'''
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')

import app  # noqa: E402
from samples import SAMPLES  # noqa: E402


class _Progress:
    '''Stand-in for gr.Progress inside a plain script.'''
    def __call__(self, *args, **kwargs):
        return None


P = _Progress()
ONE = ['nano1p1m']


def test_compare_renders_a_card():
    out = app.run_compare(SAMPLES[0]['text'], ONE, P)
    assert 'tk-card' in out, out[:300]
    assert 'tashkeel-nano-1p1m' in out
    assert '1.08 M params' in out
    assert 'NFC' in out
    print('compare tab: PASS')


def test_compare_empty_input():
    assert 'tk-empty' in app.run_compare('', ONE, P)
    assert 'tk-empty' in app.run_compare('   ', ONE, P)
    print('compare empty input: PASS')


def test_compare_unknown_key():
    out = app.run_compare('x', ['not-a-model'], P)
    assert 'tk-empty' in out and 'unknown' in out
    print('compare unknown key: PASS')


def test_bulk_tab():
    text = SAMPLES[-1]['text']
    report, out, files = app.run_bulk(text, ONE, 120, P)
    assert 'Splitting report' in report, report[:300]
    assert 'tk-card' in out
    assert files and len(files) == 3, 'expected txt + csv + json, got %r' % (files,)
    for path in files:
        assert Path(path).exists(), path
        assert Path(path).stat().st_size > 0, path
    txt = [f for f in files if f.endswith('.txt')][0]
    body = Path(txt).read_text(encoding='utf-8')
    assert body.count(chr(10)) == text.count(chr(10)), 'line structure must survive'
    assert len(body) > len(text), 'output should carry diacritics'
    print('bulk tab: PASS (%d files, %d lines preserved)'
          % (len(files), text.count(chr(10))))


def test_bulk_empty():
    report, out, files = app.run_bulk('', ONE, 400, P)
    assert 'tk-empty' in out and files is None
    print('bulk empty input: PASS')


def test_bulk_oversized_word_reported():
    report, _out, _files = app.run_bulk('x' * 400, ONE, 50, P)
    assert 'WARNING' in report, report
    print('bulk oversized word warning: PASS')


def test_score_tab():
    ref = '\u0643\u064e\u062a\u064e\u0628\u064e'
    table, details = app.run_score(ref, ONE, True, P)
    assert 'DER' in table and 'WER' in table, table[:300]
    assert 'Order-insensitive' in table
    assert 'tk-card' in details
    lit, _ = app.run_score(ref, ONE, False, P)
    assert 'LITERAL' in lit
    print('score tab: PASS')


def test_score_empty():
    table, details = app.run_score('', ONE, True, P)
    assert 'tk-empty' in table
    print('score empty input: PASS')


def test_leaderboard():
    md = app.leaderboard_md()
    assert 'quoted' in md and 'measured here' in md
    assert 'tashkeel-nano-1p1m (ours)' in md
    assert 'Claude-3.7-Sonnet' in md
    pts = app.pareto_points() if hasattr(app, 'pareto_points') else None
    print('leaderboard: PASS')


def test_builds_without_error():
    demo = app.build()
    assert demo is not None
    print('ui build: PASS')


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
    print('ALL APP TESTS PASS')