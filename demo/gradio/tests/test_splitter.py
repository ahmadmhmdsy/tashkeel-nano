r'''Splitter round-trip and boundary tests.

Run:  python demo/gradio/tests/test_splitter.py
'''
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')

from splitter import reassemble, split_text  # noqa: E402

CASES = [
    '',
    '   ',
    '\n\n\n',
    'a',
    'hello world',
    'One sentence. Two sentences! Three?',
    'بسم الله الرحمن الرحيم',
    'كان الفيلسوف يرى أن العقل مرآة الحقيقة، وقد كتب الرسالة بيده.',
    'First para.\n\nSecond para with more text here.\n\nThird.',
    '  leading and trailing  ',
    '\t tabs\t and\nnewlines\n',
    'Mixed English and العربية and 12345 and emoji 🙂 here.',
    'a' * 1000,
    'word ' * 300,
    'short. ' * 200,
    'Line one\nLine two\nLine three\n',
    'No terminator at all just words going on and on and on',
    'نص طويل جدا بدون نقطة في النهاية وسيستمر هكذا لفترة طويلة جدا جدا',
    'Question؟ Exclamation! Semicolon؛ Comma، End.',
]

BUDGETS = [1, 2, 5, 13, 40, 100, 400, 10000]


def test_roundtrip():
    failures = []
    for text in CASES:
        for budget in BUDGETS:
            cores, gaps, report = split_text(text, budget)
            got = reassemble(cores, gaps)
            if got != text:
                failures.append((text[:40], budget, len(text), len(got)))
            if len(gaps) != len(cores) + 1:
                failures.append(('GAP-INVARIANT', text[:40], budget))
    assert not failures, 'round-trip failures: %r' % failures[:10]
    print('round-trip: PASS (%d texts x %d budgets = %d checks)'
          % (len(CASES), len(BUDGETS), len(CASES) * len(BUDGETS)))


def test_budget_respected():
    text = 'short. ' * 200
    cores, _, report = split_text(text, 40)
    over = [c for c in cores if len(c) > 40]
    assert not over, 'chunks exceed budget: %r' % [len(c) for c in over][:5]
    assert report.chunks == len(cores)
    assert report.longest_chunk <= 40
    print('budget: PASS (%d chunks, longest %d)' % (len(cores), report.longest_chunk))


def test_oversized_word_kept_whole():
    big = 'x' * 500
    text = 'before ' + big + ' after'
    cores, gaps, report = split_text(text, 20)
    assert reassemble(cores, gaps) == text
    assert big in cores, 'oversized word was not kept whole'
    assert report.oversized_words == [big]
    print('oversized: PASS (flagged, not truncated)')


def test_gaps_hold_whitespace():
    cores, gaps, _ = split_text('One. Two. Three.', 6)
    assert all(c == c.strip() for c in cores), 'cores must be stripped'
    assert ''.join(gaps).strip() == '' or all(not g.strip() for g in gaps)
    print('gaps: PASS (cores stripped, whitespace isolated)')


def test_report_shape():
    text = 'Para one. More here.\n\nPara two.'
    _, _, report = split_text(text, 400)
    assert report.chars == len(text)
    assert report.paragraphs == 2
    assert report.chunks >= 1
    assert any('chars ->' in line for line in report.lines())
    print('report: PASS (%d paragraph(s), %d sentence(s))'
          % (report.paragraphs, report.sentences))


def test_empty():
    cores, gaps, report = split_text('', 100)
    assert cores == [] and gaps == ['']
    assert reassemble(cores, gaps) == ''
    print('empty: PASS')


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
    print('ALL SPLITTER TESTS PASS')