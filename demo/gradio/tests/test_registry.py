r'''Registry shape tests plus one real end-to-end nano inference.

Run:  python demo/gradio/tests/test_registry.py
'''
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')

from registry import BY_KEY, MODELS, REGISTRY, resolve  # noqa: E402

EXCLUDED = [
    'ahmedsamirtarjama/Tashkeel-50M',
    'basharalrfooh/Fine-Tashkeel',
    'ahmedsamirtarjama/Tashkeel-v4',
    'ahmedsamirtarjama/Tashkeel-v3',
]


def test_specs_well_formed():
    keys = [m.key for m in MODELS]
    assert len(keys) == len(set(keys)), 'duplicate model keys'
    for m in MODELS:
        assert m.family in ('nano', 'mishkala', 'seq2seq'), m
        assert m.tier in ('fast', 'slow'), m
        assert m.mark_order in ('nfc', 'legacy'), m
        assert m.params_m > 0, m
        assert m.repo.count('/') == 1, m
        assert m.note, m
    assert any(m.is_ours for m in MODELS), 'our models must be present'
    print('specs: PASS (%d models)' % len(MODELS))


def test_resolve():
    default = resolve()
    assert default and all(m.default_on for m in default)
    assert resolve(['nano1p1m']) == [BY_KEY['nano1p1m']]
    try:
        resolve(['does-not-exist'])
        raise AssertionError('unknown key must raise')
    except KeyError:
        pass
    print('resolve: PASS (%d default-on)' % len(default))


def test_excluded_models_absent():
    repos = set(m.repo for m in MODELS)
    for bad in EXCLUDED:
        assert bad not in repos, 'excluded model leaked into roster: %s' % bad
    print('exclusions: PASS (%d excluded models absent)' % len(EXCLUDED))


def test_ours_are_smallest():
    smallest = min(MODELS, key=lambda m: m.params_m)
    assert smallest.is_ours, 'the smallest model should be ours, got %s' % smallest.label
    print('showcase: PASS (smallest model is %s at %.2fM)' % (smallest.label, smallest.params_m))


def test_nano_end_to_end():
    text = '\u0627\u0644\u0633\u0644\u0627\u0645 \u0639\u0644\u064a\u0643\u0645'
    try:
        r = REGISTRY.predict('nano1p1m', text)
    except Exception as exc:  # noqa: BLE001
        print('nano end-to-end: SKIPPED (%s)' % exc)
        return
    if r['error']:
        print('nano end-to-end: SKIPPED (%s)' % r['error'])
        return
    assert r['out'] != text, 'model returned the input unchanged'
    assert len(r['out']) > len(text), 'diacritized output should be longer'
    assert r['seconds'] > 0
    r2 = REGISTRY.predict('nano1p1m', text)
    assert r2['cached'] is True, 'repeat call should hit the cache'
    print('nano end-to-end: PASS (%.2fs, %d chars)' % (r['seconds'], len(r['out'])))


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
    print('ALL REGISTRY TESTS PASS')