'''Tashkeel Compare - a public side-by-side Arabic diacritization demo.

Four tabs:
  Compare      - run our models and published third-party models on the same text
  Bulk         - split a large paste/upload, diacritize all of it, download results
  Score        - score every model against a reference you supply
  Leaderboard  - the published SadeedDiac-25 table plus a cost-vs-quality scatter

Honesty rules enforced here:
  * only downloadable, published models appear;
  * quoted benchmark numbers are labelled 'quoted', never 'measured here';
  * scoring is order-insensitive by default and says so on screen.
'''
from __future__ import annotations

import csv
import html
import io
import json
import os
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

from registry import MODELS, REGISTRY, resolve
from samples import NAMES, SAMPLES, by_name
from scoring import (
    OUR_ROWS,
    QUOTED_COLUMNS,
    QUOTED_ROWS,
    pareto_points,
    score_pair,
    strip_marks,
)
from splitter import reassemble, split_text

CHOICES = [(m.label, m.key) for m in MODELS]
DEFAULT_KEYS = [m.key for m in MODELS if m.default_on]
SPEC_BY_KEY = {m.key: m for m in MODELS}

CSS = Path(__file__).with_name('theme.css').read_text(encoding='utf-8')


# ------------------------------------------------------------------ rendering

def _esc(text: str) -> str:
    return html.escape(text or '').replace(chr(10), '<br>')


def _stats(spec, seconds: float, cached: bool, chars: int) -> str:
    rate = (chars / seconds) if seconds > 0 and chars else 0.0
    bits = ['%.2f M params' % spec.params_m, '~%.1f MB' % spec.size_mb]
    if cached:
        bits.append('cached')
    else:
        bits.append('%.2f s' % seconds)
        if rate:
            bits.append('%.0f chars/s' % rate)
    return ' &middot; '.join(bits)


def _badges(spec) -> str:
    out = []
    if spec.is_ours:
        out.append('<span class="tk-b ours">ours</span>')
    else:
        out.append('<span class="tk-b third">third-party</span>')
    out.append('<span class="tk-b order">%s</span>' % spec.mark_order.upper())
    if spec.tier == 'slow':
        out.append('<span class="tk-b slow">slow on CPU</span>')
    return ''.join(out)


def _render_cards(results) -> str:
    if not results:
        return '<div class="tk-empty">Select at least one model.</div>'
    cards = []
    for r in results:
        spec = SPEC_BY_KEY[r['key']]
        err = r.get('error')
        body = _esc(r['out'])
        if err:
            body = ('<div class="tk-err">%s failed on this input:<br>%s</div>'
                    % (html.escape(spec.label), html.escape(err)))
        cards.append(
            '<div class="tk-card%s">'
            '<div class="tk-head"><span class="tk-name">%s</span>%s</div>'
            '<div class="tk-stats">%s</div>'
            '<div class="tk-out" dir="rtl">%s</div>'
            '<div class="tk-note">%s</div>'
            '</div>'
            % (' ours' if spec.is_ours else '', html.escape(spec.label),
               _badges(spec),
               _stats(spec, r['seconds'], r['cached'], len(r['out'] or '')),
               body, html.escape(spec.note))
        )
    return '<div class="tk-grid">%s</div>' % ''.join(cards)


def _guard(keys):
    try:
        return resolve(list(keys) if keys else None), None
    except KeyError as exc:
        return [], str(exc)


# -------------------------------------------------------------- tab: compare

def run_compare(text, keys, progress=gr.Progress()):
    specs, err = _guard(keys)
    if err:
        return '<div class="tk-empty">%s</div>' % html.escape(err)
    if not text or not text.strip():
        return '<div class="tk-empty">Paste some Arabic text above to compare.</div>'
    results = []
    for i, spec in enumerate(specs):
        progress((i, len(specs)), desc='Running %s' % spec.label)
        results.append(REGISTRY.predict(spec.key, text))
    return _render_cards(results)


# ----------------------------------------------------------------- tab: bulk

def _write_temp(name: str, content: str) -> str:
    path = Path(tempfile.mkdtemp(prefix='tashkeel_')) / name
    path.write_text(content, encoding='utf-8')
    return str(path)


def run_bulk(text, keys, max_chars, progress=gr.Progress()):
    specs, err = _guard(keys)
    empty = '<div class="tk-empty">%s</div>'
    if err:
        return '', empty % html.escape(err), None
    if not text or not text.strip():
        return '', empty % 'Paste or upload text, then press Split and diacritize.', None

    cores, gaps, report = split_text(text, int(max_chars))
    report_md = '### Splitting report\n' + '\n'.join(
        '- ' + line for line in report.lines()
    )

    per_model = {}
    failures = {}
    total = max(1, len(cores) * len(specs))
    step = 0
    for spec in specs:
        outs = []
        for core in cores:
            progress((step, total), desc='%s: chunk %d/%d'
                     % (spec.label, len(outs) + 1, len(cores)))
            step += 1
            r = REGISTRY.predict(spec.key, core)
            if r.get('error'):
                failures[spec.label] = r['error']
            outs.append(r['out'])
        per_model[spec.key] = outs

    if failures:
        report_md += '\n\n### Failures\n' + '\n'.join(
            '- %s: %s' % (html.escape(k), html.escape(v))
            for k, v in failures.items()
        )

    order = [s.key for s in specs]
    results = [
        {
            'key': spec.key,
            'label': spec.label,
            'out': reassemble(per_model[spec.key], gaps),
            'seconds': 0.0,
            'cached': False,
            'error': failures.get(spec.label),
        }
        for spec in specs
    ]

    files = []
    for spec in specs:
        files.append(_write_temp('%s.txt' % spec.key,
                                 results[order.index(spec.key)]['out']))

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(['chunk', 'source'] + [s.label for s in specs])
    for i, core in enumerate(cores):
        writer.writerow([i + 1, core] + [per_model[s.key][i] for s in specs])
    files.append(_write_temp('comparison.csv', buf.getvalue()))

    payload = {
        'splitting_report': report.lines(),
        'chunk_budget': int(max_chars),
        'chunks': len(cores),
        'models': [
            {
                'key': s.key,
                'label': s.label,
                'repo': s.repo,
                'params_m': s.params_m,
                'mark_order': s.mark_order,
                'text': results[order.index(s.key)]['out'],
            }
            for s in specs
        ],
    }
    files.append(_write_temp('comparison.json',
                             json.dumps(payload, ensure_ascii=False, indent=2)))

    return report_md, _render_cards(results), files


# ---------------------------------------------------------------- tab: score

def run_score(reference, keys, canonical, progress=gr.Progress()):
    specs, err = _guard(keys)
    if err:
        return '<div class="tk-empty">%s</div>' % html.escape(err), ''
    if not reference or not reference.strip():
        return '<div class="tk-empty">Paste a diacritized reference above.</div>', ''

    bare = strip_marks(reference)
    if not bare.strip():
        return '<div class="tk-empty">The reference has no base letters.</div>', ''

    rows = []
    for i, spec in enumerate(specs):
        progress((i, len(specs)), desc='Scoring %s' % spec.label)
        r = REGISTRY.predict(spec.key, bare)
        s = score_pair(reference, r['out'], canonical=canonical)
        rows.append((spec, s, r))

    table = ['| Model | Params | DER | WER | letter errors | words |',
             '|---|---|---|---|---|---|']
    for spec, s, _r in rows:
        table.append('| %s | %.2f M | %.2f%% | %.2f%% | %d | %d |'
                     % (spec.label, spec.params_m, s['der'] * 100,
                        s['wer'] * 100, s['der_errors'], s['words']))
    note = ('Order-insensitive: mark order is canonicalised on both sides.'
            if canonical else
            'LITERAL mode: shadda-first vs fatha-first now counts as an error.')

    details = []
    for spec, s, r in rows:
        diff = ''.join(
            '<span class="tk-w%s" dir="rtl">%s</span>'
            % ('' if d['ok'] else ' bad', _esc(d['ref']))
            for d in s['details'][:200]
        )
        details.append(
            '<div class="tk-card"><div class="tk-head">'
            '<span class="tk-name">%s</span>%s</div>'
            '<div class="tk-stats">wrong words highlighted</div>'
            '<div class="tk-out" dir="rtl">%s</div></div>'
            % (html.escape(spec.label), _badges(spec), diff)
        )

    return ('\n'.join(table) + '\n\n' + note,
            '<div class="tk-grid">%s</div>' % ''.join(details))


# ----------------------------------------------------------- tab: leaderboard

def _table(headers, rows) -> str:
    out = ['| ' + ' | '.join(headers) + ' |', '|' + '---|' * len(headers)]
    for row in rows:
        out.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(out)


def leaderboard_md() -> str:
    ours = [[row['model'], '%.2f M' % row['params_m']]
            + ['%.2f' % v for v in row['values']] + ['measured here']
            for row in OUR_ROWS]
    quoted = [[row['model'],
               ('%.0f M' % row['params_m']) if row['params_m'] else 'n/a']
              + ['%.2f' % v for v in row['values']] + ['quoted']
              for row in sorted(QUOTED_ROWS, key=lambda r: r['values'][0])]
    return _table(['Model', 'Params'] + QUOTED_COLUMNS + ['Source'], ours + quoted)


# ------------------------------------------------------------------ UI build

def build():
    with gr.Blocks(title='Tashkeel Compare') as demo:
        gr.Markdown(
            '# Tashkeel Compare\n'
            'Arabic diacritization, side by side. Our **1.08 M parameter** model '
            'runs next to published third-party models on the same text, with '
            'parameters, size and latency shown for each.\n\n'
            '*Every model here is a downloadable checkpoint. Nothing unpublished '
            'is compared. Benchmark numbers marked **quoted** come from the '
            'published SadeedDiac-25 table; numbers marked **measured here** were '
            'produced locally with that benchmark own evaluator.*'
        )

        with gr.Tab('Compare'):
            with gr.Row():
                with gr.Column(scale=1):
                    c_text = gr.Textbox(label='Arabic text', lines=8,
                                        value=SAMPLES[0]['text'])
                    c_sample = gr.Dropdown(choices=NAMES, value=NAMES[0],
                                           label='Load a sample')
                    c_keys = gr.CheckboxGroup(choices=CHOICES,
                                              value=DEFAULT_KEYS, label='Models')
                    c_btn = gr.Button('Diacritize', variant='primary')
                with gr.Column(scale=2):
                    c_out = gr.HTML('<div class="tk-empty">Results appear here.</div>')
            c_sample.change(lambda n: by_name(n), c_sample, c_text)
            c_btn.click(run_compare, [c_text, c_keys], c_out)

        with gr.Tab('Bulk'):
            gr.Markdown(
                'Paste or upload a large document. It is split on paragraph and '
                'sentence boundaries, every chunk is diacritized, and the result '
                'is reassembled with your original line structure intact.'
            )
            with gr.Row():
                with gr.Column(scale=1):
                    b_text = gr.Textbox(label='Text', lines=10,
                                        value=SAMPLES[-1]['text'])
                    b_file = gr.File(label='or upload .txt / .csv / .md',
                                     file_types=['.txt', '.csv', '.md'],
                                     type='filepath')
                    b_sample = gr.Dropdown(choices=NAMES, value=NAMES[-1],
                                           label='Load a sample')
                    b_keys = gr.CheckboxGroup(choices=CHOICES,
                                              value=DEFAULT_KEYS, label='Models')
                    b_budget = gr.Slider(80, 2000, value=400, step=20,
                                         label='Chunk budget (characters)')
                    b_btn = gr.Button('Split and diacritize', variant='primary')
                with gr.Column(scale=2):
                    b_report = gr.Markdown('')
                    b_out = gr.HTML('<div class="tk-empty">Results appear here.</div>')
                    b_files = gr.File(label='Download results', file_count='multiple')

            def _load(path):
                if not path:
                    return gr.update()
                try:
                    return Path(path).read_text(encoding='utf-8', errors='replace')
                except Exception:
                    return gr.update()

            b_file.change(_load, b_file, b_text)
            b_sample.change(lambda n: by_name(n), b_sample, b_text)
            b_btn.click(run_bulk, [b_text, b_keys, b_budget],
                        [b_report, b_out, b_files])

        with gr.Tab('Score'):
            gr.Markdown(
                'Paste a **diacritized** reference. Each model diacritizes the '
                'bare version and is scored against it.\n\n'
                '**Order-insensitive by default.** SadeedDiac-25 gold writes the '
                'shadda before the vowel, which is not NFC; comparing that '
                'literally against NFC output marks every geminate wrong. '
                'Canonicalisation removes that artefact. Turn it off to see the '
                'effect it has.'
            )
            with gr.Row():
                with gr.Column(scale=1):
                    s_ref = gr.Textbox(label='Diacritized reference', lines=8)
                    s_keys = gr.CheckboxGroup(choices=CHOICES,
                                              value=DEFAULT_KEYS, label='Models')
                    s_canon = gr.Checkbox(
                        value=True, label='Canonicalise mark order (recommended)')
                    s_btn = gr.Button('Score', variant='primary')
                with gr.Column(scale=2):
                    s_table = gr.Markdown('')
                    s_out = gr.HTML('')
            s_btn.click(run_score, [s_ref, s_keys, s_canon], [s_table, s_out])

        with gr.Tab('Leaderboard'):
            gr.Markdown(
                '### SadeedDiac-25\n'
                'Columns are the benchmark own: DER and WER, with and without '
                'their CE correction. **quoted** rows are the published table; '
                '**measured here** rows were produced locally by that benchmark '
                'evaluator with mark order normalised.'
            )
            gr.Markdown(leaderboard_md())
            gr.Markdown('### Cost vs quality\n'
                        'Lower is better. Log scale on parameters.')
            gr.ScatterPlot(value=pd.DataFrame(pareto_points()),
                           x='params_m', y='der', color='source',
                           x_title='parameters (millions, log scale)',
                           y_title='DER (CE)', height=420)
            gr.Markdown(
                '### What this demo does not prove\n'
                '- The 8 B-13 B systems in the quoted table cannot run on free '
                'CPU; their numbers are quoted, not re-measured.\n'
                '- SadeedDiac-25 gold is not NFC, and the benchmark does not '
                'state whether its own leaderboard systems were '
                'order-normalised, so the ranking is approximate.\n'
                '- wikinews2024 is excluded from the registered tracker because '
                'it is contaminated.\n'
                '- Third-party models run code published in their own HF repos.'
            )
    return demo


if __name__ == '__main__':
    build().queue(max_size=8, default_concurrency_limit=1).launch(
        css=CSS,
        theme=gr.themes.Soft(),
        server_name=os.environ.get('GRADIO_SERVER_NAME', '127.0.0.1'),
        server_port=int(os.environ.get('GRADIO_SERVER_PORT', '7862')),
    )
