'''Text splitting and lossless reassembly for the Tashkeel Compare Space.

Invariant: for any input ``text`` and any ``max_chars``,

    reassemble(*split_text(text, max_chars)[:2]) == text

The splitter never drops, duplicates, or reorders a character. Whitespace that
falls between chunks is carried as a *gap* rather than handed to a model, so a
model that normalises or strips whitespace cannot corrupt the user's layout.
'''
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple

# A sentence ends after a non-space token whose last character is one of these.
TERMINATORS = '.!?؟؛\n'

_TOKEN_RE = re.compile(r'\s+|\S+')
_PARA_RE = re.compile(r'\n\s*\n')


@dataclass
class SplitReport:
    chars: int = 0
    paragraphs: int = 0
    sentences: int = 0
    chunks: int = 0
    longest_chunk: int = 0
    max_chars: int = 0
    hard_split_sentences: int = 0
    oversized_words: List[str] = field(default_factory=list)

    def lines(self) -> List[str]:
        out = [
            '%d chars -> %d chunk(s), budget %d chars'
            % (self.chars, self.chunks, self.max_chars),
            '%d paragraph(s), %d sentence(s), longest chunk %d chars'
            % (self.paragraphs, self.sentences, self.longest_chunk),
        ]
        if self.hard_split_sentences:
            out.append(
                '%d sentence(s) exceeded the budget and were split at word '
                'boundaries' % self.hard_split_sentences
            )
        if self.oversized_words:
            shown = ', '.join(self.oversized_words[:5])
            out.append(
                'WARNING: %d word(s) longer than the budget were kept whole '
                '(never truncated): %s' % (len(self.oversized_words), shown)
            )
        return out


def _sentences(text: str) -> List[str]:
    '''Split into sentence-ish spans, preserving every character.'''
    out: List[str] = []
    buf: List[str] = []
    for part in _TOKEN_RE.findall(text):
        buf.append(part)
        if not part.isspace() and part[-1] in TERMINATORS:
            out.append(''.join(buf))
            buf = []
    if buf:
        out.append(''.join(buf))
    return out


def _hard_split(sentence: str, max_chars: int, oversized: List[str]) -> List[str]:
    '''Split one oversized sentence at word boundaries.

    ''".join(result) == sentence'' always holds. A single word longer than the
    budget is emitted whole and recorded in ``oversized`` - never truncated.
    '''
    out: List[str] = []
    cur = ''
    for word in _TOKEN_RE.findall(sentence):
        if word.isspace():
            cur += word
            continue
        if len(word) > max_chars:
            if out:
                out[-1] += cur
            elif cur:
                out.append(cur)
            cur = ''
            oversized.append(word)
            out.append(word)
            continue
        if len(cur) + len(word) > max_chars and cur.strip():
            out.append(cur)
            cur = word
        else:
            cur += word
    if cur.strip():
        out.append(cur)
    elif cur:
        if out:
            out[-1] += cur
        else:
            out.append(cur)
    return out


def _to_cores_gaps(chunks: List[str]) -> Tuple[List[str], List[str]]:
    '''Convert chunk strings into (cores, gaps).

    Convention: ``gaps[0] + cores[0] + gaps[1] + cores[1] + ... + gaps[n]``
    reproduces the input exactly, so ``len(gaps) == len(cores) + 1`` and
    ``gaps[i]`` is the whitespace *preceding* ``cores[i]``. Keeping every
    whitespace run out of the cores means a model that strips or collapses
    whitespace cannot corrupt the user's layout.
    '''
    cores: List[str] = []
    gaps: List[str] = []
    pending = ''
    for chunk in chunks:
        if not chunk.strip():
            pending += chunk
            continue
        lead = chunk[: len(chunk) - len(chunk.lstrip())]
        trail = chunk[len(chunk.rstrip()) :]
        body = chunk[len(lead) : len(chunk) - len(trail)]
        pending += lead
        gaps.append(pending)
        cores.append(body)
        pending = trail
    gaps.append(pending)
    return cores, gaps


def split_text(text: str, max_chars: int = 400) -> Tuple[List[str], List[str], SplitReport]:
    '''Split ``text`` into model-sized cores plus the gaps between them.'''
    report = SplitReport(chars=len(text), max_chars=int(max_chars))
    if not text:
        return [], [''], report
    report.paragraphs = len([p for p in _PARA_RE.split(text) if p.strip()])
    sentences = _sentences(text)
    report.sentences = len(sentences)

    chunks: List[str] = []
    cur = ''
    for sentence in sentences:
        if len(sentence) > max_chars:
            if cur.strip():
                chunks.append(cur)
                cur = ''
            pieces = _hard_split(sentence, max_chars, report.oversized_words)
            report.hard_split_sentences += 1
            if pieces:
                chunks.extend(pieces[:-1])
                cur = pieces[-1]
            continue
        if len(cur) + len(sentence) > max_chars and cur.strip():
            chunks.append(cur)
            cur = sentence
        else:
            cur += sentence
    if cur.strip():
        chunks.append(cur)
    elif cur and chunks:
        chunks[-1] += cur
    elif cur:
        chunks.append(cur)

    cores, gaps = _to_cores_gaps(chunks)
    assert len(gaps) == len(cores) + 1, 'gap invariant violated'
    report.chunks = len(cores)
    report.longest_chunk = max((len(c) for c in cores), default=0)
    return cores, gaps, report


def reassemble(cores: List[str], gaps: List[str]) -> str:
    '''Inverse of ``split_text``: rebuild the exact original string.

    Walks ``gaps[0] + cores[0] + gaps[1] + ... + gaps[n]`` so that a
    whitespace-only input, which yields zero cores, still round-trips.
    '''
    parts = [gaps[0] if gaps else '']
    for i, core in enumerate(cores):
        parts.append(core)
        parts.append(gaps[i + 1] if i + 1 < len(gaps) else '')
    return ''.join(parts)


def diacritize_chunks(cores: List[str], predict) -> List[str]:
    '''Apply ``predict`` to each core, returning one output per core.

    A failure on one core yields that core unchanged rather than aborting the
    whole document, and the caller can surface the failure count.
    '''
    out: List[str] = []
    for core in cores:
        try:
            out.append(predict(core))
        except Exception:
            out.append(core)
    return out
