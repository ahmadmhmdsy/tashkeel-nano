/*
 * Splitting and order-insensitive scoring - a port of space/splitter.py and
 * space/scoring.py.
 *
 * Splitter invariant (asserted by selfTest): for any text and budget,
 *   reassemble(splitText(text, budget).cores, ...gaps) === text
 * Scoring canonicalises mark order on both sides, so shadda-first and
 * fatha-first spellings of the same reading compare EQUAL. That is the E-104
 * lesson: comparing raw order scored the same predictions 8.78 vs 3.78 DER.
 */

// ------------------------------------------------------------------ splitting

export const TERMINATORS = ".!?\u061f\u061b\n";
const TOKEN_RE = /\s+|\S+/g;
const PARA_RE = /\n\s*\n/;

export function sentences(text) {
  const out = [];
  let buf = "";
  for (const part of text.match(TOKEN_RE) || []) {
    buf += part;
    if (!/^\s+$/.test(part) && TERMINATORS.indexOf(part[part.length - 1]) !== -1) {
      out.push(buf);
      buf = "";
    }
  }
  if (buf) out.push(buf);
  return out;
}

function hardSplit(sentence, maxChars, oversized) {
  const out = [];
  let cur = "";
  for (const word of sentence.match(TOKEN_RE) || []) {
    if (/^\s+$/.test(word)) { cur += word; continue; }
    if (word.length > maxChars) {
      if (out.length) out[out.length - 1] += cur;
      else if (cur) out.push(cur);
      cur = "";
      oversized.push(word);
      out.push(word);
      continue;
    }
    if (cur.length + word.length > maxChars && cur.trim()) {
      out.push(cur);
      cur = word;
    } else {
      cur += word;
    }
  }
  if (cur.trim()) out.push(cur);
  else if (cur) { if (out.length) out[out.length - 1] += cur; else out.push(cur); }
  return out;
}

function toCoresGaps(chunks) {
  const cores = [], gaps = [];
  let pending = "";
  for (const chunk of chunks) {
    if (!chunk.trim()) { pending += chunk; continue; }
    const lead = chunk.slice(0, chunk.length - chunk.trimStart().length);
    const trail = chunk.slice(chunk.trimEnd().length);
    const body = chunk.slice(lead.length, chunk.length - trail.length);
    pending += lead;
    gaps.push(pending);
    cores.push(body);
    pending = trail;
  }
  gaps.push(pending);
  return { cores, gaps };
}

export function splitText(text, maxChars) {
  maxChars = Math.max(1, Math.floor(maxChars));
  const report = {
    chars: text.length, paragraphs: 0, sentences: 0, chunks: 0,
    longest_chunk: 0, max_chars: maxChars, hard_split_sentences: 0,
    oversized_words: [],
  };
  if (!text) return { cores: [], gaps: [""], report };
  report.paragraphs = text.split(PARA_RE).filter((p) => p.trim()).length;
  const sents = sentences(text);
  report.sentences = sents.length;

  const chunks = [];
  let cur = "";
  for (const sentence of sents) {
    if (sentence.length > maxChars) {
      if (cur.trim()) { chunks.push(cur); cur = ""; }
      const pieces = hardSplit(sentence, maxChars, report.oversized_words);
      report.hard_split_sentences += 1;
      if (pieces.length) {
        for (const p of pieces.slice(0, -1)) chunks.push(p);
        cur = pieces[pieces.length - 1];
      }
      continue;
    }
    if (cur.length + sentence.length > maxChars && cur.trim()) {
      chunks.push(cur);
      cur = sentence;
    } else {
      cur += sentence;
    }
  }
  if (cur.trim()) chunks.push(cur);
  else if (cur && chunks.length) chunks[chunks.length - 1] += cur;
  else if (cur) chunks.push(cur);

  const { cores, gaps } = toCoresGaps(chunks);
  if (gaps.length !== cores.length + 1) throw new Error("gap invariant violated");
  report.chunks = cores.length;
  report.longest_chunk = cores.reduce((m, c) => Math.max(m, c.length), 0);
  return { cores, gaps, report };
}

export function reassemble(cores, gaps) {
  const parts = [gaps && gaps.length ? gaps[0] : ""];
  for (let i = 0; i < cores.length; i++) {
    parts.push(cores[i]);
    parts.push(gaps && i + 1 < gaps.length ? gaps[i + 1] : "");
  }
  return parts.join("");
}

/*
 * The splitting report is prose, so it is translated here rather than in the
 * dictionary module: the numbers are already known at this point, and keeping
 * the templates beside the splitter means a new field in SplitReport has an
 * obvious place to be described.
 */
const REPORT_TEXT = {
  en: {
    summary: "{chars} chars -> {chunks} chunk(s), budget {max} chars",
    structure: "{paragraphs} paragraph(s), {sentences} sentence(s), longest chunk {longest} chars",
    hard: "{n} sentence(s) exceeded the budget and were split at word boundaries",
    oversized: "WARNING: {n} word(s) longer than the budget were kept whole (never truncated): {list}",
    sep: ", ",
  },
  ar: {
    summary: "{chars} \u062d\u0631\u0641\u064b\u0627 \u2190 {chunks} \u0645\u0642\u0637\u0639\u064b\u0627\u060c \u0628\u062d\u062f\u0651 {max} \u062d\u0631\u0641\u064b\u0627",
    structure: "{paragraphs} \u0641\u0642\u0631\u0629\u060c {sentences} \u062c\u0645\u0644\u0629\u060c \u0623\u0637\u0648\u0644 \u0645\u0642\u0637\u0639 {longest} \u062d\u0631\u0641\u064b\u0627",
    hard: "{n} \u062c\u0645\u0644\u0629 \u062a\u062c\u0627\u0648\u0632\u062a \u0627\u0644\u062d\u062f\u0651 \u0648\u0642\u064f\u0633\u0651\u0645\u062a \u0639\u0646\u062f \u062d\u062f\u0648\u062f \u0627\u0644\u0643\u0644\u0645\u0627\u062a",
    oversized: "\u062a\u0646\u0628\u064a\u0647: {n} \u0643\u0644\u0645\u0629 \u0623\u0637\u0648\u0644 \u0645\u0646 \u0627\u0644\u062d\u062f\u0651 \u0623\u064f\u0628\u0642\u064a\u062a \u0643\u0627\u0645\u0644\u0629 (\u0644\u0645 \u062a\u064f\u0642\u062a\u0637\u0639): {list}",
    sep: "\u060c ",
  },
};

export function reportLines(report, lang) {
  const L = REPORT_TEXT[lang] || REPORT_TEXT.en;
  const fill = (tpl, vars) => tpl.replace(/\{(\w+)\}/g, (m, k) => (vars[k] !== undefined ? String(vars[k]) : m));
  const out = [
    fill(L.summary, { chars: report.chars, chunks: report.chunks, max: report.max_chars }),
    fill(L.structure, { paragraphs: report.paragraphs, sentences: report.sentences, longest: report.longest_chunk }),
  ];
  if (report.hard_split_sentences) {
    out.push(fill(L.hard, { n: report.hard_split_sentences }));
  }
  if (report.oversized_words.length) {
    out.push(fill(L.oversized, { n: report.oversized_words.length,
      list: report.oversized_words.slice(0, 5).join(L.sep) }));
  }
  return out;
}

// -------------------------------------------------------------------- scoring

const DIACRITIC_CHARS = new Set();
for (let c = 0x064b; c <= 0x0655; c++) DIACRITIC_CHARS.add(String.fromCharCode(c));
DIACRITIC_CHARS.add(String.fromCharCode(0x0670));

const CANON_ORDER = new Map([
  [0x0651, 0], [0x064e, 1], [0x064f, 2], [0x0650, 3],
  [0x064b, 4], [0x064c, 5], [0x064d, 6], [0x0652, 7], [0x0670, 8],
  [0x0653, 9], [0x0654, 10], [0x0655, 11],
]);
const LETTER_RE = /[\u0621-\u064a\u0671-\u06d3]/;
const WORD_RE = /\S+/g;

function sortMarks(marks) {
  return marks.slice().sort((a, b) => {
    const oa = CANON_ORDER.has(a.codePointAt(0)) ? CANON_ORDER.get(a.codePointAt(0)) : 99;
    const ob = CANON_ORDER.has(b.codePointAt(0)) ? CANON_ORDER.get(b.codePointAt(0)) : 99;
    return oa - ob;
  });
}

/** (letters, marksPerLetter) for one word; canonical sorts marks into fixed order. */
export function analyzeWord(word, canonical) {
  const letters = [], marks = [];
  for (const ch of word) {
    if (DIACRITIC_CHARS.has(ch)) {
      if (letters.length) marks[marks.length - 1].push(ch);
    } else if (LETTER_RE.test(ch)) {
      letters.push(ch);
      marks.push([]);
    }
  }
  if (canonical) return { letters, marks: marks.map(sortMarks) };
  return { letters, marks };
}

/** Bare text for scoring: drop scoring's diacritic set, then NFC. */
export function bareForScoring(text) {
  let out = "";
  for (const ch of text) if (!DIACRITIC_CHARS.has(ch)) out += ch;
  return out.normalize("NFC");
}

export function canonicalize(text) {
  const out = [];
  let pending = [];
  for (const ch of text.normalize("NFC")) {
    if (DIACRITIC_CHARS.has(ch)) { pending.push(ch); continue; }
    out.push(...sortMarks(pending));
    pending = [];
    out.push(ch);
  }
  out.push(...sortMarks(pending));
  return out.join("");
}

export function scorePair(reference, hypothesis, canonical) {
  const refWords = reference.match(WORD_RE) || [];
  const hypWords = hypothesis.match(WORD_RE) || [];
  const n = Math.min(refWords.length, hypWords.length);

  let derErrors = 0, derTotal = 0, werErrors = 0, werTotal = 0;
  for (let i = 0; i < n; i++) {
    const r = analyzeWord(refWords[i], canonical);
    const h = analyzeWord(hypWords[i], canonical);
    const refMarked = r.marks.filter((m) => m.length).length;
    let errs;
    if (r.letters.join("") !== h.letters.join("")) {
      errs = Math.max(r.letters.length, 1);
      werErrors += 1;
    } else {
      errs = 0;
      for (let k = 0; k < r.marks.length; k++) {
        if (r.marks[k].join("") !== h.marks[k].join("")) errs += 1;
      }
      if (errs) werErrors += 1;
    }
    derErrors += errs;
    derTotal += refMarked;
    werTotal += 1;
  }
  const extra = Math.abs(refWords.length - hypWords.length);
  if (extra) { werErrors += extra; werTotal += extra; derErrors += extra; derTotal += extra; }

  return {
    der: derTotal ? derErrors / derTotal : 0,
    wer: werTotal ? werErrors / werTotal : 0,
    der_errors: derErrors, der_total: derTotal,
    wer_errors: werErrors, words: werTotal,
    word_count_mismatch: refWords.length !== hypWords.length,
  };
}

// ---------------------------------------------------------------- self-test

/** Round-trip the splitter over many texts x budgets. Returns a failure count. */
export function selfTest() {
  const texts = [
    "", " ", "   ", "\n", "\n\n", "a", "hello world",
    "\u0627\u0644\u0633\u0644\u0627\u0645 \u0639\u0644\u064a\u0643\u0645",
    "one. two! three? four", "a\n\nb\n\nc",
    "\u0627\u0644\u0641\u0635\u0644 \u0627\u0644\u0627\u0648\u0644\n\n\u0643\u0627\u0646\u062a \u0627\u0644\u0645\u062f\u064a\u0646\u0629 \u0647\u0627\u062f\u0626\u0629\n\n\u0627\u0644\u0641\u0635\u0644 \u0627\u0644\u062b\u0627\u0646\u064a",
    "supercalifragilisticexpialidocious",
    "\u0645\u0631\u062d\u0628\u0627 \ud83d\ude42 123 abc",
    "x".repeat(500), ("word ".repeat(300)).trim(),
    "\u0643\u0644\u0645\u0629 ".repeat(200).trim(),
    "a\t\tb\r\nc",
  ];
  const budgets = [1, 2, 3, 7, 40, 100, 400, 5000];
  let fails = 0, checks = 0;
  for (const t of texts) {
    for (const b of budgets) {
      const { cores, gaps } = splitText(t, b);
      const back = reassemble(cores, gaps);
      checks++;
      if (back !== t) { fails++; if (fails < 4) console.log("ROUNDTRIP FAIL b=" + b + " " + JSON.stringify(t.slice(0, 40))); }
    }
  }
  return { checks, fails };
}
