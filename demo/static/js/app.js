/*
 * Tashkeel Compare - static Space UI.
 *
 * Everything is built with DOM calls rather than innerHTML: model output is
 * untrusted text and must never be parsed as markup.
 *
 * Every tab keeps the last run in memory and repaints from it when the language
 * changes, so switching language never re-runs a model. That matters: a model
 * run costs seconds, and a reader flipping to Arabic to check a word should not
 * pay for it twice.
 */
import { setOrt, loadNano } from "./engine.js";
import { splitText, reassemble, reportLines, scorePair, bareForScoring, selfTest, canonicalize } from "./text.js";
import { t, getLang, initLang, initSwitch, onLangChange, sampleName, modelLabel, modelName } from "./i18n.js";

setOrt(window.ort);
window.ort.env.wasm.numThreads = 1;              // no COOP/COEP on a static host
window.ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";

const DATA_URL = "data/reference_outputs.json";
const DIACRITIC = new Set();
for (let c = 0x064b; c <= 0x0655; c++) DIACRITIC.add(String.fromCharCode(c));
DIACRITIC.add(String.fromCharCode(0x0670));

// ------------------------------------------------------------------ helpers

function el(tag, attrs, children) {
  const n = document.createElement(tag);
  if (attrs) for (const k of Object.keys(attrs)) {
    if (k === "class") n.className = attrs[k];
    else if (k === "text") n.textContent = attrs[k];
    else if (k === "html") n.innerHTML = attrs[k];
    else if (attrs[k] !== null && attrs[k] !== undefined) n.setAttribute(k, attrs[k]);
  }
  for (const c of children || []) if (c) n.appendChild(c);
  return n;
}
const txt = (s) => document.createTextNode(s);
const clear = (n) => { while (n.firstChild) n.removeChild(n.firstChild); };

/** Split a diacritized string into per-letter units {letter, marks}. */
function units(s) {
  const out = [];
  for (const ch of s) {
    if (DIACRITIC.has(ch)) { if (out.length) out[out.length - 1].marks += ch; }
    else out.push({ letter: ch, marks: "" });
  }
  return out;
}

/**
 * Render hyp with every letter whose canonical mark set differs from base
 * tinted. This is the page's signature: the marks that disagree are the one
 * thing a reader actually needs to see.
 */
function diffNode(base, hyp) {
  const a = units(canonicalize(base));
  const b = units(canonicalize(hyp));
  const frag = document.createDocumentFragment();
  let differed = 0;
  for (let i = 0; i < b.length; i++) {
    const same = a[i] && a[i].letter === b[i].letter && a[i].marks === b[i].marks;
    const piece = b[i].letter + b[i].marks;
    if (!same && a[i]) {
      differed++;
      frag.appendChild(el("span", { class: "d", text: piece }));
    } else {
      frag.appendChild(txt(piece));
    }
  }
  return { node: frag, differed };
}

function table(columns, rows) {
  const thead = el("thead", null, [el("tr", null, columns.map((c) =>
    el("th", { class: c.num ? "num" : null, text: c.label })))]);
  const tbody = el("tbody", null, rows.map((r) =>
    el("tr", { class: r.ours ? "ours" : null }, columns.map((c) =>
      el("td", { class: c.num ? "num" : (r.zero ? "zero" : null), text: String(r[c.key]) })))));
  return el("table", null, [thead, tbody]);
}

// -------------------------------------------------------------------- state

let DATA = null;
const nanoCache = new Map();
const RERENDER = [];
const rerenderAll = () => { for (const fn of RERENDER) fn(); };

async function getNano(key) {
  if (!nanoCache.has(key)) nanoCache.set(key, loadNano(key, ""));
  return nanoCache.get(key);
}

function refOutputFor(text, key) {
  if (!DATA) return null;
  if (text === DATA.default_gold) return DATA.gold_ref[key] || null;
  for (const s of DATA.samples) if (s.text === text) return s.ref[key] || null;
  return null;
}

function modelSpec(key) { return DATA.models.find((m) => m.key === key); }

/*
 * registry.py defines "ours" as "the label ends with (ours)", and the exported
 * data carries the resolved flag. The label test is repeated here so a data file
 * from an older generator cannot silently drop the badge that tells a reader
 * which rows are ours.
 */
function isOurs(spec) {
  return spec.is_ours === true || /\(ours\)$/.test(spec.label || "");
}

/** Run one model. Returns {key, out, seconds, error, precomputed}. */
async function runModel(key, text) {
  const spec = modelSpec(key);
  if (!spec) return { key, out: null, seconds: 0, error: t("err.unknownModel"), precomputed: false };
  if (spec.family === "nano") {
    const t0 = performance.now();
    try {
      const m = await getNano(key);
      const out = await m.diacritize(text);
      return { key, out, seconds: (performance.now() - t0) / 1000, error: null, precomputed: false };
    } catch (e) {
      return { key, out: null, seconds: 0, error: String(e && e.message || e), precomputed: false };
    }
  }
  const cached = refOutputFor(text, key);
  if (cached === null) {
    return { key, out: null, seconds: 0, precomputed: true, error: t("err.preOnlySamples") };
  }
  return { key, out: cached, seconds: 0, error: null, precomputed: true };
}

function card(res, baseOut) {
  const spec = modelSpec(res.key);
  const cls = "card" + (spec.family === "nano" ? " ours" : " precomputed");
  const badges = [];
  if (isOurs(spec)) badges.push(el("span", { class: "badge ours", text: t("card.ours") }));
  if (res.precomputed) badges.push(el("span", { class: "badge pre", text: t("card.precomputed") }));
  if (spec.mark_order && spec.mark_order !== "nfc") {
    badges.push(el("span", { class: "badge order", text: t("card.order") }));
  }

  const head = el("div", { class: "card-head" }, [
    el("span", { class: "card-name", text: modelLabel(spec) }),
    ...badges,
    el("span", { class: "card-stats" }, [
      el("b", { text: t("card.params", { v: spec.params_m.toFixed(2) }) }),
      txt(res.seconds ? "  -  " + t("card.seconds", { s: res.seconds.toFixed(2) }) : ""),
    ]),
  ]);

  const body = el("div", { class: "card-out" });
  let foot = null;
  if (res.error) {
    body.appendChild(el("span", { class: "err", text: res.error }));
  } else if (baseOut && res.out !== baseOut) {
    const d = diffNode(baseOut, res.out);
    body.appendChild(d.node);
    body.classList.add("diffed");
    foot = el("div", { class: "card-foot", text: t("card.differed", { n: d.differed }) });
  } else {
    body.appendChild(txt(res.out));
  }

  return { node: el("div", { class: cls }, [head, body, foot]), differed: foot ? true : false };
}

// --------------------------------------------------------------------- tabs

const PANELS = ["compare", "bulk", "score", "board"];
function selectTab(name) {
  for (const p of PANELS) {
    document.getElementById("tab-" + p).setAttribute("aria-selected", String(p === name));
    document.getElementById("panel-" + p).hidden = p !== name;
  }
  if (name === "board") renderBoard(false);
}
for (const p of PANELS) {
  document.getElementById("tab-" + p).addEventListener("click", () => selectTab(p));
}

// ------------------------------------------------------------ model pickers

function buildPicker(host, keys, selected) {
  clear(host);
  for (const key of keys) {
    const spec = modelSpec(key);
    const cb = el("input", { type: "checkbox" });
    cb.checked = selected.indexOf(key) !== -1;
    cb.value = key;
    const tag = spec.family === "nano"
      ? t("picker.live", { v: spec.params_m.toFixed(2) })
      : t("picker.pre");
    host.appendChild(el("label", null, [cb, txt(modelLabel(spec)), el("span", { class: "mk-tag", text: tag })]));
  }
}
function picked(host) {
  return Array.from(host.querySelectorAll("input:checked")).map((c) => c.value);
}

// ------------------------------------------------------------------ compare

let cmpLast = null;                       // { out, base } of the last run

function paintCompare(results, out) {
  clear(results);
  if (!out.length) return;
  const base = (out.find((r) => !r.error) || {}).out || null;
  let any = false;
  for (const r of out) {
    const c = card(r, base);
    results.appendChild(c.node);
    if (c.differed) any = true;
  }
  if (any) {
    results.insertBefore(el("div", { class: "disagree" }, [
      el("span", { class: "key" }),
      el("span", { text: t("cmp.disagree") }),
    ]), results.firstChild);
  }
}

function initCompare() {
  const input = document.getElementById("cmp-input");
  const samples = document.getElementById("cmp-samples");
  const picker = document.getElementById("cmp-models");
  const status = document.getElementById("cmp-status");
  const results = document.getElementById("cmp-results");

  const refresh = () => {
    clear(samples);
    for (const s of DATA.samples) {
      samples.appendChild(el("button", { class: "sample", type: "button", text: sampleName(s.name),
        onclick: () => { input.value = s.text; } }));
    }
    const keep = picked(picker);
    buildPicker(picker, DATA.models.map((m) => m.key),
      keep.length ? keep : DATA.models.filter((m) => m.default_on).map((m) => m.key));
  };
  refresh();
  input.value = DATA.samples[0].text;
  RERENDER.push(() => { refresh(); if (cmpLast) paintCompare(results, cmpLast.out); });

  document.getElementById("cmp-run").addEventListener("click", async (ev) => {
    const keys = picked(picker);
    clear(results);
    if (!keys.length) { results.appendChild(el("p", { class: "empty", text: t("cmp.pickOne") })); return; }
    if (!input.value.trim()) { results.appendChild(el("p", { class: "empty", text: t("cmp.pasteFirst") })); return; }
    ev.target.disabled = true;
    const out = [];
    for (let i = 0; i < keys.length; i++) {
      status.textContent = t("cmp.running", { model: modelLabel(modelSpec(keys[i])), i: i + 1, n: keys.length });
      out.push(await runModel(keys[i], input.value));
    }
    status.textContent = "";
    ev.target.disabled = false;

    cmpLast = { out };
    paintCompare(results, out);
  });
}

// --------------------------------------------------------------------- bulk

let bulkLast = null;                      // { keys, perModel, rep }

function paintBulk(report, results, last) {
  clear(report); clear(results);
  for (const line of reportLines(last.rep, getLang())) report.appendChild(el("div", { text: line }));
  for (const key of last.keys) {
    const spec = modelSpec(key);
    const body = el("div", { class: "card-out" });
    body.appendChild(txt(last.perModel[key].text));
    results.appendChild(el("div", { class: "card ours" }, [
      el("div", { class: "card-head" }, [
        el("span", { class: "card-name", text: modelLabel(spec) }),
        el("span", { class: "badge ours", text: t("card.ours") }),
        el("span", { class: "card-stats" }, [
          el("b", { text: t("card.params", { v: spec.params_m.toFixed(2) }) }),
          txt("  -  " + t("card.seconds", { s: last.perModel[key].seconds.toFixed(2) })),
        ]),
      ]),
      body,
      el("div", { class: "card-foot", text: t("bulk.foot", { n: last.rep.chunks }) }),
    ]));
  }
}

function initBulk() {
  const input = document.getElementById("bulk-input");
  const picker = document.getElementById("bulk-models");
  const budget = document.getElementById("bulk-budget");
  const budgetOut = document.getElementById("bulk-budget-out");
  const report = document.getElementById("bulk-report");
  const results = document.getElementById("bulk-results");
  const status = document.getElementById("cmp-status");
  const dl = document.getElementById("bulk-download");
  const cp = document.getElementById("bulk-copy");
  let lastOut = null;

  const refresh = () => {
    const keep = picked(picker);
    buildPicker(picker, DATA.models.filter((m) => m.family === "nano").map((m) => m.key),
      keep.length ? keep : ["nano1p1m"]);
  };
  refresh();
  budget.addEventListener("input", () => { budgetOut.textContent = budget.value; });
  RERENDER.push(() => { refresh(); if (bulkLast) paintBulk(report, results, bulkLast); });

  document.getElementById("bulk-file").addEventListener("change", async (ev) => {
    const f = ev.target.files && ev.target.files[0];
    if (f) input.value = await f.text();
  });

  document.getElementById("bulk-run").addEventListener("click", async (ev) => {
    clear(results); clear(report); lastOut = null; bulkLast = null; dl.disabled = true; cp.disabled = true;
    const keys = picked(picker);
    if (!input.value.trim()) { results.appendChild(el("p", { class: "empty", text: t("bulk.pasteFirst") })); return; }
    if (!keys.length) { results.appendChild(el("p", { class: "empty", text: t("cmp.pickOne") })); return; }

    const { cores, gaps, report: rep } = splitText(input.value, Number(budget.value));
    for (const line of reportLines(rep, getLang())) report.appendChild(el("div", { text: line }));
    const back = reassemble(cores, gaps);
    if (back !== input.value) report.appendChild(el("div", { class: "err", text: t("bulk.roundTrip") }));
    if (back !== input.value) return;

    ev.target.disabled = true;
    const perModel = {};
    for (const key of keys) {
      const model = await getNano(key);
      const outs = [];
      const t0 = performance.now();
      for (let i = 0; i < cores.length; i++) {
        status.textContent = t("bulk.chunkOf", { model: modelLabel(modelSpec(key)), i: i + 1, n: cores.length });
        try { outs.push(await model.diacritize(cores[i])); }
        catch (e) { outs.push(cores[i]); }
      }
      perModel[key] = { text: reassemble(outs, gaps), seconds: (performance.now() - t0) / 1000 };
    }
    status.textContent = "";
    ev.target.disabled = false;

    bulkLast = { keys, perModel, rep };
    paintBulk(report, results, bulkLast);
    lastOut = perModel[keys[0]].text;
    dl.disabled = false; cp.disabled = true;
    cp.disabled = false;
  });

  dl.addEventListener("click", () => {
    if (!lastOut) return;
    const blob = new Blob([lastOut], { type: "text/plain;charset=utf-8" });
    const a = el("a", { href: URL.createObjectURL(blob), download: "diacritized.txt" });
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
  });
  cp.addEventListener("click", async () => {
    if (lastOut) { try { await navigator.clipboard.writeText(lastOut); } catch (e) { /* ignore */ } }
  });
}

// -------------------------------------------------------------------- score

let scoreLast = null;                     // { gold, canon, rows }

function paintScore(results, last) {
  clear(results);
  const rows = last.rows.slice().sort((a, b) => a.sortKey - b.sortKey).map((r) => ({
    model: modelName(r.model),
    der: r.runnable ? (r.der * 100).toFixed(2) + " %" : "\u2014",
    wer: r.runnable ? (r.wer * 100).toFixed(2) + " %" : "\u2014",
    note: r.runnable
      ? (r.precomputed ? t("score.precomputed") : t("score.inBrowser", { s: r.seconds.toFixed(2) }))
      : t("score.notRunnable"),
    ours: r.ours, zero: !r.runnable,
  }));
  results.appendChild(table([
    { key: "model", label: t("score.col.model") },
    { key: "der", label: t("score.col.der"), num: true },
    { key: "wer", label: t("score.col.wer"), num: true },
    { key: "note", label: t("score.col.how") },
  ], rows));
  results.appendChild(el("p", { class: "hint", text: t("score.hint") }));
}

function initScore() {
  const ref = document.getElementById("score-ref");
  const canon = document.getElementById("score-canon");
  const results = document.getElementById("score-results");
  ref.value = DATA.default_gold;
  RERENDER.push(() => { if (scoreLast) paintScore(results, scoreLast); });

  document.getElementById("score-run").addEventListener("click", async (ev) => {
    clear(results);
    const gold = ref.value;
    if (!gold.trim()) { results.appendChild(el("p", { class: "empty", text: t("score.empty") })); return; }
    // For the built-in reference use the exact bare string the precomputed
    // reference models were given, so every column answers the same question.
    const bare = (gold === DATA.default_gold && DATA.gold_bare)
      ? DATA.gold_bare : bareForScoring(gold);
    ev.target.disabled = true;
    const rows = [];
    for (const spec of DATA.models) {
      // Our models re-vocalize the bare text live. The reference models can only
      // be shown where a precomputed output exists, which is keyed on the
      // DIACRITIZED default reference, not on the bare text we feed our own.
      let res;
      if (spec.family === "nano") {
        res = await runModel(spec.key, bare);
      } else {
        const pre = refOutputFor(gold, spec.key);
        res = pre === null
          ? { key: spec.key, out: null, seconds: 0, precomputed: true, error: t("score.preOnlyDefault") }
          : { key: spec.key, out: pre, seconds: 0, precomputed: true, error: null };
      }
      if (res.error) {
        rows.push({ model: spec.label, der: 0, wer: 0, seconds: 0, precomputed: false,
                    runnable: false, ours: isOurs(spec), sortKey: 9e9 });
        continue;
      }
      const s = scorePair(gold, res.out, canon.checked);
      rows.push({ model: spec.label, der: s.der, wer: s.wer, seconds: res.seconds,
                  precomputed: res.precomputed, runnable: true, ours: isOurs(spec), sortKey: s.der });
    }
    ev.target.disabled = false;
    scoreLast = { gold, canon: canon.checked, rows };
    paintScore(results, scoreLast);
  });
}

// -------------------------------------------------------------- leaderboard

function renderBoard(force) {
  const host = document.getElementById("board-body");
  if (host.dataset.done && !force) return;
  host.dataset.done = "1";
  clear(host);
  const lb = DATA.leaderboard;

  const quoted = lb.quoted_rows.map((r) => ({
    model: modelName(r.model),
    params: r.params_m ? (r.params_m / 1000).toFixed(r.params_m >= 1000 ? 1 : 2) + " B" : t("board.closed"),
    der: r.values[0].toFixed(2), wer: r.values[1].toFixed(2),
    der2: r.values[2].toFixed(2), wer2: r.values[3].toFixed(2), ours: false, zero: false,
  }));
  const ours = lb.our_rows.map((r) => ({
    model: modelName(r.model),
    params: r.params_m.toFixed(2) + " M",
    der: r.values[0].toFixed(2), wer: r.values[1].toFixed(2),
    der2: r.values[2].toFixed(2), wer2: r.values[3].toFixed(2), ours: true, zero: false,
  }));

  host.appendChild(el("div", { class: "board-block" }, [
    el("h2", { text: t("board.h1") }),
    el("p", { class: "note", text: t("board.note1") }),
    table([
      { key: "model", label: t("board.col.system") }, { key: "params", label: t("board.col.params") },
      { key: "der", label: t("board.col.der"), num: true }, { key: "wer", label: t("board.col.wer"), num: true },
      { key: "der2", label: t("board.col.der2"), num: true }, { key: "wer2", label: t("board.col.wer2"), num: true },
    ], ours.concat(quoted)),
  ]));

  host.appendChild(el("div", { class: "board-block" }, [
    el("h2", { text: t("board.h2") }),
    el("p", { class: "note", text: t("board.note2") }),
    scatter(lb.pareto),
  ]));

  host.appendChild(el("div", { class: "board-block" }, [
    el("h2", { text: t("board.h3") }),
    el("p", { class: "note", text: t("board.note3") }),
    table([
      { key: "model", label: t("board.col.model") },
      { key: "der", label: t("board.col.derLit"), num: true },
      { key: "wer", label: t("board.col.werLit"), num: true },
    ], lb.our_rows.map((r) => {
      const lit = lb.ours_as_emitted[r.model] || r.values;
      return { model: modelName(r.model), der: lit[0].toFixed(2), wer: lit[1].toFixed(2), ours: true, zero: false };
    })),
  ]));
}

function scatter(points) {
  const W = 900, H = 380, L = 62, R = 24, T = 18, B = 46;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 " + W + " " + H);
  svg.setAttribute("class", "scatter");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", t("board.scatterAria"));

  const xs = points.map((p) => Math.log10(p.params_m));
  const x0 = Math.floor(Math.min.apply(null, xs) * 10) / 10;
  const x1 = Math.ceil(Math.max.apply(null, xs) * 10) / 10;
  const y1 = Math.ceil(Math.max.apply(null, points.map((p) => p.der)) / 10) * 10;
  const px = (v) => L + ((Math.log10(v) - x0) / (x1 - x0)) * (W - L - R);
  const py = (v) => H - B - (v / y1) * (H - T - B);

  const add = (tag, attrs, textContent) => {
    const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const k of Object.keys(attrs)) n.setAttribute(k, attrs[k]);
    if (textContent !== undefined) n.textContent = textContent;
    svg.appendChild(n);
    return n;
  };

  for (let d = 0; d <= y1; d += Math.max(10, y1 / 5)) {
    add("line", { class: "grid", x1: L, x2: W - R, y1: py(d), y2: py(d) });
    add("text", { x: L - 8, y: py(d) + 3, "text-anchor": "end" }, d.toFixed(0));
  }
  for (let e = Math.ceil(x0); e <= x1; e++) {
    const v = Math.pow(10, e);
    add("line", { class: "grid", x1: px(v), x2: px(v), y1: T, y2: H - B });
    add("text", { x: px(v), y: H - B + 16, "text-anchor": "middle" },
      v >= 1000 ? (v / 1000) + "B" : v + "M");
  }
  add("line", { class: "axis", x1: L, x2: W - R, y1: H - B, y2: H - B });
  add("line", { class: "axis", x1: L, x2: L, y1: T, y2: H - B });
  add("text", { x: (W + L) / 2, y: H - 8, "text-anchor": "middle" }, t("board.axisX"));
  add("text", { x: 14, y: (H - B + T) / 2, "text-anchor": "middle",
    transform: "rotate(-90 14 " + ((H - B + T) / 2) + ")" }, t("board.axisY"));

  for (const p of points) {
    const c = add("circle", { class: p.ours ? "pt-ours" : "pt-quoted",
      cx: px(p.params_m), cy: py(p.der), r: p.ours ? 6.5 : 4.5 });
    c.appendChild(document.createElementNS("http://www.w3.org/2000/svg", "title")).textContent =
      t("board.point", { model: modelName(p.model), p: p.params_m, der: p.der,
        source: t(p.source === "quoted" ? "source.quoted" : "source.measured") });
  }
  return svg;
}

// --------------------------------------------------------------------- boot

(async function boot() {
  initLang();
  initSwitch(document.getElementById("lang-switch"));
  onLangChange(() => { if (DATA) { rerenderAll(); renderBoard(true); } });

  const t0 = selfTest();
  if (t0.fails) console.warn("splitter self-test failures:", t0.fails, "of", t0.checks);
  try {
    const r = await fetch(DATA_URL);
    DATA = await r.json();
  } catch (e) {
    document.getElementById("cmp-status").textContent = t("cmp.loadFailed", { e: e });
    return;
  }
  initCompare();
  initBulk();
  initScore();
  document.getElementById("cmp-status").textContent = "";
})();
