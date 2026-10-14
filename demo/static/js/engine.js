/*
 * Tashkeel Nano - in-browser inference engine.
 *
 * An exact port of the published packages' tokenizer, 15-class label renderer,
 * passthrough rules and decode strategies. Runs the ONNX export of the same
 * weights via ONNX Runtime Web, so no server is involved.
 *
 * Fidelity notes (these are deliberate, not incidental):
 *  - Tokenization is by CODE POINT, like the Python reference.
 *  - stripMarks drops the same mark ranges and then applies NFC.
 *  - nano1p1m uses the CENTERED sliding-window decode and emits NFC.
 *  - nano4p5m uses the HARD-CHUNK decode and emits SHADDA-FIRST (it does not
 *    normalize), which is what that published package actually does.
 */

// ---------------------------------------------------------------- tokenizer

const SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"];

function buildBaseChars() {
  const chars = [];
  const add = (s) => { for (const ch of s) if (chars.indexOf(ch) === -1) chars.push(ch); };
  let a = "";
  for (let o = 0x0621; o < 0x064b; o++) a += String.fromCharCode(o);
  add(a);
  add(String.fromCharCode(0x0671));
  add(String.fromCharCode(0x0640));
  add("0123456789");
  let d = "";
  for (let o = 0x0660; o < 0x066a; o++) d += String.fromCharCode(o);
  add(d);
  d = "";
  for (let o = 0x06f0; o < 0x06fa; o++) d += String.fromCharCode(o);
  add(d);
  add(" \t\n.,!?;:\u060c\u061b\u061f()[]{}<>\"'\u2026\u00ab\u00bb-/\u0640=+*&@#%$~^|\\\u00a7");
  let asc = "";
  for (let o = 0x21; o < 0x7f; o++) asc += String.fromCharCode(o);
  add(asc);
  return chars;
}

const ID_OF = new Map();
const VOCAB = [];
for (const ch of SPECIALS) { ID_OF.set(ch, VOCAB.length); VOCAB.push(ch); }
for (const ch of buildBaseChars()) {
  if (!ID_OF.has(ch)) { ID_OF.set(ch, VOCAB.length); VOCAB.push(ch); }
}
const UNK = ID_OF.get("<unk>");

export const VOCAB_SIZE = VOCAB.length;

/** str -> int[]; unknown code points become UNK (context-only, never fatal). */
export function encode(text) {
  const out = [];
  for (const ch of text) out.push(ID_OF.has(ch) ? ID_OF.get(ch) : UNK);
  return out;
}

// ------------------------------------------------------------------- labels

export const CLASS_NAMES = [
  "bare", "fatha", "damma", "kasra", "sukun",
  "fathatan", "dammatan", "kasratan",
  "shadda", "shadda_fatha", "shadda_damma", "shadda_kasra",
  "shadda_fathatan", "shadda_dammatan", "shadda_kasratan",
];

const FATHA = "\u064e", DAMMA = "\u064f", KASRA = "\u0650";
const FATHATAN = "\u064b", DAMMATAN = "\u064c", KASRATAN = "\u064d";
const SUKUN = "\u0652", SHADDA = "\u0651";
const VOWEL_LIST = [FATHA, DAMMA, KASRA];
const TANWIN_LIST = [FATHATAN, DAMMATAN, KASRATAN];

/**
 * Class id -> combining-mark string.
 * nfc=true reproduces the 1p1m renderer (vowel before shadda);
 * nfc=false reproduces the 4p5m renderer (shadda first, unnormalized).
 */
export function marksForLabel(label, nfc) {
  if (label === 0) return "";
  if (label >= 1 && label <= 4) return [FATHA, DAMMA, KASRA, SUKUN][label - 1];
  if (label >= 5 && label <= 7) return TANWIN_LIST[label - 5];
  if (label === 8) return SHADDA;
  if (label >= 9 && label <= 14) {
    const pair = SHADDA + VOWEL_LIST.concat(TANWIN_LIST)[label - 9];
    return nfc ? pair.normalize("NFC") : pair;
  }
  throw new Error("bad label " + label);
}

// -------------------------------------------------------------- passthrough

/** Arabic base letters (primary block + alef wasla); tatweel is NOT a base. */
export function isArabicBase(ch) {
  const o = ch.codePointAt(0);
  if (o === 0x0640) return false;
  return (o >= 0x0621 && o <= 0x064a) || o === 0x0671;
}

const MARK_RANGES = [[0x064b, 0x0652], [0x0670, 0x0670], [0x0653, 0x065f], [0x06d6, 0x06ed]];

/** Bare text: drop every combining mark, then NFC-normalize. */
export function stripMarks(text) {
  let out = "";
  for (const ch of text) {
    const o = ch.codePointAt(0);
    let drop = false;
    for (const r of MARK_RANGES) if (o >= r[0] && o <= r[1]) { drop = true; break; }
    if (!drop) out += ch;
  }
  return out.normalize("NFC");
}

// ------------------------------------------------------------------ decoding

/** Start offsets for overlapping windows covering [0, n). */
export function windowStarts(n, window, stride) {
  if (n <= window) return [0];
  const starts = [];
  for (let s = 0; s <= n - window; s += stride) starts.push(s);
  const last = n - window;
  if (starts[starts.length - 1] !== last) starts.push(last);
  return starts;
}

/**
 * Centered sliding-window decode (nano1p1m). Overlapping windows are scored and
 * each base letter takes the mark from the window whose centre is nearest; ties
 * break on the mark's codepoint order. No majority vote.
 */
export async function diacritizeCentered(runModel, text, ctx, nfc) {
  const bare = Array.from(stripMarks(text));
  if (!bare.join("").trim()) return text;
  const n = bare.length;
  const stride = Math.max(1, Math.floor(ctx / 2));
  const votes = [];
  for (let i = 0; i < n; i++) votes.push([]);
  for (const s of windowStarts(n, ctx, stride)) {
    const e = Math.min(s + ctx, n);
    const piece = bare.slice(s, e);
    const pred = await runModel(piece.join(""));
    const center = (s + e - 1) / 2;
    for (let wi = 0; wi < piece.length; wi++) {
      votes[s + wi].push([marksForLabel(pred[wi], nfc), center]);
    }
  }
  let out = "";
  for (let pos = 0; pos < n; pos++) {
    const ch = bare[pos];
    out += ch;
    if (!isArabicBase(ch)) continue;
    const cands = votes[pos];
    if (!cands.length) continue;
    let best = cands[0];
    for (const c of cands) {
      const d = Math.abs(pos - c[1]), bd = Math.abs(pos - best[1]);
      if (d < bd || (d === bd && c[0] < best[0])) best = c;
    }
    out += best[0];
  }
  return nfc ? out.normalize("NFC") : out;
}

/** Hard-chunk decode (nano4p5m): non-overlapping ctx-sized pieces. */
export async function diacritizeChunked(runModel, text, ctx, nfc) {
  const bare = Array.from(stripMarks(text));
  if (!bare.join("").trim()) return text;
  let out = "";
  for (let o = 0; o < bare.length; o += ctx) {
    const piece = bare.slice(o, o + ctx);
    const pred = await runModel(piece.join(""));
    for (let i = 0; i < piece.length; i++) {
      const ch = piece[i];
      out += ch;
      if (isArabicBase(ch)) out += marksForLabel(pred[i], nfc);
    }
  }
  return nfc ? out.normalize("NFC") : out;
}

// ------------------------------------------------------------- ONNX runtime

let ORT = null;

/** Inject the ONNX Runtime (window.ort in the browser, the npm module in Node). */
export function setOrt(o) { ORT = o; }
export function getOrt() { return ORT; }

/** Spec of each nano model: how it was exported and how it must be decoded. */
export const NANO_SPECS = {
  nano1p1m: { file: "models/nano1p1m.onnx", ctx: 256, decode: "centered", nfc: true },
  nano4p5m: { file: "models/nano4p5m.onnx", ctx: 128, decode: "chunked", nfc: false },
};

/** Load one nano model and return an object with a diacritize() method. */
export async function loadNano(key, baseUrl) {
  const spec = NANO_SPECS[key];
  if (!spec) throw new Error("unknown nano model: " + key);
  const session = await ORT.InferenceSession.create(baseUrl + spec.file, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  return new NanoModel(session, spec);
}

export class NanoModel {
  constructor(session, spec) {
    this.session = session;
    this.spec = spec;
    this.ctx = spec.ctx;
  }

  /** Run the encoder over one piece and return per-position class ids. */
  async predict(text) {
    const ids = encode(text);
    const T = ids.length;
    if (T === 0) return [];
    const buf = new BigInt64Array(T);
    for (let i = 0; i < T; i++) buf[i] = BigInt(ids[i]);
    const feeds = {};
    feeds.input_ids = new ORT.Tensor("int64", buf, [1, T]);
    const out = await this.session.run(feeds);
    const logits = out.logits;
    const data = logits.data;
    const pred = new Array(T);
    for (let t = 0; t < T; t++) {
      const base = t * 15;
      let best = 0, bv = data[base];
      for (let c = 1; c < 15; c++) {
        const v = data[base + c];
        if (v > bv) { bv = v; best = c; }
      }
      pred[t] = best;
    }
    return pred;
  }

  async diacritize(text) {
    const run = (t) => this.predict(t);
    if (this.spec.decode === "centered") {
      return diacritizeCentered(run, text, this.ctx, this.spec.nfc);
    }
    return diacritizeChunked(run, text, this.ctx, this.spec.nfc);
  }
}
