/*
 * Tashkeel Compare - language layer.
 *
 * The page ships English and Arabic. Arabic is the language the content is
 * about, so an Arabic-reading visitor should land on an Arabic page rather than
 * on an English page about Arabic; the choice is remembered, and the manual
 * switch always wins over detection.
 *
 * Only user-facing prose lives here. Model names stay as their repo identifiers
 * because they are what a reader will type into a search box, and the numbers
 * are formatted the same way in both languages on purpose - a decimal point is
 * a decimal point.
 */

export const LANGS = ["en", "ar"];
const STORAGE_KEY = "tashkeel.lang";

const DICT = {
  en: {
    "doc.title": "Tashkeel Compare - Arabic diacritization in your browser",
    "doc.description": "Compare a 1.08M-parameter Arabic diacritizer against larger systems. Everything runs on your device; no server.",

    "lang.label": "Language",
    "lang.en": "English",
    "lang.ar": "\u0627\u0644\u0639\u0631\u0628\u064a\u0629",

    "eyebrow": "Arabic diacritization \u00b7 runs entirely on your device",
    "h1": "Tashkeel <em>Compare</em>",
    "lede": "A 1.08 million parameter model vocalizes Arabic with no server and no GPU. Put it beside far larger systems, paste a whole document, and judge the marks yourself.",
    "fact.smallest": "Our smallest",
    "fact.smallest.v": "1.08 M",
    "fact.runs": "Runs on",
    "fact.runs.v": "Your CPU",
    "fact.network": "Network",
    "fact.network.v": "None",

    "nav.sections": "Sections",
    "tab.compare": "Compare",
    "tab.bulk": "Bulk text",
    "tab.score": "Score a reference",
    "tab.board": "Leaderboard",

    "cmp.input": "Arabic text to vocalize",
    "cmp.placeholder": "\u0627\u0644\u0633\u0644\u0627\u0645 \u0639\u0644\u064a\u0643\u0645",
    "cmp.samples": "Sample texts",
    "cmp.models": "Models",
    "cmp.run": "Vocalize",
    "cmp.pickOne": "Pick at least one model.",
    "cmp.pasteFirst": "Paste some Arabic text first.",
    "cmp.running": "Running {model} ({i}/{n})...",
    "cmp.disagree": "Tinted letters are vocalized differently from the first column. Comparison is order-insensitive, so a shadda written before or after its vowel does not count as a difference.",
    "cmp.loadFailed": "Could not load data: {e}",

    "bulk.input": "Paste a document, or upload a file",
    "bulk.upload": "Upload .txt",
    "bulk.budgetLabel": "Chunk budget",
    "bulk.budgetUnit": "chars",
    "bulk.models": "Models",
    "bulk.run": "Split & vocalize",
    "bulk.download": "Download result",
    "bulk.copy": "Copy result",
    "bulk.pasteFirst": "Paste a document or upload a .txt file.",
    "bulk.roundTrip": "Splitter round-trip failed; output withheld.",
    "bulk.chunkOf": "{model}: chunk {i} of {n}...",
    "bulk.foot": "{n} chunk(s) reassembled with the original whitespace",

    "score.ref": "Reference text (fully vocalized gold)",
    "score.canon": "Ignore mark order",
    "score.canonHint": "(compares readings, not spelling conventions)",
    "score.run": "Score",
    "score.blurb": "Marks are stripped from the reference, every model re-vocalizes the bare text, and each output is scored against the original.",
    "score.empty": "Paste a fully vocalized reference.",
    "score.col.model": "Model",
    "score.col.der": "DER",
    "score.col.wer": "WER",
    "score.col.how": "How",
    "score.notRunnable": "not runnable here",
    "score.precomputed": "precomputed",
    "score.inBrowser": "{s} s in browser",
    "score.preOnlyDefault": "precomputed output exists only for the default reference",
    "score.hint": "DER is the share of marked reference letters whose mark set differs; WER is the share of words that differ. Both are computed here, on this one passage, and are not the benchmark's letter-level metric. The pre-filled reference is the opening excerpt of the SadeedDiac-25 benchmark reference, taken by position rather than picked to favour anyone; the Board tab carries the full 1200-row comparison.",

    "board.h1": "SadeedDiac-25, as published",
    "board.note1": "Quoted from the benchmark's own table. These are letter-level DER/WER with the benchmark's own correction, produced by its evaluator - not by this page. Our rows are measured with the same evaluator on all 1200 rows so the columns are comparable; our interactive scores elsewhere on this page are a different, word-level metric and are never mixed into this table.",
    "board.col.system": "System",
    "board.col.params": "Params",
    "board.col.der": "DER (CE)",
    "board.col.wer": "WER (CE)",
    "board.col.der2": "DER w/o CE",
    "board.col.wer2": "WER w/o CE",
    "board.h2": "Quality against size",
    "board.note2": "Same table, plotted by parameter count (log scale). Saffron points are ours; lapis points are quoted systems.",
    "board.h3": "As emitted, before normalising mark order",
    "board.note3": "The same predictions scored literally, without canonicalising mark order. The gap between these two readings is an artefact of how the reference writes geminates, not a difference in accuracy - which is exactly why this page's interactive scoring canonicalises by default.",
    "board.col.model": "Model",
    "board.col.derLit": "DER literal",
    "board.col.werLit": "WER literal",
    "board.closed": "closed",
    "board.axisX": "parameters (log scale)",
    "board.axisY": "DER (CE)",
    "board.scatterAria": "Diacritization error against parameter count",
    "board.point": "{model} - {p} M params, DER {der} ({source})",
    "source.quoted": "quoted",
    "source.measured": "measured here",

    "card.ours": "ours",
    "card.precomputed": "precomputed",
    "card.order": "shadda-first",
    "card.params": "{v} M params",
    "card.seconds": "{s} s",
    "card.differed": "{n} letter(s) vocalized differently from the first column",
    "picker.live": "{v}M live",
    "picker.pre": "precomputed",
    "err.unknownModel": "unknown model",
    "err.preOnlySamples": "precomputed output exists only for the built-in samples and the default reference",

    "col.h2": "What this page does and does not claim",
    "col.1": "<strong>Our two models run live</strong> in your browser through ONNX Runtime Web. Their output is byte-identical to the published Python packages; that equivalence is tested.",
    "col.2": "<strong>Mishkala and FLAN-T5 outputs on the built-in samples are precomputed.</strong> They cannot run on a static page, and they are labelled as such wherever they appear.",
    "col.3": "<strong>Leaderboard numbers are quoted</strong> from the published SadeedDiac-25 table, produced by that benchmark's own letter-level evaluator. They are never mixed with ours.",
    "col.4": "<strong>Every number says which evaluator produced it.</strong> The Board's <em>measured here</em> rows come from the benchmark's own letter-level evaluator, scored on all 1200 rows. The Score tab runs a different, self-contained scorer inside this page - letter-level DER over marked letters, word-level WER - and says so on the tab. Nothing is placed in a column it does not belong to.",
  },

  ar: {
    "doc.title": "\u0645\u0642\u0627\u0631\u0646\u0629 \u0627\u0644\u062a\u0634\u0643\u064a\u0644 - \u062a\u0634\u0643\u064a\u0644 \u0627\u0644\u0639\u0631\u0628\u064a\u0629 \u062f\u0627\u062e\u0644 \u0645\u062a\u0635\u0641\u062d\u0643",
    "doc.description": "\u0642\u0627\u0631\u0646 \u0646\u0645\u0648\u0630\u062c \u062a\u0634\u0643\u064a\u0644 \u0639\u0631\u0628\u064a \u0628\u0640 1.08 \u0645\u0644\u064a\u0648\u0646 \u0645\u0639\u0627\u0645\u0644 \u0645\u0639 \u0623\u0646\u0637\u0645\u0629 \u0623\u0643\u0628\u0631. \u0643\u0644 \u0634\u064a\u0621 \u064a\u0639\u0645\u0644 \u0639\u0644\u0649 \u062c\u0647\u0627\u0632\u0643 \u062f\u0648\u0646 \u062e\u0627\u062f\u0645.",

    "lang.label": "\u0627\u0644\u0644\u063a\u0629",
    "lang.en": "English",
    "lang.ar": "\u0627\u0644\u0639\u0631\u0628\u064a\u0629",

    "eyebrow": "\u062a\u0634\u0643\u064a\u0644 \u0627\u0644\u0639\u0631\u0628\u064a\u0629 \u00b7 \u064a\u0639\u0645\u0644 \u0628\u0627\u0644\u0643\u0627\u0645\u0644 \u0639\u0644\u0649 \u062c\u0647\u0627\u0632\u0643",
    "h1": "\u0645\u0642\u0627\u0631\u0646\u0629 <em>\u0627\u0644\u062a\u0634\u0643\u064a\u0644</em>",
    "lede": "\u0646\u0645\u0648\u0630\u062c \u0628\u0640 1.08 \u0645\u0644\u064a\u0648\u0646 \u0645\u0639\u0627\u0645\u0644 \u064a\u0634\u0643\u0651\u0644 \u0627\u0644\u0639\u0631\u0628\u064a\u0629 \u062f\u0648\u0646 \u062e\u0627\u062f\u0645 \u0648\u062f\u0648\u0646 \u0643\u0627\u0631\u062a \u0631\u0633\u0648\u0645. \u0636\u0639\u0647 \u0628\u062c\u0627\u0646\u0628 \u0623\u0646\u0637\u0645\u0629 \u0623\u0643\u0628\u0631 \u0628\u0645\u0631\u0627\u062a\u0628\u060c \u0648\u0627\u0644\u0635\u0642 \u0645\u0633\u062a\u0646\u062f\u064b\u0627 \u0643\u0627\u0645\u0644\u064b\u0627\u060c \u0648\u0627\u062d\u0643\u0645 \u0639\u0644\u0649 \u0627\u0644\u0639\u0644\u0627\u0645\u0627\u062a \u0628\u0646\u0641\u0633\u0643.",
    "fact.smallest": "\u0623\u0635\u063a\u0631 \u0646\u0645\u0648\u0630\u062c \u0644\u062f\u064a\u0646\u0627",
    "fact.smallest.v": "1.08 \u0645\u0644\u064a\u0648\u0646",
    "fact.runs": "\u064a\u0639\u0645\u0644 \u0639\u0644\u0649",
    "fact.runs.v": "\u0645\u0639\u0627\u0644\u062c\u0643",
    "fact.network": "\u0627\u0644\u0634\u0628\u0643\u0629",
    "fact.network.v": "\u0644\u0627 \u0634\u064a\u0621",

    "nav.sections": "\u0627\u0644\u0623\u0642\u0633\u0627\u0645",
    "tab.compare": "\u0642\u0627\u0631\u0646",
    "tab.bulk": "\u0646\u0635 \u0637\u0648\u064a\u0644",
    "tab.score": "\u0642\u064a\u0651\u0645 \u0645\u0631\u062c\u0639\u064b\u0627",
    "tab.board": "\u0644\u0648\u062d\u0629 \u0627\u0644\u062a\u0631\u062a\u064a\u0628",

    "cmp.input": "\u0646\u0635 \u0639\u0631\u0628\u064a \u0644\u062a\u0634\u0643\u064a\u0644\u0647",
    "cmp.placeholder": "\u0627\u0644\u0633\u0644\u0627\u0645 \u0639\u0644\u064a\u0643\u0645",
    "cmp.samples": "\u0646\u0635\u0648\u0635 \u0646\u0645\u0648\u0630\u062c\u064a\u0629",
    "cmp.models": "\u0627\u0644\u0646\u0645\u0627\u0630\u062c",
    "cmp.run": "\u0634\u0643\u0651\u0644",
    "cmp.pickOne": "\u0627\u062e\u062a\u0631 \u0646\u0645\u0648\u0630\u062c\u064b\u0627 \u0648\u0627\u062d\u062f\u064b\u0627 \u0639\u0644\u0649 \u0627\u0644\u0623\u0642\u0644.",
    "cmp.pasteFirst": "\u0627\u0644\u0635\u0642 \u0646\u0635\u064b\u0627 \u0639\u0631\u0628\u064a\u064b\u0627 \u0623\u0648\u0644\u064b\u0627.",
    "cmp.running": "\u062c\u0627\u0631\u064d \u062a\u0634\u063a\u064a\u0644 {model} ({i}/{n})...",
    "cmp.disagree": "\u0627\u0644\u062d\u0631\u0648\u0641 \u0627\u0644\u0645\u0638\u0644\u0644\u0629 \u0634\u064f\u0643\u0651\u0644\u062a \u0628\u0634\u0643\u0644 \u0645\u062e\u062a\u0644\u0641 \u0639\u0646 \u0627\u0644\u0639\u0645\u0648\u062f \u0627\u0644\u0623\u0648\u0644. \u0627\u0644\u0645\u0642\u0627\u0631\u0646\u0629 \u0644\u0627 \u062a\u062a\u0623\u062b\u0631 \u0628\u062a\u0631\u062a\u064a\u0628 \u0627\u0644\u0639\u0644\u0627\u0645\u0627\u062a\u060c \u0641\u0627\u0644\u0634\u062f\u0651\u0629 \u0642\u0628\u0644 \u062d\u0631\u0643\u062a\u0647\u0627 \u0623\u0648 \u0628\u0639\u062f\u0647\u0627 \u0644\u0627 \u062a\u064f\u0639\u062f\u0651 \u0627\u062e\u062a\u0644\u0627\u0641\u064b\u0627.",
    "cmp.loadFailed": "\u062a\u0639\u0630\u0651\u0631 \u062a\u062d\u0645\u064a\u0644 \u0627\u0644\u0628\u064a\u0627\u0646\u0627\u062a: {e}",

    "bulk.input": "\u0627\u0644\u0635\u0642 \u0645\u0633\u062a\u0646\u062f\u064b\u0627 \u0623\u0648 \u0627\u0631\u0641\u0639 \u0645\u0644\u0641\u064b\u0627",
    "bulk.upload": "\u0627\u0631\u0641\u0639 \u0645\u0644\u0641 .txt",
    "bulk.budgetLabel": "\u062d\u062f\u0651 \u0627\u0644\u0645\u0642\u0637\u0639",
    "bulk.budgetUnit": "\u062d\u0631\u0641\u064b\u0627",
    "bulk.models": "\u0627\u0644\u0646\u0645\u0627\u0630\u062c",
    "bulk.run": "\u0642\u0633\u0651\u0645 \u0648\u0634\u0643\u0651\u0644",
    "bulk.download": "\u0646\u0632\u0651\u0644 \u0627\u0644\u0646\u062a\u064a\u062c\u0629",
    "bulk.copy": "\u0627\u0646\u0633\u062e \u0627\u0644\u0646\u062a\u064a\u062c\u0629",
    "bulk.pasteFirst": "\u0627\u0644\u0635\u0642 \u0645\u0633\u062a\u0646\u062f\u064b\u0627 \u0623\u0648 \u0627\u0631\u0641\u0639 \u0645\u0644\u0641 .txt.",
    "bulk.roundTrip": "\u0641\u0634\u0644 \u0627\u0644\u062a\u062d\u0642\u0642 \u0645\u0646 \u0625\u0639\u0627\u062f\u0629 \u0627\u0644\u062a\u062c\u0645\u064a\u0639\u061b \u062d\u064f\u062c\u0628\u062a \u0627\u0644\u0646\u062a\u064a\u062c\u0629.",
    "bulk.chunkOf": "{model}: \u0627\u0644\u0645\u0642\u0637\u0639 {i} \u0645\u0646 {n}...",
    "bulk.foot": "\u0623\u064f\u0639\u064a\u062f \u062a\u062c\u0645\u064a\u0639 {n} \u0645\u0642\u0637\u0639\u064b\u0627 \u0645\u0639 \u0627\u0644\u0645\u0633\u0627\u0641\u0627\u062a \u0627\u0644\u0623\u0635\u0644\u064a\u0629",

    "score.ref": "\u0627\u0644\u0646\u0635 \u0627\u0644\u0645\u0631\u062c\u0639\u064a (\u0645\u0634\u0643\u0648\u0644 \u0628\u0627\u0644\u0643\u0627\u0645\u0644)",
    "score.canon": "\u062a\u062c\u0627\u0647\u0644 \u062a\u0631\u062a\u064a\u0628 \u0627\u0644\u0639\u0644\u0627\u0645\u0627\u062a",
    "score.canonHint": "(\u064a\u0642\u0627\u0631\u0646 \u0627\u0644\u0646\u0637\u0642 \u0644\u0627 \u0627\u0644\u0625\u0645\u0644\u0627\u0621)",
    "score.run": "\u0642\u0650\u0633",
    "score.blurb": "\u062a\u064f\u0646\u0632\u0639 \u0627\u0644\u0639\u0644\u0627\u0645\u0627\u062a \u0645\u0646 \u0627\u0644\u0646\u0635 \u0627\u0644\u0645\u0631\u062c\u0639\u064a\u060c \u0648\u064a\u0639\u064a\u062f \u0643\u0644 \u0646\u0645\u0648\u0630\u062c \u062a\u0634\u0643\u064a\u0644 \u0627\u0644\u0646\u0635 \u0627\u0644\u0645\u062c\u0631\u0651\u062f\u060c \u062b\u0645 \u062a\u064f\u0642\u0627\u0633 \u0643\u0644 \u0645\u062e\u0631\u062c\u0627\u062a \u0645\u0642\u0627\u0628\u0644 \u0627\u0644\u0623\u0635\u0644.",
    "score.empty": "\u0627\u0644\u0635\u0642 \u0646\u0635\u064b\u0627 \u0645\u0631\u062c\u0639\u064a\u064b\u0627 \u0645\u0634\u0643\u0648\u0644\u064b\u0627 \u0628\u0627\u0644\u0643\u0627\u0645\u0644.",
    "score.col.model": "\u0627\u0644\u0646\u0645\u0648\u0630\u062c",
    "score.col.der": "DER",
    "score.col.wer": "WER",
    "score.col.how": "\u0637\u0631\u064a\u0642\u0629 \u0627\u0644\u062a\u0634\u063a\u064a\u0644",
    "score.notRunnable": "\u063a\u064a\u0631 \u0642\u0627\u0628\u0644 \u0644\u0644\u062a\u0634\u063a\u064a\u0644 \u0647\u0646\u0627",
    "score.precomputed": "\u0645\u062d\u0633\u0648\u0628 \u0645\u0633\u0628\u0642\u064b\u0627",
    "score.inBrowser": "{s} \u062b \u0641\u064a \u0627\u0644\u0645\u062a\u0635\u0641\u062d",
    "score.preOnlyDefault": "\u0627\u0644\u0645\u062e\u0631\u062c\u0627\u062a \u0627\u0644\u0645\u062d\u0633\u0648\u0628\u0629 \u0645\u0633\u0628\u0642\u064b\u0627 \u0645\u062a\u0627\u062d\u0629 \u0641\u0642\u0637 \u0644\u0644\u0645\u0631\u062c\u0639 \u0627\u0644\u0627\u0641\u062a\u0631\u0627\u0636\u064a",
    "score.hint": "DER \u0647\u0648 \u0646\u0633\u0628\u0629 \u0627\u0644\u062d\u0631\u0648\u0641 \u0627\u0644\u0645\u0634\u0643\u0648\u0644\u0629 \u0641\u064a \u0627\u0644\u0645\u0631\u062c\u0639 \u0627\u0644\u062a\u064a \u062a\u062e\u062a\u0644\u0641 \u0645\u062c\u0645\u0648\u0639\u0629 \u0639\u0644\u0627\u0645\u0627\u062a\u0647\u0627\u060c \u0648 WER \u0647\u0648 \u0646\u0633\u0628\u0629 \u0627\u0644\u0643\u0644\u0645\u0627\u062a \u0627\u0644\u0645\u062e\u062a\u0644\u0641\u0629. \u0643\u0644\u0627\u0647\u0645\u0627 \u064a\u064f\u062d\u0633\u0628 \u0647\u0646\u0627\u060c \u0639\u0644\u0649 \u0647\u0630\u0627 \u0627\u0644\u0645\u0642\u0637\u0639 \u0648\u062d\u062f\u0647\u060c \u0648\u0644\u064a\u0633\u0627 \u0645\u0642\u064a\u0627\u0633 \u0627\u0644\u0645\u0639\u064a\u0627\u0631 \u0639\u0644\u0649 \u0645\u0633\u062a\u0648\u0649 \u0627\u0644\u062d\u0631\u0641. \u0627\u0644\u0646\u0635 \u0627\u0644\u0645\u0631\u062c\u0639\u064a \u0627\u0644\u0645\u064f\u0639\u0628\u0651\u0623 \u0645\u0633\u0628\u0642\u064b\u0627 \u0647\u0648 \u0645\u0642\u062a\u0637\u0641 \u0627\u0641\u062a\u062a\u0627\u062d\u064a \u0645\u0646 \u0645\u0631\u062c\u0639 SadeedDiac-25\u060c \u0645\u0623\u062e\u0648\u0630 \u0628\u0627\u0644\u0645\u0648\u0642\u0639 \u0644\u0627 \u0644\u0645\u062d\u0627\u0628\u0627\u0629 \u0623\u062d\u062f\u061b \u0648\u062a\u0628\u0648\u064a\u0628 \u0644\u0648\u062d\u0629 \u0627\u0644\u062a\u0631\u062a\u064a\u0628 \u064a\u062d\u0645\u0644 \u0627\u0644\u0645\u0642\u0627\u0631\u0646\u0629 \u0627\u0644\u0643\u0627\u0645\u0644\u0629 \u0639\u0644\u0649 1200 \u0633\u0637\u0631\u064b\u0627.",

    "board.h1": "SadeedDiac-25 \u0643\u0645\u0627 \u0646\u064f\u0634\u0631\u062a",
    "board.note1": "\u0645\u0646\u0642\u0648\u0644\u0629 \u0645\u0646 \u062c\u062f\u0648\u0644 \u0627\u0644\u0645\u0639\u064a\u0627\u0631 \u0646\u0641\u0633\u0647. \u0647\u0630\u0647 \u0642\u064a\u0645 DER/WER \u0639\u0644\u0649 \u0645\u0633\u062a\u0648\u0649 \u0627\u0644\u062d\u0631\u0641 \u0645\u0639 \u062a\u0635\u062d\u064a\u062d \u0627\u0644\u0645\u0639\u064a\u0627\u0631 \u0627\u0644\u062e\u0627\u0635\u060c \u0648\u0623\u0646\u062a\u062c\u0647\u0627 \u0645\u0642\u064a\u0651\u0645\u0647 \u0644\u0627 \u0647\u0630\u0647 \u0627\u0644\u0635\u0641\u062d\u0629. \u0648\u0635\u0641\u0648\u0641\u0646\u0627 \u0642\u064a\u0633\u062a \u0628\u0627\u0644\u0645\u0642\u064a\u0651\u0645 \u0646\u0641\u0633\u0647 \u0639\u0644\u0649 1200 \u0633\u0637\u0631\u064b\u0627 \u0644\u062a\u0643\u0648\u0646 \u0627\u0644\u0623\u0639\u0645\u062f\u0629 \u0642\u0627\u0628\u0644\u0629 \u0644\u0644\u0645\u0642\u0627\u0631\u0646\u0629\u061b \u0623\u0645\u0651\u0627 \u0627\u0644\u062a\u0642\u064a\u064a\u0645 \u0627\u0644\u062a\u0641\u0627\u0639\u0644\u064a \u0641\u064a \u0628\u0627\u0642\u064a \u0627\u0644\u0635\u0641\u062d\u0629 \u0641\u0647\u0648 \u0645\u0642\u064a\u0627\u0633 \u0645\u062e\u062a\u0644\u0641 \u0644\u0627 \u064a\u064f\u062e\u0644\u0637 \u0641\u064a \u0647\u0630\u0627 \u0627\u0644\u062c\u062f\u0648\u0644.",
    "board.col.system": "\u0627\u0644\u0646\u0638\u0627\u0645",
    "board.col.params": "\u0627\u0644\u0645\u0639\u0627\u0645\u0644\u0627\u062a",
    "board.col.der": "DER (CE)",
    "board.col.wer": "WER (CE)",
    "board.col.der2": "DER \u0628\u062f\u0648\u0646 CE",
    "board.col.wer2": "WER \u0628\u062f\u0648\u0646 CE",
    "board.h2": "\u0627\u0644\u062c\u0648\u062f\u0629 \u0645\u0642\u0627\u0628\u0644 \u0627\u0644\u062d\u062c\u0645",
    "board.note2": "\u0627\u0644\u062c\u062f\u0648\u0644 \u0646\u0641\u0633\u0647 \u0645\u0631\u0633\u0648\u0645\u064b\u0627 \u062d\u0633\u0628 \u0639\u062f\u062f \u0627\u0644\u0645\u0639\u0627\u0645\u0644\u0627\u062a (\u0645\u0642\u064a\u0627\u0633 \u0644\u0648\u063a\u0627\u0631\u064a\u062a\u0645\u064a). \u0627\u0644\u0646\u0642\u0627\u0637 \u0627\u0644\u0632\u0639\u0641\u0631\u0627\u0646\u064a\u0629 \u0644\u0646\u0627\u060c \u0648\u0627\u0644\u0646\u064a\u0644\u064a\u0629 \u0623\u0646\u0637\u0645\u0629 \u0645\u0646\u0642\u0648\u0644\u0629.",
    "board.h3": "\u0643\u0645\u0627 \u0647\u064a \u0645\u064f\u062e\u0631\u064e\u062c\u0629\u060c \u0642\u0628\u0644 \u062a\u0648\u062d\u064a\u062f \u062a\u0631\u062a\u064a\u0628 \u0627\u0644\u0639\u0644\u0627\u0645\u0627\u062a",
    "board.note3": "\u0627\u0644\u062a\u0646\u0628\u0624\u0627\u062a \u0646\u0641\u0633\u0647\u0627 \u0645\u0642\u064a\u0633\u0629 \u062d\u0631\u0641\u064a\u064b\u0627\u060c \u062f\u0648\u0646 \u062a\u0648\u062d\u064a\u062f \u062a\u0631\u062a\u064a\u0628 \u0627\u0644\u0639\u0644\u0627\u0645\u0627\u062a. \u0627\u0644\u0641\u0627\u0631\u0642 \u0628\u064a\u0646 \u0627\u0644\u0642\u0631\u0627\u0621\u062a\u064a\u0646 \u0623\u062b\u0631 \u0645\u0646 \u0637\u0631\u064a\u0642\u0629 \u0643\u062a\u0627\u0628\u0629 \u0627\u0644\u0645\u0631\u062c\u0639 \u0644\u0644\u062d\u0631\u0648\u0641 \u0627\u0644\u0645\u0634\u062f\u0651\u062f\u0629\u060c \u0648\u0644\u064a\u0633 \u0641\u0631\u0642\u064b\u0627 \u0641\u064a \u0627\u0644\u062f\u0642\u0651\u0629 - \u0648\u0647\u0630\u0627 \u0628\u0627\u0644\u0636\u0628\u0637 \u0633\u0628\u0628 \u062a\u0648\u062d\u064a\u062f \u0647\u0630\u0647 \u0627\u0644\u0635\u0641\u062d\u0629 \u0644\u062a\u0631\u062a\u064a\u0628 \u0627\u0644\u0639\u0644\u0627\u0645\u0627\u062a \u0627\u0641\u062a\u0631\u0627\u0636\u064a\u064b\u0627 \u0639\u0646\u062f \u0627\u0644\u062a\u0642\u064a\u064a\u0645.",
    "board.col.model": "\u0627\u0644\u0646\u0645\u0648\u0630\u062c",
    "board.col.derLit": "DER \u062d\u0631\u0641\u064a",
    "board.col.werLit": "WER \u062d\u0631\u0641\u064a",
    "board.closed": "\u0645\u063a\u0644\u0642",
    "board.axisX": "\u0639\u062f\u062f \u0627\u0644\u0645\u0639\u0627\u0645\u0644\u0627\u062a (\u0645\u0642\u064a\u0627\u0633 \u0644\u0648\u063a\u0627\u0631\u064a\u062a\u0645\u064a)",
    "board.axisY": "DER (CE)",
    "board.scatterAria": "\u062e\u0637\u0623 \u0627\u0644\u062a\u0634\u0643\u064a\u0644 \u0645\u0642\u0627\u0628\u0644 \u0639\u062f\u062f \u0627\u0644\u0645\u0639\u0627\u0645\u0644\u0627\u062a",
    "board.point": "{model} - {p} \u0645\u0644\u064a\u0648\u0646 \u0645\u0639\u0627\u0645\u0644\u060c DER {der} ({source})",
    "source.quoted": "\u0645\u0646\u0642\u0648\u0644",
    "source.measured": "\u0645\u0642\u064a\u0633 \u0647\u0646\u0627",

    "card.ours": "\u062e\u0627\u0635\u062a\u0646\u0627",
    "card.precomputed": "\u0645\u062d\u0633\u0648\u0628 \u0645\u0633\u0628\u0642\u064b\u0627",
    "card.order": "\u0627\u0644\u0634\u062f\u0651\u0629 \u0623\u0648\u0644\u064b\u0627",
    "card.params": "{v} \u0645\u0644\u064a\u0648\u0646 \u0645\u0639\u0627\u0645\u0644",
    "card.seconds": "{s} \u062b",
    "card.differed": "{n} \u062d\u0631\u0641\u064b\u0627 \u0634\u064f\u0643\u0651\u0644 \u0628\u0634\u0643\u0644 \u0645\u062e\u062a\u0644\u0641 \u0639\u0646 \u0627\u0644\u0639\u0645\u0648\u062f \u0627\u0644\u0623\u0648\u0644",
    "picker.live": "{v}M \u0645\u0628\u0627\u0634\u0631",
    "picker.pre": "\u0645\u062d\u0633\u0648\u0628 \u0645\u0633\u0628\u0642\u064b\u0627",
    "err.unknownModel": "\u0646\u0645\u0648\u0630\u062c \u063a\u064a\u0631 \u0645\u0639\u0631\u0648\u0641",
    "err.preOnlySamples": "\u0627\u0644\u0645\u062e\u0631\u062c\u0627\u062a \u0627\u0644\u0645\u062d\u0633\u0648\u0628\u0629 \u0645\u0633\u0628\u0642\u064b\u0627 \u0645\u062a\u0627\u062d\u0629 \u0641\u0642\u0637 \u0644\u0644\u0646\u0635\u0648\u0635 \u0627\u0644\u0646\u0645\u0648\u0630\u062c\u064a\u0629 \u0648\u0644\u0644\u0645\u0631\u062c\u0639 \u0627\u0644\u0627\u0641\u062a\u0631\u0627\u0636\u064a",

    "col.h2": "\u0645\u0627 \u062a\u062f\u0651\u0639\u064a\u0647 \u0647\u0630\u0647 \u0627\u0644\u0635\u0641\u062d\u0629 \u0648\u0645\u0627 \u0644\u0627 \u062a\u062f\u0651\u0639\u064a\u0647",
    "col.1": "<strong>\u0646\u0645\u0648\u0630\u062c\u0627\u0646\u0627 \u0627\u0644\u0627\u062b\u0646\u0627\u0646 \u064a\u0639\u0645\u0644\u0627\u0646 \u062d\u064a\u0651\u064b\u0627</strong> \u0641\u064a \u0645\u062a\u0635\u0641\u062d\u0643 \u0639\u0628\u0631 ONNX Runtime Web. \u0645\u062e\u0631\u062c\u0627\u062a\u0647\u0645\u0627 \u0645\u0637\u0627\u0628\u0642\u0629 \u0628\u0627\u0644\u0628\u0627\u064a\u062a \u0644\u062d\u0632\u0645 Python \u0627\u0644\u0645\u0646\u0634\u0648\u0631\u0629\u060c \u0648\u0647\u0630\u0627 \u0627\u0644\u062a\u0637\u0627\u0628\u0642 \u0645\u062e\u062a\u0628\u0631.",
    "col.2": "<strong>\u0645\u062e\u0631\u062c\u0627\u062a Mishkala \u0648 FLAN-T5 \u0639\u0644\u0649 \u0627\u0644\u0646\u0635\u0648\u0635 \u0627\u0644\u0646\u0645\u0648\u0630\u062c\u064a\u0629 \u0645\u062d\u0633\u0648\u0628\u0629 \u0645\u0633\u0628\u0642\u064b\u0627.</strong> \u0644\u0627 \u064a\u0645\u0643\u0646 \u062a\u0634\u063a\u064a\u0644\u0647\u0645\u0627 \u0641\u064a \u0635\u0641\u062d\u0629 \u0633\u0627\u0643\u0646\u0629\u060c \u0648\u0647\u0645\u0627 \u0645\u0648\u0633\u0648\u0645\u0627\u0646 \u0628\u0630\u0644\u0643 \u062d\u064a\u062b\u0645\u0627 \u0638\u0647\u0631\u0627.",
    "col.3": "<strong>\u0623\u0631\u0642\u0627\u0645 \u0644\u0648\u062d\u0629 \u0627\u0644\u062a\u0631\u062a\u064a\u0628 \u0645\u0646\u0642\u0648\u0644\u0629</strong> \u0645\u0646 \u062c\u062f\u0648\u0644 SadeedDiac-25 \u0627\u0644\u0645\u0646\u0634\u0648\u0631\u060c \u0648\u0623\u0646\u062a\u062c\u0647\u0627 \u0645\u0642\u064a\u0651\u0645 \u0627\u0644\u0645\u0639\u064a\u0627\u0631 \u0639\u0644\u0649 \u0645\u0633\u062a\u0648\u0649 \u0627\u0644\u062d\u0631\u0641. \u0644\u0627 \u062a\u064f\u062e\u0644\u0637 \u0623\u0628\u062f\u064b\u0627 \u0628\u0623\u0631\u0642\u0627\u0645\u0646\u0627.",
    "col.4": "<strong>\u0643\u0644 \u0631\u0642\u0645 \u064a\u064f\u0630\u0643\u0631 \u0627\u0644\u0645\u0642\u064a\u0651\u0645 \u0627\u0644\u0630\u064a \u0623\u0646\u062a\u062c\u0647.</strong> \u0635\u0641\u0648\u0641 <em>\u0645\u0642\u064a\u0633 \u0647\u0646\u0627</em> \u0641\u064a \u0644\u0648\u062d\u0629 \u0627\u0644\u062a\u0631\u062a\u064a\u0628 \u062a\u0623\u062a\u064a \u0645\u0646 \u0645\u0642\u064a\u0651\u0645 \u0627\u0644\u0645\u0639\u064a\u0627\u0631 \u0639\u0644\u0649 \u0645\u0633\u062a\u0648\u0649 \u0627\u0644\u062d\u0631\u0641\u060c \u0645\u0642\u064a\u0651\u0633\u0629 \u0639\u0644\u0649 1200 \u0633\u0637\u0631. \u0623\u0645\u0651\u0627 \u062a\u0628\u0648\u064a\u0628 \u0627\u0644\u0642\u064a\u0627\u0633 \u0641\u064a\u0634\u063a\u0651\u0644 \u0645\u0642\u064a\u0651\u0645\u064b\u0627 \u0645\u0633\u062a\u0642\u0644\u0651\u064b\u0627 \u062f\u0627\u062e\u0644 \u0647\u0630\u0647 \u0627\u0644\u0635\u0641\u062d\u0629 - DER \u0639\u0644\u0649 \u0645\u0633\u062a\u0648\u0649 \u0627\u0644\u062d\u0631\u0641 \u0641\u0648\u0642 \u0627\u0644\u062d\u0631\u0648\u0641 \u0627\u0644\u0645\u0634\u0643\u0648\u0644\u0629\u060c \u0648 WER \u0639\u0644\u0649 \u0645\u0633\u062a\u0648\u0649 \u0627\u0644\u0643\u0644\u0645\u0629 - \u0648\u064a\u0642\u0648\u0644 \u0630\u0644\u0643 \u0639\u0644\u0649 \u0627\u0644\u062a\u0628\u0648\u064a\u0628 \u0646\u0641\u0633\u0647. \u0644\u0627 \u064a\u064f\u0636\u0639 \u0634\u064a\u0621 \u0641\u064a \u0639\u0645\u0648\u062f \u0644\u0627 \u064a\u0646\u062a\u0645\u064a \u0625\u0644\u064a\u0647.",
  },
};

/** Sample-text names are keyed by their English name, which is the stable id. */
const SAMPLE_AR = {
  "Modern Standard Arabic": "\u0639\u0631\u0628\u064a\u0629 \u0641\u0635\u062d\u0649 \u0645\u0639\u0627\u0635\u0631\u0629",
  "Classical prose": "\u0646\u062b\u0631 \u0643\u0644\u0627\u0633\u064a\u0643\u064a",
  "Hard morphology (geminates, hamza, tanwin)": "\u0635\u0631\u0641 \u0635\u0639\u0628 (\u0627\u0644\u0634\u062f\u0651\u0629 \u0648\u0627\u0644\u0647\u0645\u0632\u0629 \u0648\u0627\u0644\u062a\u0646\u0648\u064a\u0646)",
  "Mixed script and digits (passthrough)": "\u0646\u0635 \u0645\u062e\u062a\u0644\u0637 \u0648\u0623\u0631\u0642\u0627\u0645 (\u064a\u064f\u0645\u0631\u0651\u0631 \u0643\u0645\u0627 \u0647\u0648)",
  "Multi-paragraph (for the Bulk tab)": "\u0639\u062f\u0629 \u0641\u0642\u0631\u0627\u062a (\u0644\u062a\u0628\u0648\u064a\u0628 \u0627\u0644\u0646\u0635 \u0627\u0644\u0637\u0648\u064a\u0644)",
};

let lang = "en";
const listeners = [];

export function getLang() { return lang; }
export function isRTL() { return lang === "ar"; }

/** The visitor's preference, else the browser's, else English. */
export function detectLang() {
  try {
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (LANGS.indexOf(saved) !== -1) return saved;
  } catch (e) { /* private mode */ }
  const nav = (navigator.languages && navigator.languages[0]) || navigator.language || "";
  return /^ar\b/i.test(nav) ? "ar" : "en";
}

export function t(key, vars) {
  let s = (DICT[lang] && DICT[lang][key]);
  if (s === undefined) s = DICT.en[key];
  if (s === undefined) return key;
  if (vars) s = s.replace(/\{(\w+)\}/g, (m, k) => (vars[k] !== undefined ? vars[k] : m));
  return s;
}

/** Translate a sample's English name; unknown names pass through unchanged. */
export function sampleName(name) {
  if (lang !== "ar") return name;
  return SAMPLE_AR[name] || name;
}

/**
 * Model names are repo identifiers and stay verbatim, because they are what a
 * reader will type into a search box. The one exception is our own "(ours)"
 * suffix, which is a claim about ownership rather than part of the name - in
 * Arabic it is dropped and the "ours" badge carries the meaning.
 */
export function modelName(label) {
  if (lang !== "ar") return label;
  return String(label).replace(/\s*\(ours\)$/, "");
}

export function modelLabel(spec) { return modelName(spec.label); }

export function onLangChange(fn) { listeners.push(fn); }

export function setLang(next, persist) {
  if (LANGS.indexOf(next) === -1) next = "en";
  lang = next;
  if (persist !== false) {
    try { window.localStorage.setItem(STORAGE_KEY, next); } catch (e) { /* ignore */ }
  }
  applyDocument();
  applyStatic(document);
  for (const fn of listeners) fn(next);
}

/** <html lang/dir> plus the two strings that live outside the DOM body. */
export function applyDocument() {
  const root = document.documentElement;
  root.setAttribute("lang", lang);
  root.setAttribute("dir", isRTL() ? "rtl" : "ltr");
  document.title = t("doc.title");
  const meta = document.querySelector('meta[name="description"]');
  if (meta) meta.setAttribute("content", t("doc.description"));
}

/**
 * Fill every element that declares what it wants translated:
 *   data-i18n       -> textContent
 *   data-i18n-html  -> innerHTML (only ever our own dictionary strings)
 *   data-i18n-ph    -> placeholder attribute
 *   data-i18n-aria  -> aria-label attribute
 */
export function applyStatic(root) {
  const scope = root || document;
  for (const n of scope.querySelectorAll("[data-i18n]")) n.textContent = t(n.getAttribute("data-i18n"));
  for (const n of scope.querySelectorAll("[data-i18n-html]")) n.innerHTML = t(n.getAttribute("data-i18n-html"));
  for (const n of scope.querySelectorAll("[data-i18n-ph]")) n.setAttribute("placeholder", t(n.getAttribute("data-i18n-ph")));
  for (const n of scope.querySelectorAll("[data-i18n-aria]")) n.setAttribute("aria-label", t(n.getAttribute("data-i18n-aria")));
}

/**
 * Wire the masthead switch. The pressed state is exposed to assistive tech, and
 * every other module re-renders by subscribing through onLangChange().
 */
export function initSwitch(host) {
  if (!host) return;
  const buttons = Array.from(host.querySelectorAll("[data-lang]"));
  const sync = () => {
    for (const b of buttons) b.setAttribute("aria-pressed", String(b.getAttribute("data-lang") === lang));
  };
  for (const b of buttons) {
    b.addEventListener("click", () => setLang(b.getAttribute("data-lang")));
  }
  onLangChange(sync);
  sync();
}

export function initLang() {
  setLang(detectLang(), false);
}
