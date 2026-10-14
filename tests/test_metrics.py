"""Metrics: word-level DER semantics, multi-reference, and library parity."""
import pytest

from tashkeel_nano import eval_der as EV
from tashkeel_nano import eval_metrics as EM

W = "\u0628\u0650\u0633\u0652\u0645"                     # bi-s-m
W_ALT = W + "/" + "\u0628\u064e\u0633\u0652\u0645"       # + ba-s-m
W2 = "\u0627\u0644\u0644\u0651\u064e\u0647\u0650"     # al-lah+i
W3 = "\u0645\u0641\u062a\u0627\u062d"                    # bare


def test_perfect_prediction_scores_zero():
    text = W + " " + W2
    m = EV.score_line(text, [text])
    assert m["DER"] == 0 and m["WER"] == 0 and m["DER_nocase"] == 0
    assert m["text_preservation"] == 1.0


def test_one_wrong_vowel_costs_exactly_one_word():
    pred = "\u0628\u064e\u0633\u0652\u0645 " + W2
    m = EV.score_line(pred, [W + " " + W2])
    assert m["words"] == 2
    assert m["DER"] == pytest.approx(0.5)


def test_multi_reference_matches_any_alternative():
    # "/" inside a word is a set of accepted diacritizations: der_anyalt counts
    # the word correct when it matches ANY of them, while der_primary only
    # accepts the reference's FIRST alternative.
    alt = "\u0628\u064e\u0633\u0652\u0645"
    summary = EM.summarize(EM.score_corpus([alt], [W + "/" + alt]))
    assert summary["der_anyalt"] == 0.0
    assert summary["der_canonical"] == 0.0
    assert summary["der_primary"] == pytest.approx(1.0)
    assert summary["der_ambiguous"] == 0.0
    assert summary["ambiguous_words"] == 1


def test_registered_is_the_strict_whole_token_anchor():
    # der_registered mirrors eval_der/the registered gate exactly: the reference
    # token is compared WHOLE, so a token carrying "/" is correct only if the
    # prediction reproduces that literal string. It is the parity anchor, not
    # the alternative-aware view -- use der_anyalt for that. The two therefore
    # differ on the ~0.3-0.6 pct of gate tokens that carry "/".
    alt = "\u0628\u064e\u0633\u0652\u0645"
    summary = EM.summarize(EM.score_corpus([alt], [W + "/" + alt]))
    assert summary["der_registered"] == pytest.approx(1.0)
    assert summary["der_anyalt"] == 0.0


def test_case_ending_error_shows_in_der_but_not_der_nocase():
    gold = "text \u0645\u0641\u062a\u0627\u062d\u0650"
    pred = "text \u0645\u0641\u062a\u0627\u062d"
    m = EV.score_line(pred, [gold])
    assert m["DER"] > m["DER_nocase"]


def test_base_drift_trips_the_preservation_gate():
    # a wrong LETTER is base drift (fatal); a wrong VOWEL is not
    assert EV.score_line("\u0645\u0641\u062a\u0627", ["\u0645\u0641\u062a\u0627\u062d"])["text_preservation"] == 0.0
    assert EV.score_line("\u0645\u0641\u062a\u0627\u062d", ["\u0645\u0641\u062a\u0627\u062d\u0650"])["text_preservation"] == 1.0


def test_length_mismatch_is_counted_never_silently_shrunk():
    m = EV.score_line("word1 word2", ["a b c"])
    assert m["words"] == 3
    assert m["DER"] > 0


@pytest.mark.parametrize("pred,ref", [
    ([W + " " + W2], [W + " " + W2]),
    ([W], [W_ALT]),
    ([W3 + " " + W2], [W + " " + W2]),
    ([W], [W + " " + W2]),
    ([W + " " + W2 + " " + W3], [W + " " + W2]),
])
def test_eval_metrics_agrees_with_eval_der_word_for_word(pred, ref):
    scored = EM.score_corpus(pred, ref)
    summary = EM.summarize(scored)
    total = sum(max(len(r.split()), len(p.split()))
                for p, r in zip(pred + [""] * len(ref), ref))
    assert scored["record"]["words"] == total
    if all("/" not in r for r in ref):
        # eval_der implements the same word-level rule, so the two must agree
        assert summary["der_registered"] == pytest.approx(
            EV.aggregate([EV.score_line(p, [r]) for p, r in zip(pred, ref)])["DER"])


def test_corpus_rate_is_a_ratio_of_totals_not_a_mean_of_line_rates():
    long_ok = " ".join([W] * 100)
    summary = EM.summarize(EM.score_corpus([W3, long_ok], [W, long_ok]))
    assert summary["der_anyalt"] == pytest.approx(1.0 / 101.0)


def test_canonical_comparison_closes_the_shadda_order_gap():
    shadda_first = "\u0628\u0651\u064e"
    vowel_first = "\u0628\u064e\u0651"
    plain = "\u0628\u064e"
    assert EV.marks_of(shadda_first) != EV.marks_of(vowel_first)
    assert not EV._pairs_ok(shadda_first, vowel_first, nocase=False)
    assert EM.marks_equal_canonical(shadda_first, vowel_first)
    assert not EM.marks_equal_canonical(shadda_first, plain)


def test_confusion_matrix_is_15_class():
    summary = EM.summarize(EM.score_corpus([W + " " + W2], [W + " " + W2]))
    assert EM.N_CLASSES == 15
    assert len(summary["confusion"]) == 15
    assert len(summary["per_class"]) == 15
    assert summary["position_accuracy"] == pytest.approx(1.0)
