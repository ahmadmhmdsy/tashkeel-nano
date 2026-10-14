"""Tokenizer: fixed small vocabulary, per-code-point encoding, exact round trip."""
from tashkeel_nano import tokenizer as TK


def test_vocab_is_small_and_fixed():
    assert TK.VOCAB_SIZE == 171


def test_specials_exist():
    for name in ("PAD", "BOS", "EOS", "UNK"):
        assert hasattr(TK, name), name
    assert len({TK.PAD, TK.BOS, TK.EOS, TK.UNK}) == 4


def test_round_trip_is_identity_on_printable_text():
    for s in ("abc XYZ 123", "مرحبا بك", "mixed latin/عربي 42", "!? .,:;-()[]"):
        assert TK.decode(TK.encode(s)) == s


def test_arabic_base_block_is_in_the_vocabulary():
    for cp in range(0x0621, 0x064B):
        assert chr(cp) in TK.ID_OF, hex(cp)


def test_encoding_is_per_code_point():
    for s in ("مرحبا", "abc", "م1x"):
        assert len(TK.encode(s)) == len(s)


def test_empty_string_encodes_to_nothing():
    assert TK.encode("") == []
