"""normalize_marks.py: NFC reordering of prediction files."""
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def load_cli():
    spec = importlib.util.spec_from_file_location(
        "normalize_marks", REPO / "scripts" / "normalize_marks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def nm():
    return load_cli()


def test_normalize_undoes_the_shadda_first_render(nm):
    from tashkeel_nano.inference import reorder_shadda_first

    nfc = "\u0628\u064e\u0651\u0645 \u0627\u0644\u0644\u064e\u0651\u0647\u0650"
    assert reorder_shadda_first(nfc) != nfc
    assert nm.normalize(reorder_shadda_first(nfc)) == nfc


def test_normalize_leaves_letters_and_spacing_alone(nm):
    text = "\u0645\u0641\u062a\u0627\u062d \u0661\u0662\u0663\n"
    assert nm.normalize(text) == text


def test_main_rewrites_in_place_and_says_so(nm, tmp_path, capsys):
    src = tmp_path / "fadel_test.pred.txt"
    src.write_text("\u0628\u0651\u064e\u0645\n", encoding="utf-8", newline="")
    assert nm.main(["--dir", str(tmp_path)]) == 0
    assert src.read_text(encoding="utf-8") == "\u0628\u064e\u0651\u0645\n"
    assert "changed" in capsys.readouterr().out


def test_main_can_write_to_a_separate_directory(nm, tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    (src / "a.pred.txt").write_text("\u0628\u0651\u064e", encoding="utf-8", newline="")
    (src / "b.ref.txt").write_text("\u0628\u0651\u064e", encoding="utf-8", newline="")
    dst = tmp_path / "out"
    assert nm.main(["--dir", str(src), "--out-dir", str(dst)]) == 0
    assert (dst / "a.pred.txt").read_text(encoding="utf-8") == "\u0628\u064e\u0651"
    assert not (dst / "b.ref.txt").exists()
    assert (src / "a.pred.txt").read_text(encoding="utf-8") == "\u0628\u0651\u064e"


def test_a_directory_without_prediction_files_is_an_error(nm, tmp_path):
    with pytest.raises(SystemExit):
        nm.main(["--dir", str(tmp_path)])
