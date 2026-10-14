"""Downloader: config loading, gated-source refusal, checksum helper."""
import importlib.util
import hashlib
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent


def load_cli():
    spec = importlib.util.spec_from_file_location(
        "download_data", REPO / "scripts" / "download_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def dl():
    return load_cli()


def test_the_shipped_catalog_loads(dl):
    cfg = dl.load_config(REPO / "configs" / "data.yaml")
    assert "fadel" in cfg["catalog"]
    assert cfg["catalog"]["sadeed_tashkeela"]["requires_acceptance"] is True


def test_a_config_without_a_catalog_is_refused(dl, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("raw_dir: data/raw\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        dl.load_config(bad)


def test_a_missing_config_is_refused(dl, tmp_path):
    with pytest.raises(SystemExit):
        dl.load_config(tmp_path / "nope.yaml")


def test_a_gated_source_is_blocked_without_acceptance(dl, tmp_path):
    status, rec = dl.fetch_one(
        "gated", {"kind": "hf", "repo": "someone/thing", "license": "GPL-2.0",
                  "commercial_use": "no", "requires_acceptance": True},
        tmp_path, token=None, accept_license=False, force=False)
    assert status == "BLOCKED"
    assert rec["status"] == "BLOCKED"
    assert rec["accepted"] is False
    assert list(tmp_path.iterdir()) == []          # nothing was fetched


def test_an_unknown_kind_is_refused(dl, tmp_path):
    with pytest.raises(SystemExit):
        dl.fetch_one("weird", {"kind": "ftp"}, tmp_path, token=None,
                     accept_license=True, force=False)


def test_sha256_file_matches_hashlib(dl, tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"tashkeel")
    assert dl.sha256_file(p) == hashlib.sha256(b"tashkeel").hexdigest()


def test_every_http_source_names_its_license_in_the_record(dl, tmp_path):
    cfg = dl.load_config(REPO / "configs" / "data.yaml")
    for sid, spec in cfg["catalog"].items():
        assert isinstance(spec.get("license"), str) and spec["license"].strip(), sid
