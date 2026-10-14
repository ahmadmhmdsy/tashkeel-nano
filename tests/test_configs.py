"""Configs: every YAML parses, every model block builds, catalog is well-formed."""
import importlib.util
from pathlib import Path

import pytest
import yaml

from tashkeel_nano import tokenizer as TK
from tashkeel_nano.model import build_from_config

REPO = Path(__file__).resolve().parent.parent
ALL_CONFIGS = sorted((REPO / "configs").rglob("*.yaml"))
DATA_CONFIG = REPO / "configs" / "data.yaml"
CONFIGS = [p for p in ALL_CONFIGS
           if isinstance(yaml.safe_load(p.read_text(encoding="utf-8")), dict)
           and "model" in yaml.safe_load(p.read_text(encoding="utf-8"))]

PUBLISHED = {
    "nano1p1m.yaml": 1083328,
    "nano4p5m.yaml": 4488160,
}


def test_there_are_configs_to_check():
    assert len(CONFIGS) >= 6
    assert len(CONFIGS) < len(ALL_CONFIGS)


def test_the_data_config_declares_a_catalog_not_a_model():
    cfg = yaml.safe_load(DATA_CONFIG.read_text(encoding="utf-8"))
    assert "model" not in cfg
    assert cfg["catalog"]


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: str(p.relative_to(REPO)))
def test_yaml_parses_and_declares_a_model(path):
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(cfg, dict) and cfg
    assert "model" in cfg, path


@pytest.mark.parametrize("name,expected", sorted(PUBLISHED.items()))
def test_published_shapes_build_to_the_published_parameter_counts(name, expected):
    cfg = yaml.safe_load((REPO / "configs" / name).read_text(encoding="utf-8"))
    model = build_from_config(cfg, vocab_size=TK.VOCAB_SIZE)
    assert sum(p.numel() for p in model.parameters()) == expected


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: str(p.relative_to(REPO)))
def test_every_model_block_is_buildable(path):
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        build_from_config(cfg, vocab_size=TK.VOCAB_SIZE)
    except KeyError as exc:
        pytest.fail("config %s is missing model key %s" % (path.name, exc))


def test_data_catalog_is_well_formed():
    cfg = yaml.safe_load((REPO / "configs" / "data.yaml").read_text(encoding="utf-8"))
    catalog = cfg["catalog"]
    assert catalog and cfg.get("default_sources")
    for sid, spec in catalog.items():
        assert spec.get("kind") in ("http", "hf", "git"), sid
        assert spec.get("license"), sid
        assert "commercial_use" in spec, sid
        if spec.get("kind") == "http":
            assert spec.get("files"), sid
            assert all(u.startswith("http") for u in spec["files"].values()), sid
        if spec.get("kind") in ("hf", "git"):
            assert spec.get("repo"), sid
