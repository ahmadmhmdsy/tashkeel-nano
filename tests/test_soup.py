"""Weight soup: uniform fp32 mean, dtype preserved, members recorded."""
import importlib.util
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent


def load_cli(name):
    spec = importlib.util.spec_from_file_location(
        name, REPO / "scripts" / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def soup_cli():
    return load_cli("soup")


def make_probe(path, value, dtype=torch.float32):
    torch.save({"model": {"w": torch.full((4, 4), float(value), dtype=dtype)},
                "step": 0, "config": {"model": {}}}, path)
    return path


def test_probe_step_reads_the_step_from_the_filename(soup_cli):
    assert soup_cli.probe_step(Path("weights_step32500.pt")) == 32500
    assert soup_cli.probe_step(Path("nope.pt")) == -1


def test_parse_members_expands_a_range(soup_cli, tmp_path):
    for step in (1000, 2000, 3000, 4000):
        make_probe(tmp_path / ("weights_step%d.pt" % step), step)
    got = soup_cli.parse_members("1000:4000:1000", tmp_path)
    assert [soup_cli.probe_step(p) for p in got] == [1000, 2000, 3000, 4000]
    assert len(soup_cli.parse_members("all", tmp_path)) == 4
    assert len(soup_cli.parse_members(None, tmp_path)) == 4
    assert [soup_cli.probe_step(p) for p in soup_cli.parse_members("2000,4000", tmp_path)] == [2000, 4000]


def test_parse_members_rejects_an_absent_step(soup_cli, tmp_path):
    make_probe(tmp_path / "weights_step1000.pt", 1000)
    with pytest.raises(SystemExit):
        soup_cli.parse_members("1000:3000:1000", tmp_path)


def test_soup_is_the_elementwise_mean_and_keeps_the_member_dtype(soup_cli, tmp_path):
    probes = [make_probe(tmp_path / ("weights_step%d.pt" % (i * 1000)), i, dtype=torch.float16)
              for i in (1, 2, 3)]
    out, stats, dists = soup_cli.soup(probes, tmp_path / "out" / "model.pt", "t")
    sd = torch.load(out, map_location="cpu", weights_only=False)
    assert torch.equal(sd["model"]["w"], torch.full((4, 4), 2.0, dtype=torch.float16))
    assert sd["model"]["w"].dtype == torch.float16
    assert sd["step"] == "soup_t_3"
    assert stats["members"] == 3
    assert len(dists) == 3
    # rel_L2 from the mean of {1,2,3} to each member: 1.0, 0.0, 1/3
    assert sorted(round(d, 6) for d in dists) == [0.0, 0.333333, 1.0]


def test_soup_of_a_single_member_is_that_member(soup_cli, tmp_path):
    probe = make_probe(tmp_path / "weights_step1000.pt", 7.0)
    out, _, _ = soup_cli.soup([probe], tmp_path / "one" / "model.pt", "one")
    sd = torch.load(out, map_location="cpu", weights_only=False)
    assert torch.equal(sd["model"]["w"], torch.full((4, 4), 7.0))
