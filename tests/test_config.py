import pytest

from smart_llm.config import Config, load_config, _apply_dotted, _coerce


def test_defaults_and_prepare(tmp_path):
    cfg = Config()
    cfg.paths.root = str(tmp_path / "runs")
    cfg.prepare()
    assert cfg.paths.cache_dir.endswith("cache")
    assert cfg.n_demos == cfg.retrieval.k


def test_coerce_types():
    assert _coerce("true") is True
    assert _coerce("none") is None
    assert _coerce("8") == 8
    assert _coerce("0.5") == 0.5
    assert _coerce("a,b") == ["a", "b"]


def test_dotted_override():
    cfg = Config()
    _apply_dotted(cfg, "router.alpha=0.7")
    _apply_dotted(cfg, "data.dataset=agnews")
    assert cfg.router.alpha == 0.7
    assert cfg.data.dataset == "agnews"


def test_unknown_key_raises():
    cfg = Config()
    with pytest.raises(KeyError):
        _apply_dotted(cfg, "router.nonexistent=1")


def test_load_config_overrides(tmp_path):
    cfg = load_config(None, ["seed=123", "probe.epochs=5"])
    assert cfg.seed == 123
    assert cfg.probe.epochs == 5
