import numpy as np

from smart_llm.config import Config
from smart_llm.cdka.probe import ProbeData, fit_probe
from smart_llm.cdka.rbe import RBEData, fit_rbe
from smart_llm.cdka.calibration import Calibrator
from smart_llm.cdka.router import Router, uncertainty
from smart_llm.utils.seed import seed_everything


def _fast_cfg():
    seed_everything(0)          # deterministic torch weight init for stochastic tests
    cfg = Config()
    cfg.probe.epochs = 40
    cfg.probe.hidden_dims = []
    cfg.rbe.epochs = 60
    cfg.rbe.hidden_dims = [32]
    cfg.router.tune_grid = 6
    return cfg


def _separable(n=300, dim=16, k=3, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.normal(scale=3.0, size=(k, dim)).astype(np.float32)
    y = rng.integers(0, k, size=n)
    x = (means[y] + rng.normal(scale=0.6, size=(n, dim))).astype(np.float32)
    return x, y.astype(np.int64)


def test_probe_learns_and_confidence_valid():
    cfg = _fast_cfg()
    x, y = _separable()
    tr = ProbeData(x=x[:220], y=y[:220])
    va = ProbeData(x=x[220:], y=y[220:])
    probe, _ = fit_probe(tr, va, dim=x.shape[1], n_classes=3, pooling="mean", cfg=cfg)
    pred = probe.predict(x)
    acc = float(np.mean(pred["pred"] == y))
    assert acc > 0.7
    assert np.all(pred["confidence"] >= 0) and np.all(pred["confidence"] <= 1.0001)
    assert np.all(pred["entropy"] >= -1e-6) and np.all(pred["entropy"] <= 1.0001)


def test_rbe_regresses_linear_signal():
    cfg = _fast_cfg()
    rng = np.random.default_rng(1)
    x = rng.normal(size=(400, 24)).astype(np.float32)
    w = rng.normal(size=24)
    b_true = (x @ w * 0.3 + rng.normal(scale=0.1, size=400)).astype(np.float32)
    tr = RBEData(x=x[:320], b_true=b_true[:320])
    va = RBEData(x=x[320:], b_true=b_true[320:])
    rbe, hist = fit_rbe(tr, va, cfg)
    assert hist["val_r2"][-1] > 0.4


def test_calibration_platt_and_degenerate():
    rng = np.random.default_rng(2)
    rus = rng.normal(size=200)
    target = (rus > 0).astype(int)
    cal = Calibrator("platt").fit(rus, target)
    p = cal.transform(rus)
    assert np.all((p >= 0) & (p <= 1))
    # higher RUS -> higher calibrated probability on average
    assert p[rus > 0.5].mean() > p[rus < -0.5].mean()
    # degenerate (all one class) falls back to minmax, still in [0,1]
    cal2 = Calibrator("platt").fit(rus, np.ones_like(target))
    assert cal2.method == "minmax"
    assert np.all((cal2.transform(rus) >= 0) & (cal2.transform(rus) <= 1))


def test_router_fits_and_agrees_with_oracle():
    cfg = _fast_cfg()
    rng = np.random.default_rng(3)
    n = 500
    sim = rng.uniform(0, 1, n)
    bpred = rng.normal(size=n)
    ci = rng.uniform(0, 1, n)
    # oracle strongly driven by predicted benefit
    oracle = (bpred + 0.2 * sim - 0.1 > 0).astype(int)
    router = Router(cfg).fit(sim, bpred, ci, oracle)
    out = router.predict(sim, bpred, ci)
    agree = float(np.mean(out["decision"] == oracle))
    assert agree > 0.6
    assert 0.0 <= router.alpha <= 1.0
    assert abs(router.alpha + router.beta - 1.0) < 1e-6


def test_uncertainty_range():
    u = uncertainty(np.array([0.0, 0.5, 1.0]), np.array([1.0, 0.5, 0.0]), lam=0.5)
    assert np.all((u >= 0) & (u <= 1))
    assert u[0] < u[2]  # low entropy + high conf => low uncertainty
