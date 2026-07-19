import numpy as np

from smart_llm.analysis import metrics as M


def test_r2_perfect_and_mean():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    assert M.r2_score(y, y) == 1.0
    # predicting the mean gives R2 == 0
    assert abs(M.r2_score(np.full_like(y, y.mean()), y)) < 1e-9


def test_accuracy_and_macro_f1():
    pred = np.array([0, 1, 2, 2])
    label = np.array([0, 1, 2, 1])
    assert M.accuracy(pred, label) == 0.75
    f1 = M.macro_f1(pred, label, n_classes=3)
    assert 0.0 <= f1 <= 1.0


def test_macro_f1_perfect():
    y = np.array([0, 1, 2, 0, 1, 2])
    assert abs(M.macro_f1(y, y, 3) - 1.0) < 1e-9


def test_routing_agreement_and_regret():
    dec = np.array([1, 0, 1, 0])
    orc = np.array([1, 0, 0, 0])
    assert M.routing_agreement(dec, orc) == 0.75
    loss_p = np.array([1.0, 2.0, 1.0, 2.0])
    loss_r = np.array([0.5, 3.0, 2.0, 1.0])
    reg = M.regret_per_sample(dec, loss_p, loss_r)
    assert np.all(reg >= 0)
    # sample 0: chose retrieval (0.5) which is oracle -> regret 0
    assert reg[0] == 0.0
    # sample 2: chose retrieval (2.0) but parametric (1.0) better -> regret 1.0
    assert reg[2] == 1.0


def test_retrieval_frequency():
    assert M.retrieval_frequency(np.array([1, 1, 0, 0])) == 0.5


def test_ece_bounds():
    conf = np.array([0.9, 0.8, 0.6, 0.4])
    correct = np.array([1.0, 1.0, 0.0, 0.0])
    e = M.expected_calibration_error(conf, correct, n_bins=5)
    assert 0.0 <= e <= 1.0
