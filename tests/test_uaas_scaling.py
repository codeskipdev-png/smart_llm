import numpy as np

from smart_llm.uaas.scaling import continuous_rank, bucket_rank, rank_for_uncertainty


def test_continuous_rank_endpoints():
    r = continuous_rank(np.array([0.0, 1.0, 0.5]), r_min=4, r_max=32)
    assert r[0] == 4 and r[1] == 32 and abs(r[2] - 18) < 1e-6


def test_continuous_rank_clips():
    r = continuous_rank(np.array([-1.0, 2.0]), r_min=4, r_max=32)
    assert r[0] == 4 and r[1] == 32


def test_bucket_rank_snaps_to_nearest():
    b = bucket_rank(np.array([5.0, 15.0, 30.0]), buckets=[4, 8, 16, 24, 32])
    assert list(b) == [4, 16, 32]


def test_rank_monotonic_in_uncertainty():
    u = np.linspace(0, 1, 20)
    r = rank_for_uncertainty(u, 4, 32, [4, 8, 16, 24, 32])
    assert r[0] <= r[-1]
    assert np.all(np.diff(r.astype(float)) >= 0)  # non-decreasing
