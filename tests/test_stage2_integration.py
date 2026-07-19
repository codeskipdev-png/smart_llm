"""End-to-end Stage-2 + analysis on a SYNTHETIC feature cache (no LLM/FAISS).

Exercises FeatureWriter -> load_features -> train_cdka.run -> tables/figures on
the CPU box, validating the full CDKA training + logging + analysis path.
"""
import numpy as np
import pandas as pd

from smart_llm.config import Config
from smart_llm.experiments.cache import FeatureWriter, load_features
from smart_llm.experiments import train_cdka
from smart_llm.experiments.train_cdka import MASTER_COLUMNS


DIM, EDIM, NTOK, K, NCLS, N = 32, 16, 8, 4, 3, 150
CONDS = ["clean", "random", "adversarial"]


def _build_cache(cfg):
    rng = np.random.default_rng(0)
    means = rng.normal(scale=2.0, size=(NCLS, DIM)).astype(np.float32)
    label_names = [f"class{i}" for i in range(NCLS)]
    w = FeatureWriter(cfg.paths.cache_dir, cfg.data.dataset, CONDS, label_names,
                      shard_size=40)
    for i in range(N):
        y = int(rng.integers(0, NCLS))
        h = (means[y] + rng.normal(scale=0.7, size=DIM)).astype(np.float32)
        scalars = dict(id=f"s{i}", label=y, pred_p=int(rng.integers(0, NCLS)),
                       loss_p=float(abs(rng.normal()) + 0.2),
                       conf_llm=float(rng.uniform(0.3, 0.99)),
                       entropy_llm=float(rng.uniform(0, 1)),
                       n_prompt_tokens=int(rng.integers(20, 100)),
                       t_p=float(rng.uniform(0.05, 0.1)))
        vecs = dict(h_last=h, h_mean=(h + rng.normal(scale=0.1, size=DIM)).astype(np.float32),
                    query_emb=rng.normal(size=EDIM).astype(np.float32))
        tokens = dict(h_tokens=rng.normal(size=(NTOK, DIM)).astype(np.float32),
                      token_mask=np.ones(NTOK, dtype=np.int64))
        conds = {}
        for c in CONDS:
            benefit = rng.normal(scale=0.5)
            loss_r = float(max(0.01, scalars["loss_p"] - benefit))
            conds[c] = dict(
                centroid=rng.normal(size=EDIM).astype(np.float32),
                sim=float(rng.uniform(0, 1)), loss_r=loss_r,
                pred_r=int(rng.integers(0, NCLS)),
                btrue=float((scalars["loss_p"] - loss_r) / (abs(scalars["loss_p"]) + 1e-6)),
                oracle=int(loss_r < scalars["loss_p"]),
                retr_idx=rng.integers(0, 100, size=K).astype(np.int64),
                demo_label_ids=rng.integers(0, NCLS, size=K).astype(np.int64),
                t_r=float(rng.uniform(0.08, 0.15)),
                n_tokens_r=int(rng.integers(120, 400)))
        w.add(scalars, vecs, tokens, conds)
    w.set_timing({"embed_query_time": 1.0, "search_time": {c: 0.1 for c in CONDS},
                  "n_eval_for_amortization": N})
    w.finalize()


def _cfg(tmp_path):
    cfg = Config()
    cfg.paths.root = str(tmp_path / "runs")
    cfg.data.dataset = "synthetic"
    cfg.probe.epochs = 8
    cfg.rbe.epochs = 12
    cfg.rbe.hidden_dims = [32]
    cfg.router.tune_grid = 5
    cfg.prepare()
    return cfg


def test_cache_roundtrip(tmp_path):
    cfg = _cfg(tmp_path)
    _build_cache(cfg)
    feats = load_features(cfg.paths.cache_dir, "synthetic", need_tokens=True)
    assert feats.n == N
    assert feats.h_last.shape == (N, DIM)
    assert feats.h_tokens.shape == (N, NTOK, DIM)
    assert set(feats.conds) == set(CONDS)
    assert feats.conds["clean"]["centroid"].shape == (N, EDIM)


def test_stage2_produces_master_csv(tmp_path):
    cfg = _cfg(tmp_path)
    _build_cache(cfg)
    train_cdka.run(cfg, device="cpu")

    df = pd.read_csv(f"{cfg.paths.results_dir}/master_synthetic.csv")
    assert list(df.columns) == MASTER_COLUMNS
    # one row per (pooling x condition x sample)
    assert len(df) == len(cfg.pooling.types) * len(CONDS) * N
    for col in ["C_i", "B_pred", "RUS", "delta_C", "smart_decision"]:
        assert df[col].notna().all()
    assert set(df["smart_decision"].unique()).issubset({0, 1})


def test_analysis_tables_build(tmp_path):
    cfg = _cfg(tmp_path)
    _build_cache(cfg)
    train_cdka.run(cfg, device="cpu")
    from smart_llm.analysis import tables
    tbls = tables.build_all(cfg)
    assert len(tbls["table1_main"]) == 3           # 3 systems
    assert "Accuracy" in tbls["table1_main"].columns
    assert len(tbls["table4_noise"]) == len(CONDS)
    assert len(tbls["table_difficulty"]) >= 1      # difficulty tiers
    assert "Precision" in tbls["table2_router_oracle"].columns
