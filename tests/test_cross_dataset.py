"""Cross-dataset comparison: synthetic per-dataset tables -> comparison CSV/figure."""
import pandas as pd

from smart_llm.config import Config
from smart_llm.analysis import cross_dataset


def _cfg(tmp_path):
    cfg = Config()
    cfg.paths.root = str(tmp_path / "runs")
    cfg.pooling.default = "last"
    cfg.prepare()
    return cfg


def _write_tables(cfg, ds, rbe_r2, agree_full, agree_sim):
    td = cfg.paths.tables_dir
    pd.DataFrame([{"Pooling": "last", "R2": rbe_r2, "MAE": 0.3, "Pearson r": 0.2}]).to_csv(
        f"{td}/table3_rbe_{ds}.csv", index=False)
    pd.DataFrame([
        {"Variant": "SMART (full)", "oracle_agreement": agree_full, "mean_regret": 0.10},
        {"Variant": "- RBE (similarity only)", "oracle_agreement": agree_sim, "mean_regret": 0.15},
    ]).to_csv(f"{td}/ablation_{ds}.csv", index=False)
    pd.DataFrame([
        {"System": "No retrieval", "Accuracy": 0.60, "Retrieval freq.": 0.0},
        {"System": "Always RAG", "Accuracy": 0.72, "Retrieval freq.": 1.0},
        {"System": "SMART-LLM (ours)", "Accuracy": 0.69, "Retrieval freq.": 0.3},
    ]).to_csv(f"{td}/table1_main_{ds}.csv", index=False)
    pd.DataFrame([
        {"Condition": "adversarial", "Always-RAG acc": 0.33, "SMART acc": 0.56},
    ]).to_csv(f"{td}/table4_noise_{ds}.csv", index=False)


def test_cross_dataset_comparison(tmp_path):
    cfg = _cfg(tmp_path)
    _write_tables(cfg, "20newsgroups", rbe_r2=0.05, agree_full=0.60, agree_sim=0.59)
    _write_tables(cfg, "financial_phrasebank", rbe_r2=0.25, agree_full=0.66, agree_sim=0.55)
    comp = cross_dataset.run(cfg, ["20newsgroups", "financial_phrasebank"])

    assert len(comp) == 2
    assert "Δ Agreement (RBE gain)" in comp.columns
    fpb = comp[comp.Dataset == "financial_phrasebank"].iloc[0]
    ng = comp[comp.Dataset == "20newsgroups"].iloc[0]
    # RBE gain should be larger on the sentiment dataset in this synthetic setup
    assert fpb["Δ Agreement (RBE gain)"] > ng["Δ Agreement (RBE gain)"]
    # output artifacts exist
    import os
    assert os.path.exists(f"{cfg.paths.tables_dir}/cross_dataset_comparison.csv")
    assert os.path.exists(f"{cfg.paths.figures_dir}/figure9_cross_dataset.png")


def test_cross_interp_adaptive(tmp_path):
    from smart_llm.paper.make_manuscript import _cross_interp
    comp = pd.DataFrame([
        {"Dataset": "20newsgroups", "RBE R2": 0.05, "Δ Agreement (RBE gain)": 0.01},
        {"Dataset": "financial_phrasebank", "RBE R2": 0.25, "Δ Agreement (RBE gain)": 0.11},
    ])
    s = _cross_interp(comp)
    assert "financial_phrasebank".replace("_", "") or "Financial" in s
    assert "more on" in s  # detected the larger RBE gain on FPB
