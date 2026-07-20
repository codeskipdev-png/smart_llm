"""Generate SMART_LLM_supplementary.docx: hyperparameters, additional ablations,
implementation details, additional datasets, derivations, and the reproducibility
/ data-logging schema. Hyperparameters come from the resolved config; extra
ablation numbers come from the result CSVs (placeholders if a run is missing).
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import pandas as pd

from ..config import add_config_args, config_from_args
from ..experiments.train_cdka import MASTER_COLUMNS
from ..utils.logging import get_logger
from . import docx_utils as D

_log = get_logger("smart_llm.paper")


def _dc_table(dc) -> pd.DataFrame:
    rows = [{"Parameter": f.name, "Value": str(getattr(dc, f.name))}
            for f in dataclasses.fields(dc)]
    return pd.DataFrame(rows)


def _load(cfg, name):
    p = Path(cfg.paths.tables_dir) / f"{name}_{cfg.data.dataset}.csv"
    return pd.read_csv(p) if p.exists() else None


def build(cfg):
    doc = D.new_document()
    D.title(doc, "Supplementary Material — SMART-LLM")
    D.centered(doc, "Confidence-Driven Knowledge Arbitration and Adaptive Adapter "
                    "Scaling for Explainable Few-Shot LLM Text Classification",
               italic=True, size=10)
    doc.add_paragraph()

    # ---------------- S1 hyperparameters ----------------
    D.heading(doc, "S1  Implementation Details and Hyperparameters", level=1)
    for label, dc in [("Backbone LLM", cfg.llm), ("Embedding model", cfg.embedding),
                      ("Retrieval", cfg.retrieval), ("Data", cfg.data),
                      ("Confidence probe", cfg.probe), ("RBE", cfg.rbe),
                      ("Router", cfg.router), ("UAAS", cfg.uaas),
                      ("Explanation verification", cfg.explain)]:
        D.table_from_df(doc, _dc_table(dc), f"Table S. {label} configuration.")

    # ---------------- S2 additional ablations ----------------
    D.heading(doc, "S2  Additional Ablations and Analyses", level=1)
    for name, cap in [
        ("ablation", "Table S. Module ablation: SMART (full) vs. - RBE (similarity "
                   "only) vs. - Calibration (raw RUS) vs. confidence-only vs. "
                   "always/never/oracle (agreement, precision, recall, F1, regret, "
                   "accuracy, retrieval frequency)."),
        ("table_difficulty", "Table S. Difficulty strata (easy/medium/hard by "
                          "entropy): confidence, entropy, retrieval frequency, "
                          "accuracy."),
        ("table_behavior", "Table S. Retrieval behaviour: frequency, average "
                        "retrieved examples, prompt length, decision counts."),
        ("table_uaas", "Table S. UAAS: adaptive per-input rank vs. static LoRA "
                    "ranks 4/16/32 (accuracy, macro-F1, average rank, trainable "
                    "parameters)."),
        ("table_explain", "Table S. Explanation-verification summary (mean/median "
                        "faithfulness; faithfulness for correct vs. incorrect "
                        "predictions)."),
    ]:
        df = _load(cfg, name)
        if df is not None:
            D.table_from_df(doc, df, cap)
        else:
            D.para(doc, f"{cap}  {D.TBD}")

    # ---------------- S3 additional datasets ----------------
    D.heading(doc, "S3  Additional Datasets", level=1)
    D.para(doc,
        "Beyond 20 Newsgroups, the released code provides uniform loaders for AG News "
        "(4 classes), TweetEval sentiment (3 classes), the Financial PhraseBank "
        "(3 classes), and PubMed 20k RCT sentence-role classification. Each dataset "
        "uses the same verbalizer classifier, FAISS retrieval pool, and logging "
        "schema; a run is reproduced by passing --dataset <name> to every stage.")

    # ---------------- S4 derivations ----------------
    D.heading(doc, "S4  Mathematical Details and Derivations", level=1)
    D.heading(doc, "S4.1  Benefit normalisation and numerical stability", level=2)
    D.para(doc,
        "The naive relative benefit (Loss_p - Loss_r)/(|Loss_p| + eps) with a tiny eps "
        "is numerically unstable: when the parametric model is confidently correct, "
        "Loss_p approaches zero and the ratio diverges, producing targets several "
        "orders of magnitude larger than typical values and making the regression "
        "meaningless. We therefore floor the denominator by a constant and clip the "
        "result, B_true = clip((Loss_p - Loss_r)/(|Loss_p| + floor), -c, c) with "
        "floor = 1 and c = 5. This regularises the relative scale and bounds outliers "
        "while leaving the sign unchanged, so the oracle decision 1[Loss_r < Loss_p] = "
        "1[B_true > 0] is unaffected; the RBE thus provides a bounded, sign-consistent "
        "relaxation of the oracle indicator. Because Loss_p and Loss_r are logged, "
        "B_true is recomputed at analysis time and requires no additional LLM passes.")
    D.heading(doc, "S4.2  Calibration and the routing rule", level=2)
    D.para(doc,
        "Platt scaling fits calibrated(RUS) = sigmoid(a·RUS + b) by maximum likelihood "
        "against the oracle labels on the validation split. Because sigmoid is "
        "monotone, thresholding delta_C = calibrated(RUS) - C_i at zero is equivalent "
        "to a monotone decision boundary in (RUS, C_i) space; sweeping the threshold "
        "traces the accuracy-computation frontier of Figure 4. The weights (alpha, "
        "beta) are chosen on validation to maximise oracle agreement, with sim and "
        "B_pred z-standardised so the two terms are on a common scale.")
    D.heading(doc, "S4.3  Regret", level=2)
    D.para(doc,
        "For a decision d in {0,1} with losses (Loss_p, Loss_r), the per-sample regret "
        "is r(d) = [d·Loss_r + (1-d)·Loss_p] - min(Loss_p, Loss_r) >= 0, and equals "
        "zero iff the decision matches the oracle. Mean regret is thus a loss-weighted "
        "refinement of oracle disagreement: disagreements on samples with a large "
        "|Loss_p - Loss_r| gap are penalised more than disagreements where the two "
        "options are nearly equivalent.")
    D.heading(doc, "S4.4  Uncertainty and rank", level=2)
    D.para(doc,
        "U(x) = lam·H_norm + (1-lam)(1 - C_i) is a convex combination of two "
        "uncertainty proxies in [0,1], hence U(x) in [0,1] and r(x) = r_min + "
        "(r_max - r_min)·U(x) in [r_min, r_max]. Bucketing to the nearest available "
        "adapter rank introduces a quantisation error bounded by half the largest gap "
        "between consecutive buckets.")

    # ---------------- S5 reproducibility / logging schema ----------------
    D.heading(doc, "S5  Reproducibility and Data-Logging Schema", level=1)
    D.para(doc,
        "Every evaluated sample is logged to results/master_<dataset>.csv with one "
        "row per (pooling × retrieval-condition × sample). The columns are:")
    meanings = {
        "id": "sample identifier", "dataset": "dataset name",
        "label": "gold label index", "pooling": "hidden pooling type",
        "condition": "retrieval condition", "split": "train/val/test split",
        "C_i": "internal confidence C_i (probe)", "entropy": "normalised entropy (probe)",
        "probe_pred": "probe argmax prediction", "conf_llm": "LLM verbalizer confidence",
        "entropy_llm": "LLM normalised entropy", "sim": "similarity(x,K)",
        "B_pred": "predicted benefit", "B_true": "ground-truth benefit",
        "RUS": "retrieval utility score", "calibrated_RUS": "calibrated RUS",
        "delta_C": "confidence gap delta_C", "smart_decision": "SMART decision (1=retrieve)",
        "oracle_decision": "oracle decision (1=retrieve)",
        "loss_without_retrieval": "loss without retrieval",
        "loss_with_retrieval": "loss with retrieval",
        "pred_p": "parametric prediction", "pred_r": "retrieval prediction",
        "smart_pred": "SMART prediction", "regret": "per-sample regret",
        "t_p": "no-retrieval latency (s)", "t_r": "retrieval latency (s)",
        "n_tokens_p": "no-retrieval prompt tokens", "n_tokens_r": "retrieval prompt tokens"}
    schema = pd.DataFrame({"Column": MASTER_COLUMNS,
                           "Meaning": [meanings.get(c, "") for c in MASTER_COLUMNS]})
    D.table_from_df(doc, schema, "Table S. Master per-sample logging schema.")
    D.para(doc,
        "The pipeline is deterministic given the seed. Stage 1 (feature extraction) "
        "runs the frozen backbone once and caches hidden states, ground-truth losses, "
        "and embeddings; Stage 2 (CDKA training) and all analysis consume the cache, "
        "so results reproduce without re-running the LLM.")

    D.heading(doc, "S6  Explanation-Verification Protocol", level=1)
    D.para(doc,
        "Integrated Gradients is computed over the input-embedding layer with a "
        "pad-token baseline; token importances are the absolute summed attributions. "
        "Content tokens (length >= 3, non-stopword) are ranked and the top-k are "
        "compared, by word-boundary match, against the model's generated explanation. "
        "We report the fraction covered as the faithfulness score, stratified by "
        "prediction correctness.")

    out = Path(cfg.paths.paper_dir) / "SMART_LLM_supplementary.docx"
    Path(cfg.paths.paper_dir).mkdir(parents=True, exist_ok=True)
    doc.save(str(out))
    _log.info("Wrote %s", out)
    return str(out)


def main():
    ap = argparse.ArgumentParser(description="Build SMART_LLM_supplementary.docx")
    add_config_args(ap)
    args = ap.parse_args()
    cfg = config_from_args(args)
    build(cfg)


if __name__ == "__main__":
    main()
