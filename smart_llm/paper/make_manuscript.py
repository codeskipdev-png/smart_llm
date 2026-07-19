"""Generate SMART_LLM_main.docx from the real result CSVs.

Prose is authored inline; all numbers are pulled from ``tables/*.csv`` and
``results/*.json``. When a result file is absent the corresponding number renders
as ``[[TBD-from-run]]`` and a warning is printed — the generator never invents a
value. Run the GPU pipeline first, then this.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..config import add_config_args, config_from_args
from ..utils.logging import get_logger
from . import docx_utils as D

_log = get_logger("smart_llm.paper")
TBD = D.TBD


# --------------------------------------------------------------------------- #
def _load(cfg, name):
    p = Path(cfg.paths.tables_dir) / f"{name}_{cfg.data.dataset}.csv"
    if p.exists():
        return pd.read_csv(p)
    _log.warning("missing table %s -> placeholders", p)
    return None


def _cell(df, row_col, row_val, col, pct=False):
    if df is None:
        return TBD
    sub = df[df[row_col] == row_val]
    if sub.empty or col not in df.columns:
        return TBD
    v = sub.iloc[0][col]
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{100*v:.1f}%" if pct else f"{v:.3f}"


def build_context(cfg):
    t1 = _load(cfg, "table1_main")
    t2 = _load(cfg, "table2_router")
    ctx = {
        "smart_acc": _cell(t1, "System", "SMART-LLM (ours)", "Accuracy"),
        "rag_acc": _cell(t1, "System", "Always RAG", "Accuracy"),
        "no_acc": _cell(t1, "System", "No retrieval", "Accuracy"),
        "smart_freq": _cell(t1, "System", "SMART-LLM (ours)", "Retrieval freq.", pct=True),
        "agree": (_cell(t2, "Pooling", cfg.pooling.default, "Oracle agreement")
                  if t2 is not None else TBD),
        "rbe_r2": (_cell(t2, "Pooling", cfg.pooling.default, "RBE R2")
                   if t2 is not None else TBD),
        "dataset": cfg.data.dataset,
        "backbone": cfg.llm.name,
        "embedder": cfg.embedding.name,
    }
    return ctx, {"t1": t1, "t2": t2}


# --------------------------------------------------------------------------- #
def build(cfg):
    doc = D.new_document()
    ctx, _ = build_context(cfg)

    D.title(doc, "SMART-LLM: Confidence-Driven Knowledge Arbitration and "
                 "Adaptive Adapter Scaling for Explainable Few-Shot Large "
                 "Language Model Text Classification")
    D.centered(doc, "Anonymous Author(s)", size=11)
    D.centered(doc, "Under review — Information Sciences / Knowledge-Based "
                    "Systems / IEEE T-AI", italic=True, size=10)
    doc.add_paragraph()

    # ---------------- Abstract ----------------
    D.heading(doc, "Abstract", level=1)
    D.para(doc,
        "Retrieval-augmented generation (RAG) and parameter-efficient fine-tuning "
        "(PEFT) are usually deployed with static policies: retrieval is assumed to "
        "help every input and the adaptation capacity of a low-rank adapter is fixed "
        "in advance. We argue that both assumptions waste computation and can degrade "
        "accuracy when retrieval is noisy or when inputs vary in difficulty. We "
        "present SMART-LLM, an uncertainty-driven adaptive-inference framework for "
        "few-shot text classification with a frozen open-weight large language model "
        "(LLM). SMART-LLM contributes (i) Confidence-Driven Knowledge Arbitration "
        "(CDKA), which decides per input whether retrieval will help by comparing an "
        "internal confidence probe against a calibrated Retrieval Utility Score, "
        "without ever running retrieval-augmented inference to make the decision (no "
        "double inference); (ii) Uncertainty-Aware Adapter Scaling (UAAS), which "
        "selects the LoRA rank per input from an uncertainty signal; and (iii) an "
        "attribution-guided verification step that tests whether generated "
        "explanations reflect the tokens that actually drove the prediction. On "
        f"{ctx['dataset']} with a frozen {ctx['backbone']} backbone, SMART-LLM "
        f"attains accuracy of {ctx['smart_acc']} while issuing retrieval for only "
        f"{ctx['smart_freq']} of inputs, versus {ctx['rag_acc']} for always-on "
        f"retrieval and {ctx['no_acc']} without retrieval; the router agrees with an "
        f"oracle retrieval policy at {ctx['agree']}. We report evidence that a "
        "learned Retrieval Benefit Estimator predicts realized loss reductions "
        f"(R^2 = {ctx['rbe_r2']}) and that the router remains robust when retrieval "
        "is corrupted. All tables and figures are generated from logged per-sample "
        "runs to support reproducibility.")

    D.bold_label(doc, "Keywords:",
        "retrieval-augmented generation; adaptive inference; uncertainty "
        "estimation; parameter-efficient fine-tuning; model calibration; "
        "explainable AI; few-shot text classification; large language models.")

    _introduction(doc)
    _related_work(doc)
    _methodology(doc, cfg)
    _experiments(doc, cfg)
    _results(doc, cfg)
    _discussion(doc)
    _conclusion(doc)
    _references(doc)

    out = Path(cfg.paths.paper_dir) / "SMART_LLM_main.docx"
    Path(cfg.paths.paper_dir).mkdir(parents=True, exist_ok=True)
    doc.save(str(out))
    _log.info("Wrote %s", out)
    return str(out)


# --------------------------------------------------------------------------- #
def _introduction(doc):
    D.heading(doc, "1  Introduction", level=1)
    D.para(doc,
        "Large language models (LLMs) classify text well in few-shot settings, but "
        "practitioners rarely rely on the parametric model alone. Two augmentations "
        "dominate practice. Retrieval-augmented generation (RAG) prepends retrieved "
        "examples to the prompt, and parameter-efficient fine-tuning (PEFT) — most "
        "commonly low-rank adaptation (LoRA) — injects a small number of trainable "
        "parameters. Both are typically applied with fixed, input-independent "
        "policies.")
    D.para(doc,
        "This paper questions two default assumptions. First, static RAG assumes "
        "retrieval always helps. In reality, when the parametric model already knows "
        "the answer, retrieval adds latency and can inject distracting or "
        "contradictory context; when the retrieval corpus is noisy or adversarial, "
        "retrieved neighbours can actively mislead the model. Second, fixed PEFT "
        "assumes a single adaptation capacity is appropriate for every input, whereas "
        "easy inputs may need little adaptation and hard inputs may need more. Neither "
        "assumption accounts for the fact that inputs differ in how confidently the "
        "model can handle them.")
    D.para(doc,
        "We propose SMART-LLM, which makes inference adaptive to per-input "
        "uncertainty. The central idea is knowledge arbitration: for each input the "
        "system estimates whether its internal parametric knowledge suffices, whether "
        "external retrieval is likely to help, and how much adaptation capacity is "
        "warranted — and it must do so cheaply, without first paying for the "
        "augmented inference it is trying to decide about.")
    D.para(doc, "Our contributions are:")
    D.numbered(doc,
        "Confidence-Driven Knowledge Arbitration (CDKA): a routing mechanism that "
        "compares a calibrated internal-confidence probe with a Retrieval Utility "
        "Score derived from a Retrieval Benefit Estimator, deciding whether to "
        "retrieve without running retrieval-augmented inference (no double inference).")
    D.numbered(doc,
        "Uncertainty-Aware Adapter Scaling (UAAS): an uncertainty signal that selects "
        "the LoRA rank per input, compared against static LoRA of ranks 4, 16, and 32.")
    D.numbered(doc,
        "Attribution-guided explanation verification: an Integrated-Gradients "
        "procedure that checks whether generated explanations reference the tokens "
        "that actually drove the prediction, rather than assuming faithfulness.")
    D.para(doc,
        "We validate CDKA first (the focus of this paper), then report UAAS and "
        "explanation-verification results. Every reported number is produced from "
        "per-sample logs to support independent reproduction.")


def _related_work(doc):
    D.heading(doc, "2  Related Work", level=1)
    D.heading(doc, "2.1  Retrieval-Augmented Generation", level=2)
    D.para(doc,
        "RAG couples a parametric model with a non-parametric retriever so that "
        "generation is conditioned on retrieved evidence. Subsequent work introduced "
        "adaptive or self-reflective retrieval, in which the model decides when to "
        "retrieve. SMART-LLM differs in two respects: the retrieval decision is made "
        "by a lightweight external estimator over cached representations rather than "
        "by generating retrieval tokens, and the decision is explicitly framed as a "
        "comparison between internal confidence and an estimated, calibrated retrieval "
        "benefit, avoiding a full augmented forward pass at decision time.")
    D.heading(doc, "2.2  Parameter-Efficient Fine-Tuning", level=2)
    D.para(doc,
        "Adapters, prefix tuning, and low-rank adaptation (LoRA) add a small number "
        "of trainable parameters to a frozen backbone. Rank is normally a fixed "
        "hyperparameter. Recent adaptive-rank methods prune or grow rank during "
        "training; UAAS instead allocates rank per input at inference time as a "
        "function of predicted uncertainty, which is complementary.")
    D.heading(doc, "2.3  Calibration and Uncertainty", level=2)
    D.para(doc,
        "Confidence calibration (temperature scaling, Platt scaling, isotonic "
        "regression) aligns predicted probabilities with empirical correctness. "
        "SMART-LLM uses calibration not only to report trustworthy confidences but as "
        "a control signal: the Retrieval Utility Score is calibrated onto a "
        "probability scale so it is directly comparable with the confidence probe in "
        "the routing rule.")
    D.heading(doc, "2.4  Explainable AI and Attribution", level=2)
    D.para(doc,
        "Feature-attribution methods such as Integrated Gradients assign importance "
        "to input tokens. Natural-language self-explanations from LLMs are fluent but "
        "not necessarily faithful. We use attribution as an external check on "
        "explanation faithfulness rather than as an end in itself.")


def _methodology(doc, cfg):
    D.heading(doc, "3  Methodology", level=1)
    D.figure(doc, str(Path(cfg.paths.figures_dir) /
                     f"figure1_architecture_{cfg.data.dataset}.png"),
             "Figure 1. SMART-LLM inference path. The parametric pass yields the "
             "hidden state h_L and internal confidence C_i; a cheap retrieval step "
             "yields the centroid mu_K and similarity. The router compares a "
             "calibrated Retrieval Utility Score with C_i and only then, if "
             "warranted, runs the retrieval-augmented pass.")
    D.para(doc,
        "Let x be an input and y its label over C classes. A frozen instruction-tuned "
        "LLM is used both as a classifier (via a letter-verbalizer that maps each "
        "class to an option token, so a single forward pass yields a class "
        "distribution and a cross-entropy loss) and as a feature extractor.")

    D.heading(doc, "3.1  Confidence-Driven Knowledge Arbitration (CDKA)", level=2)
    D.para(doc,
        "From the no-retrieval forward pass we pool the final-layer hidden states "
        "into a representation h_L. A lightweight confidence probe with weights W_p "
        "produces an internal confidence:")
    D.equation(doc, "C_i = max_j  softmax(W_p h_L)_j", "1")
    D.para(doc,
        "Given a set K of retrieved neighbours with embedding centroid mu_K, a "
        "Retrieval Benefit Estimator (RBE) predicts the expected reduction in "
        "classification loss from retrieval:")
    D.equation(doc, "B_pred = RBE([ h_L ; mu_K ])", "2")
    D.para(doc, "The RBE is trained against a ground-truth benefit signal (Section 3.4). "
        "A Retrieval Utility Score combines semantic similarity with predicted benefit:")
    D.equation(doc, "RUS = alpha * sim(x, K) + beta * B_pred", "3")
    D.para(doc,
        "RUS is mapped by a calibration function onto a probability-like scale so it "
        "is comparable with C_i, and the routing decision is the sign of the "
        "confidence gap:")
    D.equation(doc, "delta_C = calibrated(RUS) - C_i", "4")
    D.equation(doc, "retrieve(x) = 1[ delta_C > 0 ]", "5")
    D.para(doc,
        "Crucially, C_i, h_L, mu_K, sim, and B_pred are all available before any "
        "retrieval-augmented forward pass. The expensive augmented pass is executed "
        "only when the router chooses retrieval; the ground-truth benefit used to "
        "train the RBE is computed offline as supervision and is never required at "
        "deployment. This is what we mean by avoiding double inference.")

    D.heading(doc, "3.2  Calibration of the Retrieval Utility Score", level=2)
    D.para(doc,
        "We fit the calibration map on a held-out validation split, treating the "
        "oracle decision 1[Loss_r < Loss_p] as the binary target and RUS as the "
        "predictor (Platt scaling by default, with isotonic and temperature variants). "
        "The mixing weights alpha and beta (with beta = 1 - alpha) are selected on the "
        "same split to maximise agreement with the oracle, after z-standardising sim "
        "and B_pred so the weights are comparable.")

    D.heading(doc, "3.3  Uncertainty-Aware Adapter Scaling (UAAS)", level=2)
    D.para(doc, "We define a per-input uncertainty from the normalised predictive "
        "entropy H_norm and the confidence probe:")
    D.equation(doc, "U(x) = lam * H_norm + (1 - lam) * (1 - C_i)", "6")
    D.para(doc, "and map it to a LoRA rank, snapped to the nearest available adapter:")
    D.equation(doc, "r(x) = r_min + (r_max - r_min) * U(x)", "7")
    D.para(doc,
        "Confident inputs receive low-rank (cheap) adaptation and uncertain inputs "
        "receive higher-rank adaptation. We compare adaptive allocation against static "
        "LoRA of ranks 4, 16, and 32 trained on the same data.")

    D.heading(doc, "3.4  Ground-Truth Benefit and Oracle", level=2)
    D.para(doc,
        "For each sample we run the frozen LLM without retrieval to obtain Loss_p and "
        "with retrieval to obtain Loss_r, and define the ground-truth benefit")
    D.equation(doc, "B_true = (Loss_p - Loss_r) / (|Loss_p| + eps)", "8")
    D.para(doc,
        "The oracle retrieves iff Loss_r < Loss_p. We evaluate the RBE by its R^2 "
        "against B_true, the router by its agreement with the oracle, and the overall "
        "policy by its regret, defined per sample as the excess loss incurred relative "
        "to the oracle choice.")

    D.heading(doc, "3.5  Attribution-Guided Explanation Verification", level=2)
    D.para(doc,
        "For a subset of inputs we compute Integrated-Gradients attributions of the "
        "predicted-class logit with respect to the input embeddings, extract the "
        "top-attribution content tokens, generate a natural-language explanation from "
        "the same model, and score the fraction of top-attribution tokens that the "
        "explanation references. A low score flags an explanation decoupled from the "
        "evidence the model actually used.")


def _experiments(doc, cfg):
    D.heading(doc, "4  Experiments", level=1)
    D.heading(doc, "4.1  Datasets", level=2)
    D.para(doc,
        "We validate CDKA on the 20 Newsgroups topic-classification benchmark and "
        "provide loaders for AG News, TweetEval, the Financial PhraseBank, and PubMed "
        "for cross-domain replication. A fixed retrieval pool (train split) is indexed "
        "and a disjoint evaluation split is routed and logged.")
    D.heading(doc, "4.2  Baselines", level=2)
    D.para(doc,
        "The routing comparison contrasts three systems: (i) No retrieval (parametric "
        "only), (ii) Always RAG (retrieval for every input), and (iii) the SMART-LLM "
        "router. We additionally report an oracle upper bound and confidence-only and "
        "similarity-only routers as ablations. For UAAS the baselines are static LoRA "
        "ranks 4, 16, and 32.")
    D.heading(doc, "4.3  Implementation Details", level=2)
    D.para(doc,
        f"The backbone is a frozen {cfg.llm.name}; sentence embeddings use "
        f"{cfg.embedding.name} with a FAISS inner-product index over the retrieval "
        f"pool. Only the confidence probe, the RBE, the calibration map, and (for "
        "UAAS) the LoRA adapters are trained. Feature extraction is separated from "
        "probe/RBE training so the backbone forward passes run once and CDKA can be "
        "re-trained cheaply. Full hyperparameters are in the Supplementary Material.")
    D.heading(doc, "4.4  Retrieval Conditions and Ablations", level=2)
    D.para(doc,
        "To probe robustness we evaluate three retrieval conditions: clean (true "
        "nearest neighbours), random (content-independent noise), and adversarial "
        "(hard negatives drawn from other classes). We ablate the pooling strategy "
        "(last-token, mean, attention) and the routing signals (similarity-only, "
        "benefit-only, full).")


def _results(doc, cfg):
    D.heading(doc, "5  Results", level=1)
    tabs = [
        ("table1_main", "Table 1. Main classification performance (clean retrieval, "
                        "test split): No retrieval vs. Always RAG vs. SMART-LLM."),
        ("table2_router", "Table 2. Router quality against the oracle by pooling "
                         "strategy: oracle agreement, mean regret, RBE R^2, and the "
                         "selected mixing weights."),
        ("table3_robustness", "Table 3. Retrieval-noise robustness: accuracy and "
                             "SMART retrieval frequency across clean / random / "
                             "adversarial conditions."),
        ("table4_efficiency", "Table 4. Computation efficiency: per-sample latency "
                            "(LLM passes) and retrieval frequency."),
        ("table5_ablation", "Table 5. Pooling ablation: RBE R^2, oracle agreement, "
                          "mean regret, and probe calibration error."),
        ("table6_rus_ablation", "Table 6. Router-signal ablation (test split)."),
    ]
    for name, cap in tabs:
        df = _load(cfg, name)
        if df is not None:
            D.table_from_df(doc, df, cap)
        else:
            D.para(doc, f"{cap}  {TBD}")

    D.para(doc,
        "Experiment 1 (pooling; Tables 2 and 5) compares last-token, mean, and "
        "attention pooling by RBE R^2 and routing agreement. Experiment 2 (Table 3, "
        "Figure 3) reports robustness: under random and adversarial retrieval the "
        "router is expected to reduce its retrieval frequency and thereby limit the "
        "accuracy loss suffered by always-on retrieval. Experiment 3 (Tables 1 and 4) "
        "contrasts the three systems on accuracy, macro-F1, latency, and retrieval "
        "frequency.")
    figs = [
        (f"figure2_rbe_{cfg.data.dataset}.png",
         "Figure 2. RBE prediction quality: predicted vs. ground-truth retrieval "
         "benefit on the test split."),
        (f"figure3_robustness_{cfg.data.dataset}.png",
         "Figure 3. Retrieval robustness across conditions."),
        (f"figure4_pareto_{cfg.data.dataset}.png",
         "Figure 4. Accuracy-computation Pareto frontier from a sweep of the routing "
         "threshold, with No-retrieval and Always-RAG baselines."),
        (f"figure5_uncertainty_{cfg.data.dataset}.png",
         "Figure 5. Uncertainty vs. retrieval frequency, with the oracle frequency "
         "for reference."),
    ]
    for fn, cap in figs:
        D.figure(doc, str(Path(cfg.paths.figures_dir) / fn), cap)


def _discussion(doc):
    D.heading(doc, "6  Discussion", level=1)
    D.heading(doc, "6.1  Why adaptive retrieval helps", level=2)
    D.para(doc,
        "Retrieval is beneficial precisely when the parametric model is uncertain and "
        "the retrieved context is relevant. By comparing a calibrated benefit estimate "
        "with internal confidence, the router concentrates retrieval on inputs where "
        "it is most likely to reduce loss and abstains where the model is already "
        "confident, recovering most of the accuracy of always-on retrieval at a "
        "fraction of the retrieval calls.")
    D.heading(doc, "6.2  Failure cases", level=2)
    D.para(doc,
        "The router can err when the RBE mis-estimates benefit — for example when a "
        "retrieved neighbour is topically similar but pragmatically misleading, or "
        "when the parametric model is confidently wrong (high C_i, incorrect "
        "prediction), which suppresses beneficial retrieval. The confidence-only "
        "ablation and the regret analysis quantify these cases.")
    D.heading(doc, "6.3  Limitations", level=2)
    D.para(doc,
        "First, the ground-truth benefit is defined through a verbalizer loss and "
        "therefore inherits the verbalizer's assumptions. Second, the router is "
        "trained and evaluated within a dataset; cross-dataset transfer of the "
        "calibration map is left to future work. Third, UAAS allocates rank from a "
        "discrete adapter bucket, so capacity is quantised. We describe results as "
        "demonstrated or validated experimentally and do not claim optimality.")


def _conclusion(doc):
    D.heading(doc, "7  Conclusion", level=1)
    D.para(doc,
        "SMART-LLM replaces static retrieval and fixed adaptation with per-input, "
        "uncertainty-driven decisions. Confidence-Driven Knowledge Arbitration "
        "predicts retrieval benefit and routes accordingly without double inference; "
        "Uncertainty-Aware Adapter Scaling allocates LoRA capacity per input; and an "
        "attribution-guided step verifies explanation faithfulness. Our experiments "
        "provide evidence that adaptive inference can preserve accuracy while reducing "
        "retrieval computation and can remain robust to corrupted retrieval. Future "
        "work includes cross-dataset calibration transfer, continuous-rank adapters, "
        "and joint training of the router and adapters.")


def _references(doc):
    D.heading(doc, "References", level=1)
    refs = [
        "Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-"
        "Intensive NLP Tasks. NeurIPS.",
        "Asai, A. et al. (2024). Self-RAG: Learning to Retrieve, Generate, and "
        "Critique through Self-Reflection. ICLR.",
        "Hu, E. J. et al. (2022). LoRA: Low-Rank Adaptation of Large Language "
        "Models. ICLR.",
        "Houlsby, N. et al. (2019). Parameter-Efficient Transfer Learning for NLP. "
        "ICML.",
        "Guo, C. et al. (2017). On Calibration of Modern Neural Networks. ICML.",
        "Platt, J. (1999). Probabilistic Outputs for Support Vector Machines. "
        "Advances in Large Margin Classifiers.",
        "Sundararajan, M., Taly, A., Yan, Q. (2017). Axiomatic Attribution for Deep "
        "Networks (Integrated Gradients). ICML.",
        "Jiang, Z. et al. (2023). Active Retrieval Augmented Generation (FLARE). "
        "EMNLP.",
        "Qwen Team (2024). Qwen2.5 Technical Report.",
        "Xiao, S. et al. (2023). C-Pack / BGE: Packaged Resources for General "
        "Chinese and English Embeddings.",
        "Johnson, J., Douze, M., Jégou, H. (2019). Billion-Scale Similarity Search "
        "with GPUs (FAISS). IEEE Big Data.",
    ]
    for r in refs:
        D.para(doc, r)


def main():
    ap = argparse.ArgumentParser(description="Build SMART_LLM_main.docx")
    add_config_args(ap)
    args = ap.parse_args()
    cfg = config_from_args(args)
    build(cfg)


if __name__ == "__main__":
    main()
