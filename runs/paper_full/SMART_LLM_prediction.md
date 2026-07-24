# Confidence-Aware Retrieval and Mixture-of-LoRA Adaptation for Explainable Few-Shot Large Language Model Text Classification (SMART-LLM)

**A Pre-Experimental Prediction Manuscript (Scientific Forecast)**

Anonymous Author(s) — prepared for submission to *Information Sciences / Knowledge-Based Systems / IEEE Transactions on Artificial Intelligence*

> **STATUS — READ FIRST.** This is a **pre-experimental prediction manuscript**, not an experimental paper. **No experiments have been conducted.** Every quantitative value in Section 5 and Section 7 is an **AI-predicted experimental outcome (before execution)** produced by scientific reasoning from prior literature and the proposed mechanism. Numbers are forecasts with stated uncertainty ranges and per-table confidence levels; they will be compared against real RTX 4090 experiments later. They must not be cited as measurements. Where this document states a value, read it as "predicted," never "observed."

---

## Abstract

Retrieval augmentation improves few-shot large language model (LLM) text classification on many inputs but not all: when the parametric model is already correct, retrieval adds cost, and when retrieved context is off-topic or misleading, it can displace a correct prediction. We study whether an LLM-based system can decide, *before retrieving*, whether retrieval will help a given input. We describe SMART-LLM, which combines (i) a calibrated internal-confidence probe over a frozen LLM, (ii) a Retrieval Benefit Estimator (RBE) that predicts the expected reduction in classification loss from retrieval using only pre-retrieval features, (iii) a Retrieval Utility Score (RUS) that fuses similarity and predicted benefit, (iv) confidence calibration that places utility and confidence on a common scale, and (v) an auxiliary Mixture-of-LoRA adaptation that scales adapter capacity with input uncertainty. Because the decision uses only cached pre-retrieval features, no retrieval-augmented forward pass is spent to decide. This manuscript **predicts**, prior to execution, the likely outcomes of a planned evaluation on 20 Newsgroups (primary) and Financial PhraseBank (secondary) with a frozen Qwen2.5-7B-Instruct backbone and bge-large embeddings, against six baselines. We forecast that selective arbitration will recover most of the accuracy benefit of always-on retrieval at roughly one-third of the retrieval rate, that calibration will be the single most important component, that the learned benefit term will add measurable value chiefly where semantic similarity is a weak cue (sentiment more than topic), and that the clearest advantage over always-on retrieval will appear under corrupted retrieval. We state confidence levels and falsification conditions for each prediction. *All results herein are AI-predicted experimental outcomes (before execution).*

**Keywords:** decision-time retrieval; selective prediction; adaptive computation; retrieval benefit estimation; confidence calibration; regret analysis; parameter-efficient fine-tuning; few-shot text classification; large language models; pre-registered prediction.

---

## 1  Introduction

Augmenting a frozen LLM with retrieved in-context examples reliably improves many, but not all, few-shot text-classification inputs. In its default form the augmentation is unconditional: every input triggers a retrieval and a longer, more expensive forward pass. This is wasteful when the parametric model already answers correctly and harmful when the retrieved context is off-topic, redundant, or adversarial and displaces a correct parametric prediction. The quantity that matters — whether retrieval will help *this* input — is normally observed only after retrieving and running the augmented model, which is precisely the cost one would like to avoid.

We treat retrieval as a per-input decision under uncertainty and ask a single question:

> **Research question.** Can an LLM-based system estimate, before performing retrieval, whether retrieval will improve its prediction for a given input, and route on that estimate to preserve accuracy while avoiding unnecessary and harmful retrieval?

This reframing places the problem alongside selective prediction and adaptive computation, rather than alongside methods that improve retrieval quality itself. SMART-LLM instantiates it with two pre-retrieval signals — a calibrated internal confidence from a lightweight probe on the parametric hidden state, and a predicted retrieval benefit from a Retrieval Benefit Estimator over the hidden state and the retrieved-neighbour centroid — combined by a calibrated comparison. Two auxiliary mechanisms (Mixture-of-LoRA adaptation and attribution-based explanation verification) are included for completeness and are not central to the decision-time claim.

**Contributions (as claims to be tested, not yet verified).**
1. A formulation of the retrieval decision as pre-retrieval benefit prediction, learned from an offline ground-truth benefit signal.
2. A calibrated arbitration rule that fuses confidence and utility, analysed by agreement with an oracle policy and by regret.
3. A pre-experimental forecast, with explicit confidence and falsification criteria, of how the method and six baselines will behave — the object of this manuscript.

Because this is a forecast, the Introduction makes no empirical claim; Section 5 states predictions and Section 9 (Prediction Reliability Assessment) states what would falsify them.

---

## 2  Related Work

**Selective prediction and decision-theoretic inference.** Selective classifiers abstain when confidence is low, trading coverage for reliability (Geifman & El-Yaniv, 2017; El-Yaniv & Wiener, 2010). We adopt the same per-input framing — "retrieve vs. trust the parametric model" in place of "predict vs. abstain" — and inherit its evaluation discipline (oracle comparison, regret).

**Adaptive computation.** Early-exit networks (Xin et al., 2020), conditional computation and mixtures of experts (Fedus et al., 2022), and confident adaptive language modelling (Schuster et al., 2022) allocate internal compute per input. SMART-LLM shares the objective but targets an *external* resource — whether to spend a retrieval-augmented pass — and decides before the expensive pass from cached features.

**Adaptive and selective retrieval (closest work).** RAG conditions predictions on retrieved evidence (Lewis et al., 2020). One family decides *during* decoding: Self-RAG emits retrieval-control tokens (Asai et al., 2024); FLARE triggers on next-token uncertainty (Jiang et al., 2023). A second family decides *before* generation but from coarse signals: popularity gating (Mallen et al., 2023), learned query-complexity routing (Adaptive-RAG; Jeong et al., 2024), and self-knowledge elicitation (SKR; Wang et al., 2023), building on the finding that LLMs are partly aware of their own competence (Kadavath et al., 2022). SMART-LLM belongs to the decide-before-retrieving family but (i) regresses a continuous, loss-calibrated benefit rather than classifying complexity/popularity, (ii) reads the hidden state and neighbour centroid cached before any augmented pass, and (iii) scores against an oracle by agreement and regret. We do not claim to beat these methods on their native QA benchmarks; we isolate the pre-retrieval benefit-estimation question and include an entropy-gated (FLARE/Adaptive-RAG-style) baseline.

**Calibration and PEFT.** Temperature/Platt/isotonic calibration (Guo et al., 2017; Platt, 1999; Zadrozny & Elkan, 2002) align probabilities with accuracy; we use calibration as a control signal so utility and confidence are comparable. LoRA (Hu et al., 2022) and Integrated Gradients (Sundararajan et al., 2017) support the auxiliary components only.

---

## 3  Methodology

### 3.1  Problem Definition

Let *x* be an input with gold label *y* over *K* classes. A frozen classifier incurs cross-entropy loss ℓ_p(x) without retrieval and ℓ_r(x) with retrieval of a neighbour set 𝒩 (|𝒩| = k). A retrieval policy π: X → {0,1} has expected loss L(π) = 𝔼_x[(1−π)ℓ_p + π ℓ_r]. The per-input **retrieval benefit** is b(x) = ℓ_p(x) − ℓ_r(x); the **oracle** retrieves iff b(x) > 0, i.e., iff ℓ_r < ℓ_p. We measure any policy by **regret** R(π) = 𝔼_x[|b(x)|·1[π(x) ≠ π*(x)]] ≥ 0. Because ℓ_r is unavailable before retrieving, the deployed policy must approximate the *sign* of b(x) from pre-retrieval features.

*Loss vs. correctness.* b(x) is defined on cross-entropy, so the oracle above is a **loss** oracle; a retrieval that lowers loss need not flip the arg-max label. We therefore also track a **correctness** oracle (retrieval changes an incorrect parametric label to correct) and report which quantity each table measures.

### 3.2  SMART-LLM Architecture

A frozen instruction-tuned LLM classifies via a letter-verbalizer (each class maps to an option token), so one forward pass yields a class distribution, a cross-entropy loss, and pooled final-layer hidden states h_L. The arbiter consumes h_L, a calibrated confidence C_i, the retrieved-neighbour centroid μ_𝒩, the query–neighbour similarity sim, and a predicted benefit B_pred. All are available before the augmented pass; the augmented pass runs only for selected inputs.

### 3.3  Retrieval Benefit Estimator (RBE)

The RBE is a small MLP that predicts the loss reduction from retrieval using pre-retrieval features only:

B_pred = RBE([ h_L ; μ_𝒩 ]).

It is trained against a numerically stable ground-truth benefit obtained offline by running the frozen LLM with and without retrieval (the augmented pass is used only to build supervision, never to decide):

B_true = clip( (ℓ_p − ℓ_r) / (|ℓ_p| + τ), −c, c ),

with a denominator floor τ (avoiding blow-up as ℓ_p → 0) and clip c, trained with a robust Huber objective. The floor/clip change magnitudes but preserve the sign, so the oracle decision is unaffected.

### 3.4  Retrieval Utility Score (RUS)

RUS fuses semantic similarity and predicted benefit:

RUS(x, 𝒩) = α · sim(x, 𝒩) + β · B_pred,

with α, β tuned on validation by oracle agreement.

### 3.5  Confidence Calibration

The raw verbalizer confidence of instruction-tuned LLMs is typically over-confident. A lightweight probe with temperature-scaled logits produces C_i = max_j softmax(W_p h_L)_j, and RUS is mapped to a probability scale by a calibrator fitted on a held-out split (Platt shown; isotonic/temperature supported). The arbiter compares the two calibrated quantities:

ΔC(x) = cal(RUS) − C_i,  retrieve(x) = 1[ ΔC(x) > 0 ].

Interpreting cal(RUS) ≈ P(b(x) > 0 | x) and C_i ≈ P(parametric correct | x), the rule retrieves when the estimated probability that retrieval helps exceeds the estimated probability that the internal answer is already correct.

### 3.6  Mixture-of-LoRA Adaptation (auxiliary)

As a complementary efficiency mechanism, an uncertainty signal U(x) = λ·H_norm + (1−λ)(1 − C_i) selects among a small bank of LoRA experts / scales adapter rank per input: r(x) = r_min + (r_max − r_min)·U(x). Confident inputs receive cheap low-rank adaptation; uncertain inputs receive more capacity. This is not central to the decision-time claim and is evaluated separately.

### 3.7  Inference Algorithm

```
Input: x, index over labelled pool
1. Parametric pass (frozen LLM): obtain class distribution, ℓ_p, h_L, C_i (calibrated)
2. Retrieve k neighbours 𝒩 from the index; compute sim, centroid μ_𝒩   # vectors only, no LLM pass
3. B_pred ← RBE([h_L; μ_𝒩]);  RUS ← α·sim + β·B_pred;  cal(RUS)
4. ΔC ← cal(RUS) − C_i
5. if ΔC > 0:  run retrieval-augmented pass -> prediction, ℓ_r   # only now is the augmented pass spent
   else:       return parametric prediction
6. (auxiliary) select LoRA capacity from U(x); (auxiliary) attribution-based explanation check
```

---

## 4  Experimental Setup

### 4.1  Datasets

- **Primary — 20 Newsgroups:** 20 topical classes, several fine-grained and overlapping (e.g., `comp.sys.ibm.pc.hardware` vs. `comp.sys.mac.hardware`; the politics/religion clusters). Headers/footers/quotes removed; stratified evaluation subset.
- **Secondary — Financial PhraseBank:** 3-class sentence-level sentiment (negative/neutral/positive), used to test generalization to a regime where semantic similarity is a weaker cue for retrieval usefulness.

### 4.2  Implementation Details

Frozen **Qwen2.5-7B-Instruct** (bf16) on a single **RTX 4090 (24 GB)**; embeddings **bge-large** with a FAISS inner-product index over the training pool; k = 8 neighbours. Only the confidence probe, the RBE, the calibration map, and (auxiliary) LoRA parameters are trained. The LLM is frozen and its forward passes are cached so it runs once. Last-token pooling selected by validation routing agreement. **Statistical protocol (planned):** 5 seeds; 95% bootstrap confidence intervals (10,000 resamples) over per-sample logs; paired McNemar (accuracy) and paired bootstrap (regret/agreement); differences reported significant only at p < 0.05.

### 4.3  Baselines

(1) No Retrieval; (2) Always Retrieval-Augmented Generation; (3) Confidence-based retrieval (retrieve iff C_i < τ); (4) Entropy-based retrieval (retrieve iff predictive entropy > τ); (5) Adaptive-RAG-style routing (learned/threshold complexity gate); (6) **SMART-LLM**. A budget-matched **Random** policy is included in the ablation to separate "decide well" from "retrieve less."

### 4.4  Evaluation Metrics

Accuracy, macro precision/recall/F1; retrieval rate; latency and token cost; calibration (ECE, Brier); router quality vs. oracle (agreement, precision/recall/F1, regret); RBE quality (R², MAE, Pearson r).

---

## 5  Predicted Experimental Results

> **All values below are AI-predicted experimental outcomes (before execution).** Each is a point forecast with a plausible range; each table carries a confidence level (High / Medium / Low) and reasoning. Ordering/direction predictions are generally more reliable than absolute magnitudes (see Section 9). Regret is reported in cross-entropy-loss units (excess CE over the oracle choice).

### Table 1 — Predicted main performance (20 Newsgroups, clean retrieval, test split)

| System | Accuracy (pt / range) | Macro-P | Macro-R | Macro-F1 | Retrieval rate |
|---|---|---|---|---|---|
| No Retrieval | 0.60 / 0.57–0.63 | 0.62 | 0.58 | 0.59 | 0.00 |
| Always RAG | 0.72 / 0.69–0.75 | 0.73 | 0.71 | 0.72 | 1.00 |
| Confidence-based | 0.67 / 0.64–0.70 | 0.69 | 0.65 | 0.67 | ~0.55 |
| Entropy-based | 0.67 / 0.63–0.70 | 0.68 | 0.65 | 0.66 | ~0.58 |
| Adaptive-RAG-style | 0.68 / 0.65–0.71 | 0.70 | 0.66 | 0.68 | ~0.50 |
| **SMART-LLM** | **0.685 / 0.66–0.71** | **0.71** | **0.67** | **0.68** | **~0.33** |

**Prediction confidence:** Medium (relative ordering: High; absolute accuracy magnitudes: Medium).
**Reasoning:** 20 Newsgroups is a topical task where retrieved same-topic exemplars usually help, so Always-RAG is predicted to be the clean-accuracy ceiling. SMART-LLM is predicted to sit just below it while retrieving for ~1/3 of inputs, and to match or slightly exceed the other selective baselines at a **lower** retrieval rate — i.e., a better accuracy/retrieval trade-off rather than a higher accuracy peak. We deliberately do **not** predict SMART-LLM > Always-RAG on clean accuracy; claiming so would be inconsistent with the mechanism.

### Table 2 — Predicted router vs. oracle (20 Newsgroups)

| Method (pooling) | Agreement | Precision | Recall | F1 | Mean regret |
|---|---|---|---|---|---|
| SMART-LLM (last) | 0.68 / 0.64–0.72 | 0.72 | 0.62 | 0.67 | 0.15 / 0.10–0.22 |
| SMART-LLM (mean) | 0.64 | 0.70 | 0.57 | 0.63 | 0.18 |
| SMART-LLM (attention) | 0.66 | 0.73 | 0.58 | 0.65 | 0.16 |
| Oracle (upper bound) | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |

**Prediction confidence:** Medium.
**Reasoning:** Agreement is predicted well above the 0.5 chance level but clearly below 1.0, because benefit sign is only partly recoverable from pre-retrieval features. Last-token pooling is predicted strongest (it concentrates the decision-relevant state for an autoregressive model). Precision > recall is predicted: the calibrated rule is conservative, retrieving mainly where utility clearly exceeds confidence, so it misses some beneficial retrievals (lower recall) while being comparatively accurate when it does retrieve.

### Table 3 — Predicted RBE prediction quality (20 Newsgroups, bounded benefit target)

| Target | R² | MAE | Pearson r |
|---|---|---|---|
| B_true (bounded) | 0.30 / 0.20–0.40 | 0.42 / 0.35–0.50 | 0.55 / 0.45–0.65 |

**Prediction confidence:** Low–Medium (this is among the least certain predictions).
**Reasoning:** h_L encodes parametric difficulty and μ_𝒩 encodes retrieval content, but their interaction — whether *this* context helps *this* input — is only partly linearly recoverable from cached features. We predict a positive but moderate R² and correlation, and emphasise that the RBE is expected to be most reliable at the **sign** level (which is what routing needs), not as a precise magnitude regressor. R² is highly sensitive to the benefit distribution and the floor/clip, so its realized value could deviate substantially.

### Table 4 — Predicted ablation (20 Newsgroups; router quality + accuracy)

| Variant | Agreement | F1 | Mean regret | Accuracy | Retrieval rate |
|---|---|---|---|---|---|
| SMART-LLM (full) | 0.68 | 0.67 | 0.15 | 0.685 | 0.33 |
| − RBE (similarity-only) | 0.66 | 0.65 | 0.17 | 0.675 | 0.34 |
| − Calibration (raw RUS) | 0.62 | 0.63 | 0.20 | 0.660 | 0.48 |
| Random (budget-matched) | 0.52 | 0.36 | 0.30 | 0.640 | 0.33 |

**Prediction confidence:** Medium (direction of each effect: High; magnitudes: Medium).
**Reasoning:** Three directional predictions are made with high confidence. (i) Removing **calibration** is predicted to be the most damaging change — agreement falls, regret rises, and retrieval rate inflates (an uncalibrated utility over-fires) — consistent with the design rationale that the rule compares two probabilities. (ii) Removing the **RBE** on a topical dataset is predicted to cost little, because similarity already proxies retrieval usefulness for topic; the Δ over similarity-only routing is predicted small and possibly not significant on 20NG. (iii) The **budget-matched Random** policy is predicted well below all learned routers, isolating that the value is in *deciding well*, not merely in *retrieving less*.

### Table 5 — Predicted calibration (20 Newsgroups)

| Confidence signal | ECE | Brier |
|---|---|---|
| Raw LLM verbalizer confidence | 0.30 / 0.22–0.38 | 0.34 / 0.28–0.40 |
| Calibrated probe C_i | 0.06 / 0.04–0.09 | 0.17 / 0.14–0.20 |

**Prediction confidence:** High.
**Reasoning:** Temperature/Platt scaling reliably reduces the systematic over-confidence of instruction-tuned LLM verbalizer probabilities; a large ECE/Brier improvement is one of the most robust findings in the calibration literature. This is our highest-confidence prediction, and because Proposition-style reasoning ties routing quality to calibration, it also underwrites Table 4's calibration ablation.

### Table 6 — Predicted computational efficiency (20 Newsgroups)

| System | Retrieval rate | Avg prompt tokens | Latency (ms/sample) | Rel. compute |
|---|---|---|---|---|
| No Retrieval | 0.00 | ~350 / 300–420 | ~70 / 55–95 | 0.35 |
| Always RAG | 1.00 | ~1400 / 1200–1600 | ~180 / 150–210 | 1.00 |
| **SMART-LLM** | **~0.33** | **~700 / 600–820** | **~120 / 100–140** | **~0.65** |

**Prediction confidence:** High for ordering and token cost; Medium for absolute latency.
**Reasoning:** Prompt length is dominated by retrieved demonstrations, so token cost tracks retrieval rate almost mechanically; the ordering No-Retr < SMART < Always-RAG is near-certain. Absolute latency depends on batching, KV-cache reuse, and prefill efficiency on the RTX 4090, so the millisecond values carry more uncertainty. We explicitly predict that SMART-LLM is **not** a uniform latency win over No-Retrieval (it always pays one parametric pass); its advantage is fewer augmented passes and shorter prompts than Always-RAG, plus robustness.

### Table 7 — Predicted cross-dataset generalization

| Dataset | RBE R² | Pearson r | Router agreement | Δ Agreement (full − sim-only) | SMART acc | Always-RAG acc | No-retr acc |
|---|---|---|---|---|---|---|---|
| 20 Newsgroups (topic) | 0.30 | 0.55 | 0.68 | +0.02 (predicted n.s.) | 0.685 | 0.72 | 0.60 |
| Financial PhraseBank (sentiment) | 0.40 | 0.62 | 0.70 | +0.06 (predicted sig.) | 0.80 | 0.81 | 0.78 |

**Prediction confidence:** Medium (topic row); Low–Medium (the cross-dataset Δ Agreement claim is the single riskiest prediction).
**Reasoning:** The central mechanistic prediction is that the **learned benefit term earns its place where similarity is a weak cue**. Sentiment polarity is less aligned with embedding similarity than topic, so we predict the RBE's marginal gain over similarity-only routing is larger and (unlike 20NG) statistically significant on Financial PhraseBank, with a higher RBE R². We also predict retrieval helps *less* overall on sentiment (No-Retr already high ~0.78), so all three systems cluster. This row is where real results are most likely to diverge from the forecast.

---

## 6  Analysis

**Why retrieval sometimes hurts.** Retrieved demonstrations enter the prompt as evidence the model attends to. When neighbours share the true label they reinforce the correct class; when they are topically similar but wrong-labelled, redundant, or adversarial, they can shift probability mass toward an incorrect class and override a correct parametric prediction. The expected harm grows as retrieval quality degrades and as the parametric model's own competence on the input rises (there is more to lose). We therefore predict that always-on retrieval will fall **below** the no-retrieval baseline under a constructed adversarial condition (injected hard negatives), while a selective policy that declines low-utility retrievals will not.

**Why learned arbitration should outperform similarity-only retrieval — and when it should not.** Similarity measures whether neighbours are *close*, not whether they are *useful for this decision*. On topic classification the two coincide strongly, so we predict the learned RBE adds little over similarity-only routing on 20 Newsgroups. On sentiment, closeness in embedding space is a weaker proxy for label-usefulness, so we predict the RBE's model-internal signal (via h_L) contributes information similarity alone lacks. This is a conditional, testable prediction, not a universal claim.

**Failure cases (anticipated).** (i) *Confidently wrong parametric answers* — high C_i but incorrect — suppress beneficial retrieval; this failure is bounded by calibration quality but not eliminated. (ii) *Misleading similar neighbours* inflate RUS and trigger harmful retrieval. (iii) *RBE mis-estimation* on out-of-distribution inputs mis-routes. (iv) *Calibration drift* under distribution shift decouples C_i from true correctness. These are the structural failure modes of any confidence-plus-utility rule.

**Expected limitations.** Two datasets and two task types do not establish general regimes; the method always pays one parametric pass (so it is not advantageous when retrieval is nearly always correct and cheap); benefit magnitude is predicted only partly learnable; and the auxiliary Mixture-of-LoRA and explanation-verification components are not expected to carry the core result.

---

## 7  Case Studies (Predicted qualitative behavior)

> The following ten cases describe **predicted qualitative behavior (before execution)**; they illustrate the mechanism and are not transcripts of real runs.

**Predicted successful cases**
1. *Ambiguous cross-posted 20NG document* (reads between `comp.graphics` and `comp.os.ms-windows.misc`): parametric confidence low, neighbour similarity high and same-label → cal(RUS) > C_i → SMART retrieves → predicted correct. Mechanism: uncertainty opens the gate; useful neighbours cross it.
2. *Clear-cut `rec.sport.baseball` post*: high C_i, low predicted benefit → ΔC < 0 → SMART skips retrieval → predicted correct at reduced compute. Mechanism: confidence vetoes unnecessary retrieval.
3. *Negated financial sentence* ("results were not as weak as feared"): parametric model wavers between neutral/positive; similar labelled exemplars disambiguate → SMART retrieves → predicted correct. Mechanism: benefit predicted where polarity cues are subtle.
4. *Adversarially corrupted retrieval*: injected hard negatives lower sim and predicted utility → RUS low → SMART declines retrieval → predicted correct where Always-RAG is misled. Mechanism: the safeguard always-on policies lack.
5. *Rare-class financial input with weak neighbour similarity but informative hidden state*: sim low yet RBE predicts benefit from h_L → SMART retrieves → predicted correct. Mechanism: the learned term acting beyond similarity (the sentiment-regime hypothesis).

**Predicted failure cases**
1. *Confidently wrong parametric answer* (high C_i, incorrect): ΔC < 0 suppresses beneficial retrieval → predicted wrong. Bounded by calibration but not removed.
2. *Topically similar but misleading neighbour*: high sim inflates RUS → SMART retrieves → context corrupts an otherwise-correct parametric prediction → predicted wrong.
3. *Out-of-distribution input for the RBE*: benefit mis-estimated → mis-routing in either direction → predicted wrong.
4. *Genuinely ambiguous multi-topic document*: neither parametric nor retrieval path is correct → predicted wrong regardless of routing (a data-limited, not method-limited, failure).
5. *Calibration drift under domain shift* (probe fit on one corpus, applied to another): C_i decouples from true correctness → systematic routing errors → predicted wrong on a subset.

---

## 8  Discussion

**Scientific contribution (as proposed).** The framing of the retrieval decision as *pre-retrieval benefit prediction* scored against an oracle by agreement and regret, and the explicit dependence of routing quality on calibration, are the intended contributions. If the predictions hold, the practical value is a favourable accuracy/compute trade-off and a robustness safeguard, not a new accuracy ceiling.

**Limitations.** Narrow task coverage (topic + sentiment); dependence on retriever and embedding quality; the always-paid parametric pass; partial predictability of benefit magnitude; auxiliary components not central.

**Expected reviewer criticism (anticipated, with our planned responses).**
- *"The RBE adds little."* Predicted true on topic; we position it as conditional and test the sentiment regime (Table 7). If it fails there too, we will report the RBE as a negative result and rest the contribution on calibrated routing + robustness.
- *"The adversarial condition is a strawman."* Agreed; we present it as a constructed worst case and rely on the budget-matched Random comparison (Table 4) as the honest like-for-like evidence.
- *"Only two datasets; single backbone."* Acknowledged as the primary threat to generality; framed as the central open question.
- *"The theory is idealised."* The optimality argument assumes perfect calibration and a correct benefit-probability model; we present it as design rationale, not a guarantee, and measure residuals empirically.
- *"Mixture-of-LoRA and explanation verification are underdeveloped."* We scope them as auxiliary and expect no core claim to depend on them.

**How experiments may falsify the hypothesis.** The central hypothesis — that an LLM can profitably learn *when* retrieval helps — is falsified if, on real data, a learned selective policy cannot beat a budget-matched random policy (agreement ≈ chance; accuracy-at-budget ≤ Random), if calibration does not improve routing, or if a single confidence threshold dominates the full method on the accuracy/compute frontier. These are explicit, pre-registered failure criteria.

---

## 9  Conclusion and Prediction Reliability Assessment

We have described SMART-LLM and, without conducting experiments, **predicted** the likely outcomes of a planned RTX 4090 evaluation, with per-table confidence and reasoning. The forecast's core is: selective arbitration recovers most of always-on retrieval's accuracy benefit at ~1/3 the retrieval rate; calibration is the most important component; the learned benefit term matters mainly where similarity is weak; and the clearest advantage over always-on retrieval appears under corrupted retrieval.

### Prediction Reliability Assessment

**1. Highly reliable predictions (High confidence).**
- Large calibration improvement from temperature/Platt scaling (Table 5).
- Compute/token ordering No-Retrieval < SMART-LLM < Always-RAG (Table 6).
- Removing calibration degrades routing and inflates retrieval rate (Table 4).
- Always-on retrieval degrades (and can fall below no-retrieval) under adversarial retrieval.
- Learned routers beat a budget-matched Random policy (Table 4).

**2. Uncertain predictions (Medium–Low confidence).**
- Absolute accuracy magnitudes for all systems (±5–8 absolute points is plausible).
- RBE R² and Pearson r (Table 3) — sensitive to the benefit distribution and floor/clip.
- Exact retrieval rates and the exact accuracy gap between SMART-LLM and Always-RAG.
- The cross-dataset Δ Agreement significance claim (Table 7) — the single riskiest prediction.
- Millisecond-level latency (implementation-dependent).

**3. Outcomes that would invalidate SMART-LLM.**
- Router oracle agreement ≈ 0.5 (no learnable signal), or SMART accuracy-at-budget ≤ budget-matched Random.
- Calibration failing to improve routing quality.
- RBE R² ≤ 0 *and* similarity-only ≥ full on **every** dataset (the learned term never helps).
- A single confidence/entropy threshold Pareto-dominating the full method on accuracy vs. compute.
- Always-on retrieval never being harmful in any realistic condition (removing the reason to arbitrate).

**4. Expected differences between prediction and real RTX 4090 results.**
I expect the **directions and orderings** to hold more often than the **magnitudes**. Realistic expectation: most High-confidence directional predictions hold; absolute accuracies land within roughly ±5–8 points of the point forecasts; retrieval rates within ±10–15 points; RBE R² potentially off by ±0.15; latency values the most likely to differ due to serving details. The riskiest items are Table 3 (RBE magnitude) and the Table 7 sentiment-regime claim.

**A necessary caution on expecting an exact match.** A pre-experimental forecast that reproduced real results *identically* would be a statistical coincidence, not evidence of good reasoning — genuine measurements carry seed variance, implementation choices, and data idiosyncrasies that no prior forecast can hit to the decimal. The scientific success criterion here is **calibration**, not identity: that the real numbers fall within the stated ranges, that the High-confidence directional claims hold, and that any pre-registered falsification condition either fails to trigger or, if it does, is reported honestly. Treating a non-identical outcome as a failure of the prediction would itself be a methodological error; treating it as a test of calibrated reasoning is the intended use.

---

*Reminder: this is a scientific forecast manuscript. No experiments have been conducted. Every quantitative value is an AI-predicted experimental outcome (before execution), to be compared against real results and revised accordingly.*
