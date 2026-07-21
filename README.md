# SMART-LLM

**Decision-Time Retrieval Arbitration for Efficient and Robust Few-Shot Large
Language Model Text Classification**

> **Thesis.** The contribution is **decision-time retrieval benefit estimation** —
> deciding whether retrieval will help *before* retrieving — not a
> "RAG + LoRA + explainability" pipeline. SMART-LLM predicts retrieval utility
> from pre-retrieval features and routes on a calibrated comparison with internal
> confidence, so it never runs retrieval-augmented inference to make the decision
> (**no double inference**).

This repository implements SMART-LLM and, on a GPU machine, runs one **focused,
ten-part behavioural study** on **20 Newsgroups** (depth over breadth — a
behavioural evaluation of retrieval arbitration, not a benchmark sweep).

> **Environment note.** This code is written to run on a **CUDA GPU machine**
> (primary backbone: `Qwen/Qwen2.5-7B-Instruct`, ~16 GB in bf16). It was authored
> on a CPU-only box and is *not* meant to be executed there. No experimental
> numbers are shipped in this repo — every table/figure/number in the manuscript
> is generated from the CSVs your GPU run produces (`results/`). See
> [`REPRODUCE.md`](REPRODUCE.md).

---

## Contributions (one primary, one key innovation, two supporting)

1. **CDKA — Confidence-Driven Knowledge Arbitration** *(primary).* A
   decision-theoretic rule that routes retrieval by comparing a calibrated
   internal confidence `C_i` with a predicted, calibrated retrieval utility —
   with **no double inference**.
2. **RBE — Retrieval Benefit Estimator** *(key innovation).* Predicts the loss
   reduction from retrieval, `B_pred = RBE([h_L ‖ μ_K])`, from **pre-retrieval**
   features; evaluated against an oracle (R², MAE, Pearson, regret).
3. **UAAS — Uncertainty-Aware Adapter Scaling** *(supporting).* Per-input LoRA
   rank `r(x)` from uncertainty; vs. static LoRA `r ∈ {4, 16, 32}`.
4. **Attribution-Guided Explanation Verification** *(supporting).* Integrated
   Gradients test of whether explanations reference the tokens that drove the
   prediction.

### The ten-part study (one dataset, many analyses)

overall performance · router-vs-oracle · RBE prediction · retrieval behaviour ·
noise robustness (clean/random/adversarial) · calibration · module ablation ·
difficulty strata · qualitative case study · computation. → 7 core tables + 8
figures, each answering one scientific question. See `REPRODUCE.md`.

## Pipeline at a glance

The expensive LLM work (Stage 1) is **separated** from the cheap probe/RBE
training (Stage 2), so the 7B forward passes run **once** and all of CDKA can be
re-trained/ablated in seconds on cached features.

```
Stage 1 (GPU, heavy)                 Stage 2 (light, CPU-ok)
─────────────────────                ───────────────────────
generate_features.py   ── cache ──▶  train_cdka.py ──▶ results/*.csv
  • frozen LLM forward passes          • confidence probe          │
  • per-token hidden states            • RBE (retrieval benefit)   ▼
  • Loss_p / Loss_r (GT benefit)       • RUS calibration       analysis/  ──▶ figures/ tables/
  • bge embeddings + FAISS             • router vs oracle           │
  • retrieval conditions                                           ▼
                                                              paper/*.docx
```

## Repository layout

```
smart_llm/
├── configs/                 # YAML configs (default + per-dataset + per-model)
├── smart_llm/               # the package
│   ├── config.py            # typed config (dataclasses) + YAML/CLI override
│   ├── utils/               # seeding, logging, device, io/caching
│   ├── data/                # dataset loaders (20NG + AG News/TweetEval/FPB/PubMed), few-shot
│   ├── embeddings/          # bge/minilm encoder + FAISS index + retrieval conditions
│   ├── llm/                 # frozen backbone, pooling, verbalizer classifier+loss, prompts
│   ├── cdka/                # confidence probe, RBE, RUS calibration, router
│   ├── uaas/                # adaptive LoRA rank (Contribution 2)
│   ├── explain/             # Integrated-Gradients explanation verification (Contribution 3)
│   ├── experiments/         # Stage-1 features + Stage-2 training + exp1/2/3 drivers
│   ├── analysis/            # metrics, figures (2-5), tables (1-5)
│   └── paper/               # DOCX manuscript + supplementary generators
├── scripts/                 # end-to-end run scripts
├── tests/                   # unit tests for pure-logic components (run on CPU)
├── requirements.txt
├── REPRODUCE.md
└── README.md
```

## Quick start (RTX 4090, 24 GB)

```bash
pip install -r requirements.txt        # install a CUDA torch build first

# Core behavioural study through the manuscript (Stage 1 is the only heavy step):
bash scripts/run_phase1a.sh configs/default.yaml 20newsgroups

# Everything (adds UAAS + explanation verification):
bash scripts/run_all.sh configs/default.yaml 20newsgroups
```

Step-by-step commands and the analysis→table/figure map are in
[`REPRODUCE.md`](REPRODUCE.md). Fast sanity check on a small model:
`bash scripts/run_all.sh configs/debug.yaml 20newsgroups`.

## Scientific-integrity rules honored by this code

- **No invented numbers.** Manuscript generators read `results/*.csv` and refuse
  to emit unfilled tables; missing values render as `[[TBD-from-run]]`.
- **No double inference in the router.** The router consumes `h_L` (already
  computed on the no-retrieval path) + the retrieval centroid `μ_K` (cheap) +
  `B_pred`; the retrieval forward pass is only run when the router decides to
  retrieve. Ground-truth `Loss_r` is computed **offline** for supervision only.
- **Claims language.** We say *demonstrated / validated experimentally / provided
  evidence*, never *proved*.

See [`REPRODUCE.md`](REPRODUCE.md) for exact commands, hyperparameters, and the
data-logging schema.
