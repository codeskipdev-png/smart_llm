# SMART-LLM

**Confidence-Driven Knowledge Arbitration and Adaptive Adapter Scaling for
Explainable Few-Shot Large Language Model Text Classification**

This repository implements and (on a GPU machine) experimentally validates
SMART-LLM, an uncertainty-driven adaptive-inference framework for few-shot text
classification with open-weight LLMs.

> **Environment note.** This code is written to run on a **CUDA GPU machine**
> (primary backbone: `Qwen/Qwen2.5-7B-Instruct`, ~16 GB in bf16). It was authored
> on a CPU-only box and is *not* meant to be executed there. No experimental
> numbers are shipped in this repo — every table/figure/number in the manuscript
> is generated from the CSVs your GPU run produces (`results/`). See
> [`REPRODUCE.md`](REPRODUCE.md).

---

## Three scientific contributions

1. **CDKA — Confidence-Driven Knowledge Arbitration** *(Phase 1A, fully
   implemented & the focus of the first validation).* Decide *per input* whether
   retrieval will help, **without** running full RAG first (no double inference).
2. **UAAS — Uncertainty-Aware Adapter Scaling.** Choose LoRA rank `r(x)` per input
   from an uncertainty signal; compared against static LoRA `r ∈ {4, 16, 32}`.
3. **Attribution-Guided Explanation Verification.** Use Integrated Gradients to
   test whether the tokens that actually drove the prediction appear in the
   generated explanation.

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

## Quick start (GPU machine)

```bash
pip install -r requirements.txt

# Stage 1: extract frozen features + ground-truth retrieval benefit (heavy, GPU)
python -m smart_llm.experiments.generate_features --config configs/default.yaml \
       --dataset 20newsgroups

# Stage 2: train CDKA (probe + RBE + calibration), evaluate router vs oracle (light)
python -m smart_llm.experiments.train_cdka --config configs/default.yaml \
       --dataset 20newsgroups

# Phase-1A experiments (produce the master results CSV)
python -m smart_llm.experiments.run_phase1a --config configs/default.yaml \
       --dataset 20newsgroups

# Analysis -> figures + tables ; then the manuscript
python -m smart_llm.analysis.make_all      --config configs/default.yaml
python -m smart_llm.paper.make_manuscript  --config configs/default.yaml
python -m smart_llm.paper.make_supplementary --config configs/default.yaml
```

The full sequence is wrapped in [`scripts/run_phase1a.sh`](scripts/run_phase1a.sh).

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
