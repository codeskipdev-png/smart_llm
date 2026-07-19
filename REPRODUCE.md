# Reproducing SMART-LLM

This file gives the exact commands, the two-stage design, the data-logging schema,
and the resource notes needed to reproduce every number in the manuscript. **All
heavy steps require a CUDA GPU.** Nothing in this repo ships precomputed numbers;
the manuscript is generated from the CSVs you produce.

---

## 0. Environment

```bash
# 1) install a CUDA build of torch that matches your driver, e.g. CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121
# 2) the rest:
pip install -r requirements.txt        # or: pip install -e .
```

Backbone download: `Qwen/Qwen2.5-7B-Instruct` (~16 GB bf16). Set
`llm.load_in_4bit=true` (needs `bitsandbytes`) if VRAM < 20 GB. Gated models
(Llama-3.1) require `huggingface-cli login`.

Fast end-to-end sanity check (0.5B backbone, tiny data — runs on a small GPU):
```bash
bash scripts/run_all.sh configs/debug.yaml 20newsgroups
```

---

## 1. Two-stage design (why it is cheap to iterate)

The 7B forward passes are isolated in **Stage 1** and cached. Everything else
(**Stage 2** CDKA training, ablations, analysis, paper) reads the cache, so you
re-train/ablate the router in seconds without touching the LLM.

| Stage | Script | Cost | Produces |
|------|--------|------|----------|
| 1 | `smart_llm.experiments.generate_features` | heavy (GPU) | `runs/*/cache/<dataset>/` |
| 2 | `smart_llm.experiments.train_cdka` | light | `results/master_<dataset>.csv`, `cdka_metrics_<dataset>.json` |
| 2b | `smart_llm.experiments.ablation` | light | `tables/table6_rus_ablation_*.csv` |
| 3 | `smart_llm.analysis.make_all` | light | `tables/*.csv`, `figures/*.png,pdf` |
| 4 | `smart_llm.paper.make_manuscript` / `make_supplementary` | light | `paper/*.docx` |

Per-stage commands (Phase-1A):
```bash
CFG=configs/default.yaml ; DS=20newsgroups
python -m smart_llm.experiments.generate_features --config $CFG --dataset $DS
python -m smart_llm.experiments.train_cdka        --config $CFG --dataset $DS
python -m smart_llm.experiments.ablation          --config $CFG --dataset $DS
python -m smart_llm.analysis.make_all             --config $CFG --dataset $DS
python -m smart_llm.paper.make_manuscript         --config $CFG --dataset $DS
python -m smart_llm.paper.make_supplementary      --config $CFG --dataset $DS
```
or just `bash scripts/run_phase1a.sh $CFG $DS`.

Contributions 2 & 3 (Phase 2):
```bash
python -m smart_llm.experiments.train_uaas     --config $CFG --dataset $DS --train-samples 1000
python -m smart_llm.experiments.explain_verify --config $CFG --dataset $DS
```

Other datasets: `--dataset {agnews, tweeteval, financial_phrasebank, pubmed}`.
Secondary backbone: `--config configs/llama.yaml`.

---

## 2. Which experiment maps to which output

* **Experiment 1 — pooling** (last / mean / attention): Tables 2 & 5, Figure 2.
  Driven by `cdka_metrics_*.json` (RBE R², routing agreement) and the master CSV.
* **Experiment 2 — retrieval conditions** (clean / random / adversarial): Table 3,
  Figure 3. The frozen router (fit on clean) is applied to every condition.
* **Experiment 3 — routing** (No-retrieval / Always-RAG / SMART): Tables 1 & 4,
  Figures 4 & 5.
* **Router-signal ablation**: Table 6 (`ablation.py`).
* **UAAS**: `table_uaas_*.csv`. **Explanation verification**: `table_explain_*.csv`.

---

## 3. Ground-truth generation (offline supervision)

For each sample, Stage 1 runs the frozen LLM without and with retrieval:

```
B_true = (Loss_p - Loss_r) / (|Loss_p| + eps)      eps = 1e-6
oracle = 1[Loss_r < Loss_p]
```

`Loss_p`, `Loss_r` are cross-entropies of the gold label under the letter
verbalizer (single forward pass each). The router is evaluated by: RBE **R²** vs
`B_true`; router **oracle-agreement**; and **regret = chosen_loss − min(Loss_p,
Loss_r)**. The retrieval-augmented pass is used here for supervision only — at
deployment the router never runs it to decide (no double inference).

---

## 4. Master logging schema (`results/master_<dataset>.csv`)

One row per (pooling × retrieval-condition × sample):

`id, dataset, label, pooling, condition, split, C_i, entropy, conf_llm,
entropy_llm, sim, B_pred, B_true, RUS, calibrated_RUS, delta_C, smart_decision,
oracle_decision, loss_without_retrieval, loss_with_retrieval, pred_p, pred_r,
smart_pred, regret, t_p, t_r`

Every table and figure is derived from this file plus `cdka_metrics_*.json`
(which additionally stores amortised embedding/index/search timings). If any
result file is missing, the manuscript generator emits `[[TBD-from-run]]` rather
than inventing a value.

---

## 5. Determinism & resources

* One global seed (`seed:`) drives numpy/torch/splits. Set
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` is done automatically for deterministic matmul.
* Stage 1 rough cost: `n_eval × (1 + |conditions|)` 7B forward passes
  (defaults: 1500 × 4 = 6000). Budget accordingly; use `--limit N` to dry-run.
* Stage 2/analysis/paper run on CPU in seconds-to-minutes.

---

## 6. Tests (run anywhere, no GPU)

```bash
pip install pytest
pytest -q
```

The suite validates the pure-logic components (metrics, pooling, probe, RBE,
calibration, router, UAAS rank schedule, config) and runs a full **synthetic**
Stage-2 → master-CSV → tables integration without any LLM or FAISS.
