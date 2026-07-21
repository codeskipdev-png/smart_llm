# Reproducing SMART-LLM

A **focused, single-dataset behavioural study** of decision-time retrieval
arbitration on 20 Newsgroups. Every number in the manuscript is generated from
per-sample logs — nothing is precomputed or invented. **Heavy steps need a CUDA
GPU (target: RTX 4090, 24 GB).**

---

## 0. Environment (RTX 4090)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -r requirements.txt        # or: pip install -e .
```

`Qwen/Qwen2.5-7B-Instruct` in bf16 (~16 GB) + `bge-large` fit comfortably in
24 GB. If VRAM is tight set `--set llm.load_in_4bit=true` (needs `bitsandbytes`).
Attribution runs the 7B in bf16 with batched IG steps (`explain.ig_internal_batch`)
so it also fits 24 GB.

Fast sanity check (0.5B model, tiny data):
```bash
bash scripts/run_all.sh configs/debug.yaml 20newsgroups
```

### 0b. No local GPU? Rent one (the heavy stage is the only GPU step)

Only `generate_features` (Stage 1) needs a GPU; everything else runs on CPU in
seconds off the cache. On a fresh cloud box (RunPod / Lambda / Vast / any
single 24 GB card):

```bash
git clone <this repo> && cd smart_llm
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
export HF_HUB_DISABLE_XET=1              # classic HTTPS downloads
# full multi-seed study, both datasets, ~a few GPU-hours at max_eval=1500:
bash scripts/run_cross.sh configs/default.yaml 20newsgroups twitter_financial
python -m smart_llm.experiments.ablation --config configs/default.yaml \
       --dataset 20newsgroups --seeds 0,1,2,3,4     # multi-seed ablation + baselines
python -m smart_llm.paper.make_manuscript --config configs/default.yaml \
       --dataset 20newsgroups                        # real numbers replace TBD
```

Then copy `runs/**/paper/SMART_LLM_main.docx` back. **Without a GPU you cannot
produce real numbers** — the two document generators emit clearly-marked
placeholders (`make_full_paper.py`) or `[[TBD-from-run]]` (`make_manuscript.py`),
never fabricated measurements. Validate the CPU logic first with the smoke test
in §6 before renting time.

### 0c. What is illustrative vs. real

`make_full_paper.py` builds `runs/paper_full/SMART_LLM_full.docx` from a single
source-of-truth dict of **illustrative template numbers** (internally consistent,
NOT measurements) so the structure, positioning, proofs, baselines, and
statistical protocol can be reviewed before the run. `make_manuscript.py` builds
the submission doc from per-sample logs and renders `[[TBD-from-run]]` for any
table that has not been produced yet. Only the latter is citable.

---

## 1. Execution order (copy/paste)

```bash
CFG=configs/default.yaml ; DS=20newsgroups

# --- Stage 1 (HEAVY, GPU): frozen features + ground-truth benefit + timings ---
python -m smart_llm.experiments.generate_features --config $CFG --dataset $DS

# --- Stage 2 (light): CDKA training -> master CSV + metrics JSON ---
python -m smart_llm.experiments.train_cdka        --config $CFG --dataset $DS

# --- Module ablation (Analysis 7 -> Table 6 / Figure 7) ---
python -m smart_llm.experiments.ablation          --config $CFG --dataset $DS

# --- Qualitative case study (Analysis 9 -> Figure 8, case_study.md) [uses LLM] ---
python -m smart_llm.experiments.case_study        --config $CFG --dataset $DS

# --- Supporting Contribution 2: UAAS (adaptive LoRA rank) [uses LLM] ---
python -m smart_llm.experiments.train_uaas        --config $CFG --dataset $DS --train-samples 1000

# --- Supporting Contribution 3: explanation verification [uses LLM] ---
python -m smart_llm.experiments.explain_verify    --config $CFG --dataset $DS

# --- Tables (1-7 + behaviour + difficulty) and Figures (1-8) ---
python -m smart_llm.analysis.make_all             --config $CFG --dataset $DS

# --- Manuscript + supplementary (numbers pulled from the CSVs) ---
python -m smart_llm.paper.make_manuscript         --config $CFG --dataset $DS
python -m smart_llm.paper.make_supplementary      --config $CFG --dataset $DS
```

One-liners:
* Core study (through the paper, incl. case study): `bash scripts/run_phase1a.sh $CFG $DS`
* Everything (adds UAAS + explanation verification): `bash scripts/run_all.sh $CFG $DS`

Only `generate_features` is expensive. Everything else consumes the cache, so you
can re-run CDKA/ablation/analysis/paper in seconds. Use `--limit N` on Stage 1 for
a dry run; lower `data.max_eval` to shorten a full run.

### 1b. Cross-dataset generalization (does the learned RBE earn its place?)

20 Newsgroups is strongly similarity-shaped (topic), so similarity-only routing is
hard to beat. Financial PhraseBank (sentiment) is a weaker-similarity regime where
the *learned* benefit estimator should contribute more — the decisive test for
contribution #1. Both datasets share one output root so they compare directly.

```bash
# second dataset end-to-end + comparison + primary manuscript rebuild (one command):
bash scripts/run_cross.sh configs/default.yaml 20newsgroups financial_phrasebank
```
or manually:
```bash
bash scripts/run_phase1a.sh configs/default.yaml financial_phrasebank
python -m smart_llm.analysis.cross_dataset --config configs/default.yaml \
       --datasets 20newsgroups,financial_phrasebank
python -m smart_llm.paper.make_manuscript --config configs/default.yaml --dataset 20newsgroups
```
Outputs: `tables/cross_dataset_comparison.csv`, `figures/figure9_cross_dataset.*`, and
a new manuscript **Analysis 11** whose text is computed from the comparison — it
credits the learned RBE only if its gain over similarity-only routing is actually
larger on Financial PhraseBank, and says so honestly otherwise.

---

## 2. The ten analyses -> outputs

| # | Analysis | Evidence |
|---|----------|----------|
| 1 | Overall performance (acc / macro P,R,F1 / latency / freq) | Table 1 |
| 2 | Router vs oracle (agreement / P / R / F1 / regret) | Table 2, Figure 2 |
| 3 | RBE prediction (R² / MAE / Pearson / residuals) | Table 3, Figure 4 |
| 4 | Retrieval behaviour (freq / demos / prompt length / margins) | Table (behaviour), Figure 5 |
| 5 | Noise robustness (clean / random / adversarial) | Table 4, Figure 6 |
| 6 | Calibration (ECE / Brier / reliability) | Table 5, Figure 3 |
| 7 | Ablation (− RBE / − Calibration) **+ external decision-policy baselines** (Random budget-matched, Confidence-gated, Entropy-gated / Adaptive-RAG-style) with 95% bootstrap CIs and paired McNemar/bootstrap significance vs. SMART | Table 6, Figure 7 |
| 8 | Difficulty (easy / medium / hard) | Table (difficulty) |
| 9 | Qualitative case study (5 success + 5 failure) | Figure 8, `results/case_study_*.md` |
| 10 | Computation (latency / retrieval reduction / compute / tokens) | Table 7 |

Systems compared throughout: **No retrieval**, **Always RAG**, **SMART-LLM**.

---

## 3. Ground-truth benefit (offline supervision only)

```
B_true = (Loss_p - Loss_r) / (|Loss_p| + eps)      eps = 1e-6
oracle = 1[Loss_r < Loss_p]
regret = chosen_loss - min(Loss_p, Loss_r)
```

`Loss_p`/`Loss_r` are label cross-entropies under the letter verbalizer (one
forward pass each). The retrieval-augmented pass is used **only** to build this
supervision; the deployed arbiter never runs it to decide (no double inference).

---

## 4. Master logging schema (`results/master_<dataset>.csv`)

One row per (pooling × retrieval-condition × sample):

`id, dataset, label, pooling, condition, split, C_i, entropy, probe_pred,
conf_llm, entropy_llm, sim, B_pred, B_true, RUS, calibrated_RUS, delta_C,
smart_decision, oracle_decision, loss_without_retrieval, loss_with_retrieval,
pred_p, pred_r, smart_pred, regret, t_p, t_r, n_tokens_p, n_tokens_r`

Amortised embedding/index/search timings live in `cdka_metrics_<dataset>.json`.
Missing result files render as `[[TBD-from-run]]` in the manuscript — never faked.

---

## 5. Determinism, resources, honesty notes

* One seed drives numpy/torch/splits; deterministic matmul is enabled. The
  **multi-seed** aggregate (`ablation --seeds 0,1,2,3,4`) re-draws the split and
  re-fits the probe/RBE/calibrator per seed on the SAME cached features (no LLM
  re-run), and reports seed-mean ± across-seed std alongside per-sample bootstrap
  CIs. Significance uses a paired McNemar test (accuracy) and a paired bootstrap
  (mean regret / oracle agreement); differences are marked `*` (p<0.05) or `n.s.`
* Stage-1 cost ≈ `max_eval × (1 + |conditions|)` 7B passes (defaults 1500 × 4).
* **Latency is reported honestly:** SMART pays a parametric pass to obtain `C_i`
  and `h_L`, so its advantage is *fewer augmented passes and robustness*, not a
  uniform latency win (Table 7 / Figure 6 make this explicit).
* `pubmed` uses HF id `armanc/pubmed-rct20k`; if it 404s in your cache, edit the
  preset in `smart_llm/data/datasets.py`. (Not needed for the 20NG study.)

---

## 6. Tests (no GPU)

```bash
pip install pytest && pytest -q
```

Validates the pure-logic components (metrics incl. macro P/R/F1, Brier,
reliability, difficulty; probe, RBE, calibration, router, UAAS schedule, config)
and runs a full **synthetic** Stage-2 → master-CSV → tables/figures integration
with no LLM or FAISS.
