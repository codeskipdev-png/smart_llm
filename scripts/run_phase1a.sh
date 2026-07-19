#!/usr/bin/env bash
# SMART-LLM core behavioural study (CDKA + RBE) on one dataset, CUDA GPU.
# Produces the master log, all core tables/figures, the case study, and the paper.
# Usage:  bash scripts/run_phase1a.sh [CONFIG] [DATASET]
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"
DATASET="${2:-20newsgroups}"
PY="python -m"

# reduce CUDA fragmentation on 24GB cards
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== SMART-LLM behavioural study | config=$CONFIG dataset=$DATASET ==="

# Stage 1 (heavy, GPU): frozen features + ground-truth retrieval benefit + timings
$PY smart_llm.experiments.generate_features --config "$CONFIG" --dataset "$DATASET"

# Stage 2 (light): CDKA training + router evaluation -> master CSV + metrics
$PY smart_llm.experiments.train_cdka --config "$CONFIG" --dataset "$DATASET"

# Module ablation (Analysis 7 -> Table 6 / Figure 7)
$PY smart_llm.experiments.ablation --config "$CONFIG" --dataset "$DATASET"

# Qualitative case study (Analysis 9 -> Figure 8 + case_study.md)  [uses the LLM]
$PY smart_llm.experiments.case_study --config "$CONFIG" --dataset "$DATASET"

# Tables (1-7 + difficulty/behaviour) and Figures (1-8)
$PY smart_llm.analysis.make_all --config "$CONFIG" --dataset "$DATASET"

# Manuscript + supplementary (numbers pulled from the CSVs above)
$PY smart_llm.paper.make_manuscript   --config "$CONFIG" --dataset "$DATASET"
$PY smart_llm.paper.make_supplementary --config "$CONFIG" --dataset "$DATASET"

echo "=== Done. See runs/*/results, runs/*/tables, runs/*/figures, runs/*/paper ==="
