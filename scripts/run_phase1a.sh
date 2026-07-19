#!/usr/bin/env bash
# End-to-end Phase-1A (CDKA validation) on a CUDA GPU machine.
# Usage:  bash scripts/run_phase1a.sh [CONFIG] [DATASET]
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"
DATASET="${2:-20newsgroups}"
PY="python -m"

echo "=== SMART-LLM Phase-1A | config=$CONFIG dataset=$DATASET ==="

# Stage 1 (heavy, GPU): frozen features + ground-truth retrieval benefit
$PY smart_llm.experiments.generate_features --config "$CONFIG" --dataset "$DATASET"

# Stage 2 (light): CDKA training + router evaluation -> master CSV + metrics
$PY smart_llm.experiments.train_cdka --config "$CONFIG" --dataset "$DATASET"

# Router-signal ablation (supplementary Table 6)
$PY smart_llm.experiments.ablation --config "$CONFIG" --dataset "$DATASET"

# Tables (1-5) + figures (1-5)
$PY smart_llm.analysis.make_all --config "$CONFIG" --dataset "$DATASET"

# Manuscript + supplementary (numbers pulled from the CSVs above)
$PY smart_llm.paper.make_manuscript   --config "$CONFIG" --dataset "$DATASET"
$PY smart_llm.paper.make_supplementary --config "$CONFIG" --dataset "$DATASET"

echo "=== Done. See runs/*/results, runs/*/tables, runs/*/figures, runs/*/paper ==="
