#!/usr/bin/env bash
# Full study: core CDKA/RBE behavioural study + supporting UAAS (Contribution 2)
# and explanation verification (Contribution 3), then rebuild the manuscript.
# Usage:  bash scripts/run_all.sh [CONFIG] [DATASET]
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"
DATASET="${2:-20newsgroups}"
PY="python -m"

# core study (Stage 1/2 + ablation + case study + tables/figures + paper)
bash scripts/run_phase1a.sh "$CONFIG" "$DATASET"

echo "=== Supporting Contribution 2: UAAS (adaptive LoRA rank) ==="
$PY smart_llm.experiments.train_uaas --config "$CONFIG" --dataset "$DATASET" --train-samples 1000

echo "=== Supporting Contribution 3: attribution-guided explanation verification ==="
$PY smart_llm.experiments.explain_verify --config "$CONFIG" --dataset "$DATASET"

echo "=== Rebuild manuscript + supplementary with UAAS/explanation tables ==="
$PY smart_llm.analysis.make_all        --config "$CONFIG" --dataset "$DATASET"
$PY smart_llm.paper.make_manuscript    --config "$CONFIG" --dataset "$DATASET"
$PY smart_llm.paper.make_supplementary --config "$CONFIG" --dataset "$DATASET"

echo "=== Full study complete ==="
