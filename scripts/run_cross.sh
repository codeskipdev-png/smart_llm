#!/usr/bin/env bash
# Add a second dataset and build the cross-dataset generalization comparison.
# Assumes the PRIMARY dataset has already been run through make_all.
# Usage:  bash scripts/run_cross.sh [CONFIG] [PRIMARY] [SECOND]
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"
PRIMARY="${2:-20newsgroups}"
SECOND="${3:-financial_phrasebank}"
PY="python -m"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

echo "=== Second dataset: $SECOND (full pipeline) ==="
# Stage 1 is the only heavy step; both datasets share one output root so the
# cross-dataset comparison can read them together.
bash scripts/run_phase1a.sh "$CONFIG" "$SECOND"

echo "=== Cross-dataset comparison: $PRIMARY vs $SECOND ==="
$PY smart_llm.analysis.cross_dataset --config "$CONFIG" --datasets "$PRIMARY,$SECOND"

echo "=== Rebuild PRIMARY manuscript with the cross-dataset section (Analysis 11) ==="
$PY smart_llm.paper.make_manuscript    --config "$CONFIG" --dataset "$PRIMARY"
$PY smart_llm.paper.make_supplementary --config "$CONFIG" --dataset "$PRIMARY"

echo "=== Done. Cross-dataset table/figure in runs/*/tables|figures; manuscript updated ==="
