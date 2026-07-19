#!/usr/bin/env bash
# Full pipeline: Phase-1A (CDKA) + Contribution 2 (UAAS) + Contribution 3
# (explanation verification) + manuscript. GPU machine.
# Usage:  bash scripts/run_all.sh [CONFIG] [DATASET]
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"
DATASET="${2:-20newsgroups}"
PY="python -m"

bash scripts/run_phase1a.sh "$CONFIG" "$DATASET"

echo "=== Contribution 2: UAAS (adaptive LoRA rank) ==="
$PY smart_llm.experiments.train_uaas --config "$CONFIG" --dataset "$DATASET" --train-samples 1000

echo "=== Contribution 3: attribution-guided explanation verification ==="
$PY smart_llm.experiments.explain_verify --config "$CONFIG" --dataset "$DATASET"

echo "=== Rebuild manuscript + supplementary with UAAS/explanation tables ==="
$PY smart_llm.paper.make_manuscript    --config "$CONFIG" --dataset "$DATASET"
$PY smart_llm.paper.make_supplementary --config "$CONFIG" --dataset "$DATASET"

echo "=== Full pipeline complete ==="
