#!/usr/bin/env bash
set -euo pipefail

REPO="${TITAN_REPO:-./TITAN}"
TEST="${TEST:-datasets/trained_model_test.csv}"
TCRS="${TCRS:-datasets/tcr_full.csv}"
EPITOPES="${EPITOPES:-datasets/epitopes.smi}"
MODEL="${MODEL:-trained_model}"
SAVE="${SAVE:-repro}"

cd "${REPO}"
mkdir -p "${MODEL}/results"

python scripts/flexible_model_eval.py \
  "${TEST}" \
  "${TCRS}" \
  "${EPITOPES}" \
  "${MODEL}" \
  bimodal_mca \
  "${SAVE}"

ls -la "${MODEL}/results"
