#!/usr/bin/env bash
set -euo pipefail

REPO="${EPITCR_REPO:-./epiTCR}"
TEST="${TEST:-data/splitData/withoutMHC/test/test01.csv}"
MODEL="${MODEL:-$(cd "$(dirname "$0")" && pwd)/models_unzipped/rdforestWithoutMHCModel.pickle}"
OUT="${OUT:-$(cd "$(dirname "$0")" && pwd)/epitcr_predictions.csv}"

cd "${REPO}"

python predict.py \
  --testfile "${TEST}" \
  --modelfile "${MODEL}" \
  --chain ce \
  --outfile "${OUT}"

ls -la "${OUT}"
