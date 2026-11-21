#!/usr/bin/env bash
set -euo pipefail

PY=""
for c in python3.11 python3.10 python3.9; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done

if [[ -z "$PY" ]]; then
  echo "need python 3.9, 3.10 or 3.11. brew install python@3.11"
  exit 1
fi

"$PY" -m venv epitcr_env
./epitcr_env/bin/pip install --upgrade pip
./epitcr_env/bin/pip install -r requirements_mac.txt

REPO="${EPITCR_REPO:-./epiTCR}"
if [[ -d "$REPO" ]]; then
  mkdir -p models_unzipped
  unzip -o -q "$REPO/models/rdforestWithoutMHCModel.pickle.zip" -d models_unzipped
fi

echo "source epitcr_env/bin/activate"
