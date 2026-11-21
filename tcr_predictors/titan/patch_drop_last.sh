#!/usr/bin/env bash
set -euo pipefail
sed -i.bak 's/drop_last=True,/drop_last=False,/g' "${TITAN_REPO:-./TITAN}/scripts/flexible_model_eval.py"
grep -n "drop_last" "${TITAN_REPO:-./TITAN}/scripts/flexible_model_eval.py"
