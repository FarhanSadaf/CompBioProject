#!/usr/bin/env bash
set -euo pipefail

ARCH="$(uname -m)"
ENV_NAME="titan"

if ! command -v git >/dev/null 2>&1; then
  echo "git required"
  exit 1
fi

if [[ "${ARCH}" == "arm64" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda required. brew install --cask miniforge && conda init zsh"
    exit 1
  fi
  CONDA_SUBDIR=osx-64 conda create -y -n "${ENV_NAME}" python=3.9
  conda run -n "${ENV_NAME}" conda config --env --set subdir osx-64
  conda run -n "${ENV_NAME}" pip install -r requirements_mac.txt
  echo "conda activate ${ENV_NAME}"
else
  if ! command -v python3.9 >/dev/null 2>&1; then
    echo "python3.9 required. brew install python@3.9"
    exit 1
  fi
  python3.9 -m venv titan_env
  ./titan_env/bin/pip install --upgrade pip
  ./titan_env/bin/pip install -r requirements_mac.txt
  echo "source titan_env/bin/activate"
fi
