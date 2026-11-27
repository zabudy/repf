#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python312="${REPF_PYTHON:-python3.12}"
"$python312" -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
if [[ ! -f .env ]]; then
  cp .env.example .env
fi
if [[ ! -s data/processed/cases.json || ! -s data/processed/glossary.json ]]; then
  .venv/bin/python -m scripts.generate_knowledge_base
fi
.venv/bin/python -m scripts.import_data
exec .venv/bin/python run.py
