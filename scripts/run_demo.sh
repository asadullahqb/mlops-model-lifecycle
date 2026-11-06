#!/usr/bin/env bash
# One-command local demo of the closed loop.
# Requires: python 3.10+, pip. (Runs the components directly; no Docker needed.)
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> 1/5 install dependencies"
pip install -q -r requirements.txt

echo "==> 2/5 reset dataset + train baseline model"
python -m src.data reset
python -m src.train

echo "==> 3/5 start serve (background) and hit /predict"
python -m uvicorn src.serve:app --host 127.0.0.1 --port 8000 >/tmp/mlops-serve.log 2>&1 &
SERVE_PID=$!
trap 'kill $SERVE_PID 2>/dev/null || true' EXIT

for i in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then break; fi
  sleep 1
done

echo "  GET /health -> $(curl -fsS http://127.0.0.1:8000/health)"
echo "  POST /predict -> $(curl -fsS -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{"features":[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]}')"

echo "==> 4/5 inject drift + run the closed loop"
python -m src.orchestrate --inject-drift

echo "==> 5/5 confirm serve now uses the promoted model"
echo "  POST /predict -> $(curl -fsS -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{"features":[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]}')"

echo ""
echo "Done. See runs/ for the audit trail and models/ for the versioned registry."
