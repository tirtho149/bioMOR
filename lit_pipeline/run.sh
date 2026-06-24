#!/usr/bin/env bash
# Gene-classification literature pipeline: harvest metadata -> download free PDFs.
# Usage:  ./run.sh [MAX]        MAX = optional cap for a quick test (e.g. ./run.sh 50)
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
MAX="${1:-0}"
YEAR="${YEAR:-2020-2026}"
QUERY="${QUERY:-gene classification | genomic sequence classification | gene expression classification}"

echo "=== Step 1: harvest metadata (Semantic Scholar bulk) ==="
$PY harvest.py --query "$QUERY" --year "$YEAR" --max "$MAX"

echo
echo "=== Step 2: download PDFs (all free APIs) ==="
$PY get_pdfs.py --max "$MAX"

echo
echo "Done. PDFs in ./pdfs/ , status in ./manifest.csv"
