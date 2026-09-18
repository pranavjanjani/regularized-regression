#!/bin/sh
# Copy the generated artefacts into the website folder so that `site/` is a
# self-contained, deployable static site (any static host, or file:// locally).
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p site/data site/assignments site/notebooks site/scripts
cp -f data/*.csv                site/data/            2>/dev/null || true
cp -f assignments/*.docx        site/assignments/     2>/dev/null || true
cp -f assignments/*.pdf         site/assignments/     2>/dev/null || true
cp -f notebooks/*.ipynb         site/notebooks/       2>/dev/null || true
cp -f notebooks/*.zip           site/notebooks/       2>/dev/null || true
cp -f scripts/make_data.py      site/scripts/         2>/dev/null || true
# Key each asset URL to its contents so a deploy cannot serve new HTML with
# stale CSS or JS from the browser cache.
python3 "$ROOT/scripts/stamp_assets.py"
echo "synced -> site/{data,assignments,notebooks,scripts}"
