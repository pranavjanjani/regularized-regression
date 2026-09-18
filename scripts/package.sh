#!/bin/sh
# Zip the notebooks + data + generator into one archive for the Notebooks page.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
rm -rf build/pkg && mkdir -p build/pkg/regression_course/data build/pkg/regression_course/scripts
cp notebooks/*.ipynb build/pkg/regression_course/
cp data/*.csv       build/pkg/regression_course/data/
cp scripts/make_data.py build/pkg/regression_course/scripts/
cat > build/pkg/regression_course/README.txt << 'TXT'
Regularized Regression - From Likelihood to Lasso
Course notebooks and datasets

  *_starter.ipynb    scaffolding only; the estimators are yours to write
  *_solution.ipynb   complete and already executed, with figures

  data/uber_surge_mumbai.csv   200 trips, the main dataset
  data/new_city_features.csv   60 trips x 120 features, for Task 3b
  data/new_city_truth.csv      ground truth - do not open until Task 3b step 3
  scripts/make_data.py         regenerates both datasets from seed 42

Requires numpy, pandas, matplotlib and scikit-learn. Runs in Colab unchanged:
upload the notebook and the data/ folder, then Run all.

Work through them in order: 01 -> 02 -> 03 -> 04.
TXT
cd build/pkg && zip -qr "$ROOT/notebooks/regression_course_notebooks.zip" regression_course
cd "$ROOT" && rm -rf build/pkg
echo "packaged: $(du -h notebooks/regression_course_notebooks.zip | cut -f1)"
