

# Beyond Accuracy: A Comprehensive Benchmark of Uncertainty in Tabular Foundation Models

Recent Tabular Foundation Models (TFMs) have set new standards in predictive performance on tabular data, frequently outperforming Gradient-Boosted Decision Trees (GBDTs). However, their uncertainty quantification, a key aspect of model trustworthiness, has received far less attention.

This repository investigates that gap through an extensive comparison of TFMs, GBDTs, and classical baselines on the 112 datasets of the TALENT benchmark.

<p align="center">
<img width="600" alt="benchmarkinTFMs" src="https://github.com/user-attachments/assets/f3a931ff-d6c1-4b3c-bfd5-705e2cc6a8b0" />
</p>

### Model Performance Summary

Trade-off between normalized predictive performance (AUC) and normalized confidence (SSCS). The dashed curve indicates the trade-off: higher AUC and higher SSCS jointly indicate accurate and trustworthy models.

<p align="center">
<img width="600" alt="tradeoff-image" src="https://github.com/user-attachments/assets/065b74a9-7bc6-4885-a3ea-db92a9478d13" />
</p>

## Getting Started

### 1. Create an environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Download TALENT datasets
Place the TALENT dataset folders under `./data` (default) or point the CLI to a different location with `--dataset-path`.

### 3. Run the benchmark
```bash
python talent_benchmark.py \
  --datasets KDDCup09_upselling heloc \
  --models tabpfn lightgbm \
  --confidence-level 0.9 \
  --conformity-score lac top_k \
  --n-seeds 3
```

Results are written to `results_talent/<DATE>/talent_benchmark_seed_<SEED>_<DATE>.csv` and mirrored to Weights & Biases when enabled.

## Command-line Reference

Run `python talent_benchmark.py --help` for the full list. Highlights:

| Flag | Description |
| --- | --- |
| `--datasets / --models` | Lists of TALENT dataset names and model identifiers. |
| `--dataset-path` | Root folder containing preprocessed TALENT datasets. |
| `--dataset-summary` | CSV with metadata per dataset (`datasets_summary_talent.csv` by default). |
| `--confidence-level` | Target coverage for MAPIE prediction sets. |
| `--conformity-score` | One or more conformity scores (`lac`, `top_k`, `aps`, `raps`). Binary datasets automatically fall back to `lac`. |
| `--n-seeds` | Number of random seeds per dataset/model combination. |
| `--n-trials` | Optuna trials used during TALENT's HPO stage. Ignored in mock mode. |
| `--output-dir` | Directory used for CSV outputs and logs (`results_talent`). |
| `--mock-run` | Enables the faster smoke-test mode. |
| `--disable-wandb` | Skip Weights & Biases even if the project/entity flags are set. |
