# Beyond Accuracy: A Comprehensive Benchmark of Uncertainty in Tabular Foundation Models

Recent Tabular Foundation Models (TFMs) have demonstrated state-of-the-art predictive performance, often surpassing Gradient-Boosted Decision Trees (GBDTs). However, the trustworthiness of these models, particularly their uncertainty quantification, has been largely overlooked. We investigate this gap through an extensive study comparing TFMs, GBDTs, and classical baselines on the 112 datasets of the TALENT benchmark. Our findings expose a critical trade-off: while TFMs consistently lead in predictive accuracy (AUC score), they exhibit significantly poorer stability under prediction confidence compared to GBDTs. Complementary experiments on synthetic datasets further characterize the regimes in which this effect intensifies. We conclude that while TFMs advance predictive frontiers, achieving well-calibrated uncertainty remains a major open challenge for the current generation of these models. Code is available at: \todo{add code link

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

## Repository Structure
```
configs/                 # TALENT model defaults and hyper-parameter spaces
model/                   # TALENT model implementations and utilities
talent_benchmark.py      # Entry point to execute the benchmark
utils.py                 # Shared helpers (dataset prep, Optuna tuning, evaluation)
requirements.txt         # Python dependencies
```