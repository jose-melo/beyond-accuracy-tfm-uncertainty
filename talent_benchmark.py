from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import wandb

from utils import evaluate_talent_dataset, set_seed

LOGGER = logging.getLogger(__name__)

DEFAULT_DATASETS = [
    "BLE_RSSI_dataset_for_Indoor_localization",
    "Bank_Customer_Churn_Dataset",
    "Basketball_c",
    "Contaminant-detection-in-packaged-cocoa-hazelnut-spread-jars-using-Microwaves-Sensing-and-Machine-Learning-10.0GHz(Urbinati)",
    "Contaminant-detection-in-packaged-cocoa-hazelnut-spread-jars-using-Microwaves-Sensing-and-Machine-Learning-10.5GHz(Urbinati)",
    "Contaminant-detection-in-packaged-cocoa-hazelnut-spread-jars-using-Microwaves-Sensing-and-Machine-Learning-11.0GHz(Urbinati)",
    "Contaminant-detection-in-packaged-cocoa-hazelnut-spread-jars-using-Microwaves-Sensing-and-Machine-Learning-9.0GHz(Urbinati)",
    "Contaminant-detection-in-packaged-cocoa-hazelnut-spread-jars-using-Microwaves-Sensing-and-Machine-Learning-9.5GHz(Urbinati)",
    "Customer_Personality_Analysis",
    "Diabetic_Retinopathy_Debrecen",
    "Employee",
    "FICO-HELOC-cleaned",
    "FOREX_audcad-day-High",
    "FOREX_audchf-day-High",
    "FOREX_audjpy-day-High",
    "FOREX_cadjpy-day-High",
    "Fitness_Club_c",
    "GAMETES_Epistasis_2-Way_20atts_0.1H_EDM-1_1",
    "GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_50_EDM-2_001",
    "Gender_Gap_in_Spanish_WP",
    "GesturePhaseSegmentationProcessed",
    "Heart-Disease-Dataset-(Comprehensive)",
    "Is-this-a-good-customer",
    "JapaneseVowels",
    "KDD",
    "KDDCup09_upselling",
    "Long",
    "Marketing_Campaign",
    "Mobile_Price_Classification",
    "National_Health_and_Nutrition_Health_Survey",
    "Performance-Prediction",
    "PieChart3",
    "Pima_Indians_Diabetes_Database",
    "PizzaCutter3",
    "Pumpkin_Seeds",
    "QSAR_biodegradation",
    "Telecom_Churn_Dataset",
    "VulNoneVul",
    "Water_Quality_and_Potability",
    "Waterstress",
    "Wilt",
    "abalone",
    "ada",
    "ada_agnostic",
    "ada_prior",
    "airlines_seed_0_nrows_2000_nclasses_10_ncols_100_stratify_True",
    "allbp",
    "allrep",
    "analcatdata_authorship",
    "autoUniv-au7-1100",
    "banknote_authentication",
    "baseball",
    "car-evaluation",
    "churn",
    "cmc",
    "company_bankruptcy_prediction",
    "contraceptive_method_choice",
    "credit-g",
    "delta_ailerons",
    "dis",
    "drug_consumption",
    "estimation_of_obesity_levels",
    "eye_movements",
    "first-order-theorem-proving",
    "golf_play_dataset_extended",
    "heloc",
    "ibm-employee-performance",
    "kc1",
    "kdd_ipums_la_97-small",
    "kr-vs-kp",
    "maternal_health_risk",
    "mice_protein_expression",
    "national-longitudinal-survey-binary",
    "ozone-level-8hr",
    "ozone_level",
    "page-blocks",
    "pc1",
    "pc3",
    "pc4",
    "phoneme",
    "predict_students_dropout_and_academic_success",
    "rice_cammeo_and_osmancik",
    "ringnorm",
    "rl",
    "satimage",
    "segment",
    "seismic+bumps",
    "shill-bidding",
    "shrutime",
    "spambase",
    "splice",
    "sports_articles_for_objectivity_analysis",
    "statlog",
    "steel_plates_faults",
    "svmguide3",
    "sylvine",
    "taiwanese_bankruptcy_prediction",
    "telco-customer-churn",
    "thyroid",
    "thyroid-ann",
    "thyroid-dis",
    "turiye_student_evaluation",
    "twonorm",
    "vehicle",
    "wall-robot-navigation",
    "water_quality",
    "waveform-5000",
    "waveform_database_generator_version_1",
    "website_phishing",
    "wine",
    "wine-quality-red",
    "wine-quality-white",
]

DEFAULT_MODELS = [
    "tabpfn",
    "PFN-v2",
    "tabicl",
    "mitra",
    "xgboost",
    "lightgbm",
    "catboost",
    "tabm",
    "ftt",
    "tabr",
    "knn",
    "LogReg",
]


def _configure_logging(output_root: Path) -> logging.Logger:
    """Configure logger that streams to stdout and writes daily log files."""
    logger = logging.getLogger("talent_benchmark")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        log_dir = output_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_filename = f"talent_benchmark_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_dir / log_filename)
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger


def _append_to_csv(record: dict, csv_path: Path) -> None:
    """Append a single dictionary record to CSV, creating headers if needed."""
    df = pd.DataFrame([record])
    if csv_path.exists():
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)


def _select_device() -> torch.device:
    """Return the most capable device available on the current machine."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_dataset_summary(summary_path: Path) -> pd.DataFrame:
    """Load the dataset summary file if present, returning an empty frame otherwise."""
    if not summary_path.exists():
        LOGGER.warning(
            "Dataset summary file %s not found. Falling back to default conformity scores.",
            summary_path,
        )
        return pd.DataFrame()
    return pd.read_csv(summary_path)


def _build_task_lookup(summary_df: pd.DataFrame) -> Dict[str, str]:
    """Map dataset names to their declared task_type (binclass, multiclass, ...)."""
    if (
        summary_df.empty
        or "dataset_name" not in summary_df
        or "task_type" not in summary_df
    ):
        return {}
    return dict(zip(summary_df["dataset_name"], summary_df["task_type"]))


def _determine_conformity_scores(
    task_type: str | None,
    requested_scores: Sequence[str],
) -> List[str]:
    """Force binary problems to use `lac` while keeping user-configured defaults."""
    if task_type == "binclass":
        return ["lac"]
    return list(requested_scores)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TALENT Benchmark for UQ")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="List of dataset names from TALENT benchmark.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="List of models to evaluate.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="./data",
        help="Path to TALENT datasets.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.9,
        help="Target confidence level for prediction sets.",
    )
    parser.add_argument(
        "--conformity-score",
        "--conformity-scores",
        dest="conformity_scores",
        nargs="+",
        default=["lac", "top_k", "aps", "raps"],
        help="Conformity scores to evaluate for each dataset/model pair.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="talent-uq-benchmark",
        help="W&B project name. Leave empty to disable logging.",
    )
    parser.add_argument(
        "--wandb-entity", type=str, default=None, help="W&B entity name."
    )
    parser.add_argument(
        "--n-trials", type=int, default=25, help="Number of Optuna trials for HPO."
    )
    parser.add_argument(
        "--mock-run",
        action="store_true",
        help=(
            "Run a quick smoke-test over all datasets and models by sampling smaller "
            "subsets and skipping expensive steps like HPO."
        ),
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=1,
        help="Number of random seeds to evaluate for each dataset-model combination.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_talent",
        help="Directory used to store CSV outputs and logs.",
    )
    parser.add_argument(
        "--dataset-summary",
        type=str,
        default="./data/datasets_summary_talent.csv",
        help="CSV that stores metadata such as task type for each dataset.",
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Skip initializing Weights & Biases even if project/entity are provided.",
    )
    return parser.parse_args()


def run_benchmark(args: argparse.Namespace) -> None:
    device = _select_device()
    print(f"Using device: {device}")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    run_date = datetime.now().strftime("%Y%m%d")
    csv_path = (
        output_root / run_date / f"talent_benchmark_seed_{args.seed}_{run_date}.csv"
    )
    errors_csv_path = csv_path.parent / "talent_benchmark_errors.csv"

    logger = _configure_logging(output_root)
    logger.info("Benchmark results will be written to %s", csv_path)
    if args.mock_run:
        logger.info(
            "Quick-test mode enabled: sampling smaller datasets and disabling HPO."
        )

    summary_df = _load_dataset_summary(Path(args.dataset_summary))
    task_lookup = _build_task_lookup(summary_df)

    wandb_run = None
    if not args.disable_wandb and args.wandb_project:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
        )

    all_results: List[dict] = []
    all_errors: List[dict] = []

    def handle_result(result: dict) -> None:
        all_results.append(result)
        wandb_payload = result.get("wandb_log", {})
        csv_record = {k: v for k, v in result.items() if k != "wandb_log"}
        csv_record["timestamp"] = datetime.now().isoformat(timespec="seconds")
        csv_record["mock_run"] = args.mock_run
        _append_to_csv(csv_record, csv_path)
        if wandb_run and wandb_payload:
            wandb_run.log(wandb_payload)

    def handle_error(error_record: dict) -> None:
        all_errors.append(error_record)
        error_entry = {**error_record}
        error_entry["timestamp"] = datetime.now().isoformat(timespec="seconds")
        error_entry["mock_run"] = args.mock_run
        _append_to_csv(error_entry, errors_csv_path)

    rng = np.random.default_rng(args.seed)
    set_seed(args.seed)

    for dataset_name in tqdm(args.datasets, desc="Datasets"):
        logger.info("--- Evaluating dataset: %s ---", dataset_name)
        dataset_seeds = rng.integers(
            low=0, high=np.iinfo(np.int32).max, size=args.n_seeds, dtype=np.int64
        )

        task_type = task_lookup.get(dataset_name)
        if task_type is None:
            logger.warning(
                "Dataset %s missing from %s. Using default conformity scores.",
                dataset_name,
                args.dataset_summary,
            )
        conformity_scores = _determine_conformity_scores(
            task_type, args.conformity_scores
        )

        for run_index, seed_value in enumerate(dataset_seeds, start=1):
            seed_int = int(seed_value)
            for score_index, conformity_score in enumerate(conformity_scores, start=1):
                logger.info(
                    "Running seed %d (%d/%d) with conformity score %s (%d/%d) for dataset %s",
                    seed_int,
                    run_index,
                    args.n_seeds,
                    conformity_score,
                    score_index,
                    len(conformity_scores),
                    dataset_name,
                )
                set_seed(seed_int)
                try:
                    results, errors = evaluate_talent_dataset(
                        dataset_name=dataset_name,
                        dataset_path=args.dataset_path,
                        models_to_run=args.models,
                        device=device,
                        confidence_level=args.confidence_level,
                        conformity_score=conformity_score,
                        seed=seed_int,
                        n_trials=args.n_trials,
                        logger=logger,
                        on_result=handle_result,
                        on_error=handle_error,
                        mock_run=args.mock_run,
                    )
                    logger.info(
                        "Completed dataset %s seed %d conformity score %s with %d successes and %d errors",
                        dataset_name,
                        seed_int,
                        conformity_score,
                        len(results),
                        len(errors),
                    )
                except Exception as dataset_error:
                    logger.exception(
                        "Dataset %s seed %d conformity score %s failed with an unrecoverable error: %s",
                        dataset_name,
                        seed_int,
                        conformity_score,
                        dataset_error,
                    )
                    handle_error(
                        {
                            "dataset": dataset_name,
                            "model": None,
                            "seed": seed_int,
                            "conformity_score": conformity_score,
                            "error": repr(dataset_error),
                        }
                    )
                    continue

    logger.info(
        "Benchmark finished with %d successful evaluations and %d errors. "
        "Results saved to %s",
        len(all_results),
        len(all_errors),
        csv_path,
    )

    if wandb_run:
        wandb_run.finish()


def main() -> None:
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
