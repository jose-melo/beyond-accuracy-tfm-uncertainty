from __future__ import annotations

import logging
import os
import random
import warnings
from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from mapie.classification import SplitConformalClassifier
from mapie.metrics.calibration import expected_calibration_error, top_label_ece
from mapie.metrics.classification import (
    classification_coverage_score,
    classification_mean_width_score,
    classification_ssc_score,
)
from mapie.utils import train_conformalize_test_split
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import Bunch
from skrub import TableVectorizer
from tabicl import TabICLClassifier
from tabpfn import TabPFNClassifier
from tqdm import tqdm
from xgboost import XGBClassifier

from model.lib.data import get_dataset
from model.utils import get_method, tune_hyper_parameters

try:
    from mapie.estimator.classifier import EnsembleClassifier
except Exception:  # noqa: BLE001
    EnsembleClassifier = None
else:
    if EnsembleClassifier and not hasattr(EnsembleClassifier, "__sklearn_tags__"):

        def _mapie_sklearn_tags(self):
            return BaseEstimator.__sklearn_tags__(self)

        EnsembleClassifier.__sklearn_tags__ = _mapie_sklearn_tags


try:
    from pandas.errors import SettingWithCopyWarning
except Exception:
    from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter("ignore", SettingWithCopyWarning)
warnings.simplefilter("ignore", RuntimeWarning)
warnings.simplefilter("ignore", UserWarning)

LOGGER = logging.getLogger(__name__)


@dataclass
class TalentDatasetSplits:
    """Container describing the arrays used for TALENT evaluation."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_calib: np.ndarray
    y_calib: np.ndarray
    X_test: np.ndarray
    y_test_raw: np.ndarray
    n_categorical_features: int
    info: Dict[str, Any]


def set_seed(seed: int = 42):
    """
    Sets the random seed for various libraries to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"Random seed set to {seed}")


class TalentScikitWrapper(BaseEstimator, ClassifierMixin):
    """
    A wrapper to make TALENT models compatible with the scikit-learn API.
    """

    def __init__(self, talent_method, info, n_cat_features=0):
        self.talent_method = talent_method
        self.info = info
        self.n_cat_features = n_cat_features
        self._is_fitted = False
        self.config = talent_method.args.config
        self._estimator_type = "classifier"

    def _split_features(self, X):
        """Splits the input X into numerical and categorical parts."""
        if self.n_cat_features == 0:
            N = X
            C = None
        elif self.n_cat_features == X.shape[1]:
            N = None
            C = X
        else:
            C = X[:, : self.n_cat_features]
            N = X[:, self.n_cat_features :]
        return N, C

    def fit(self, X, y):
        """
        Fits the underlying TALENT model.
        X and y are expected to be the full training data (train + validation for HPO).
        """
        N, C = self._split_features(X)

        N_dict = {"train": N, "val": N} if N is not None else None
        C_dict = {"train": C, "val": C} if C is not None else None
        y_dict = {"train": y, "val": y}

        fit_data = (N_dict, C_dict, y_dict)

        self.talent_method.fit(fit_data, self.info, train=True, config=self.config)
        self._is_fitted = True
        self.classes_ = self.talent_method.label_encoder.classes_
        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities for X.
        """
        if not self._is_fitted:
            raise RuntimeError("This model is not fitted yet.")

        N, C = self._split_features(X)
        N_dict = {"test": N} if N is not None else None
        C_dict = {"test": C} if C is not None else None
        y_dict = {"test": np.zeros((X.shape[0],), dtype=int)}

        logits = self.talent_method.predict(
            (N_dict, C_dict, y_dict), self.info, model_name="best-val"
        )

        if isinstance(logits, torch.Tensor):
            logits = np.array(logits.cpu())

        if np.any((logits < 0) | (logits > 1)) or (
            not np.allclose(logits.sum(axis=-1), 1, atol=1e-5)
        ):
            exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            logits = exps / np.sum(exps, axis=1, keepdims=True)

        return logits

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def concat_features(
    C_part: Optional[np.ndarray], N_part: Optional[np.ndarray]
) -> np.ndarray:
    """Concatenate categorical and numerical blocks keeping `[C | N]` ordering."""
    parts = [p for p in (C_part, N_part) if p is not None]
    if not parts:
        raise ValueError("No features found (need at least C_* or N_*).")
    return np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]


def evaluate_classification(y_pred, y_test, y_pred_set, y_prob):
    """Evaluate classification models (binary & multiclass)."""
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    if y_prob.shape[1] == 2:
        auc = roc_auc_score(y_test, y_prob[:, 1])
    else:
        auc = roc_auc_score(y_test, y_prob, multi_class="ovo", average="weighted")

    cr = classification_coverage_score(y_test, y_pred_set)
    mwc = classification_mean_width_score(y_pred_set)
    sscs = classification_ssc_score(y_test, y_pred_set)
    if y_prob.shape[1] == 2:
        ece = expected_calibration_error(y_test, y_prob)
    else:
        ece = top_label_ece(y_test, y_prob)

    return {
        "accuracy": acc,
        "f1_score": f1,
        "auc": auc,
        "coverage_rate": cr[0].item(),
        "mean_width": mwc[0].item(),
        "ssc_score": sscs[0].item(),
        "ece": ece,
    }


def clean_col(col):
    return (
        col.replace("<=", "_le_")
        .replace("<", "_lt_")
        .replace("[", "_")
        .replace("]", "_")
    )


def _maybe_sample_subset(
    X: np.ndarray,
    y: np.ndarray,
    limit: Optional[int],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a random subset of (X, y) if the pair exceeds the requested limit."""
    if limit is None or X.shape[0] <= limit or y.shape[0] <= limit:
        return X, y

    indices = rng.choice(X.shape[0], size=limit, replace=False)
    return X[indices], y[indices]


def _split_raw_features(
    X: np.ndarray,
    n_cat_features: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Split concatenated `[C | N]` arrays back into categorical/numerical matrices."""
    if n_cat_features <= 0:
        return None, X
    if n_cat_features >= X.shape[1]:
        return X, None
    return X[:, :n_cat_features], X[:, n_cat_features:]


def _prepare_talent_dataset(
    dataset_name: str,
    dataset_path: str,
    seed: int,
    n_trials: int,
    mock_run: bool,
) -> TalentDatasetSplits:
    """Load a TALENT dataset and return the arrays used for model evaluation."""
    dataset_args, _ = get_talent_args(
        dataset_name, dataset_path, "tabpfn", seed, n_trials
    )
    train_val_data, test_data, info = get_dataset(
        dataset_args.config["dataset"], dataset_args.config["dataset_path"]
    )
    (N_trainval, C_trainval, y_trainval) = train_val_data
    (N_test, C_test, y_test) = test_data

    N_pool = (
        np.concatenate([N_trainval["train"], N_trainval["val"]]) if N_trainval else None
    )
    C_pool = (
        np.concatenate([C_trainval["train"], C_trainval["val"]]) if C_trainval else None
    )
    y_pool = np.concatenate([y_trainval["train"], y_trainval["val"]])

    X_pool = concat_features(C_pool, N_pool)
    X_test = concat_features(
        C_test["test"] if C_test else None,
        N_test["test"] if N_test else None,
    )
    y_test_raw = y_test["test"]

    rng = np.random.default_rng(seed)
    if mock_run:
        X_pool, y_pool = _maybe_sample_subset(
            X_pool, y_pool, min(500, X_pool.shape[0]), rng
        )
        X_test, y_test_raw = _maybe_sample_subset(
            X_test, y_test_raw, min(200, X_test.shape[0]), rng
        )

    X_train, X_calib, y_train, y_calib = train_test_split(
        X_pool, y_pool, test_size=0.3, random_state=seed, stratify=y_pool
    )

    n_cat_features = C_trainval["train"].shape[1] if C_trainval else 0
    return TalentDatasetSplits(
        X_train=X_train,
        y_train=y_train,
        X_calib=X_calib,
        y_calib=y_calib,
        X_test=X_test,
        y_test_raw=y_test_raw,
        n_categorical_features=n_cat_features,
        info=info,
    )


def _extract_best_params(
    config: Dict[str, Any], search_space: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Return the tuned hyperparameters tracked from TALENT's search space."""
    best_params: Dict[str, Any] = {}
    for group_key, params_in_group in search_space.items():
        group_cfg = config.get(group_key)
        if not isinstance(group_cfg, dict):
            continue
        for param_name in params_in_group:
            if param_name in group_cfg:
                best_params[f"{group_key}_{param_name}"] = group_cfg[param_name]
    return best_params


def _merge_missing_config_values(target: Namespace, fallback: Namespace) -> None:
    """Fill empty entries in the tuned config with fallback defaults."""

    def _is_empty(value: Any) -> bool:
        return value in (None, {}) or (hasattr(value, "__len__") and len(value) == 0)  # type: ignore[arg-type]

    source_dict = vars(fallback)
    for key in list(target.config.keys()):
        if key in source_dict and _is_empty(target.config[key]):
            target.config[key] = source_dict[key]


def _run_talent_hpo(
    model_name: str,
    current_model_args: Namespace,
    opt_space_model: Dict[str, Any],
    info: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_cat_features: int,
    seed: int,
) -> Tuple[Namespace, Dict[str, Any]]:
    """Perform TALENT's Optuna-powered HPO for a specific model."""
    X_hpo_train, X_hpo_val, y_hpo_train, y_hpo_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=seed, stratify=y_train
    )
    C_hpo_train, N_hpo_train = _split_raw_features(X_hpo_train, n_cat_features)
    C_hpo_val, N_hpo_val = _split_raw_features(X_hpo_val, n_cat_features)

    hpo_data = (
        {"train": N_hpo_train, "val": N_hpo_val},
        {"train": C_hpo_train, "val": C_hpo_val},
        {"train": y_hpo_train, "val": y_hpo_val},
    )

    tuned_args = tune_hyper_parameters(
        current_model_args, opt_space_model, hpo_data, info
    )
    best_params = _extract_best_params(
        tuned_args.config, opt_space_model.get(model_name, {})
    )
    return tuned_args, best_params


def _prepare_model_arguments(
    dataset_name: str,
    dataset_path: str,
    model_name: str,
    info: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_cat_features: int,
    seed: int,
    n_trials: int,
    mock_run: bool,
) -> Tuple[Namespace, Dict[str, Any], bool, bool]:
    """Return the Namespace passed to TALENT along with metadata about HPO."""
    current_model_args, opt_space_model = get_talent_args(
        dataset_name, dataset_path, model_name, seed, n_trials
    )
    opt_space_model = opt_space_model or {}
    has_search_space = model_name in opt_space_model
    ran_hpo = False

    if has_search_space and not mock_run:
        tuned_args, best_params = _run_talent_hpo(
            model_name,
            current_model_args,
            opt_space_model,
            info,
            X_train,
            y_train,
            n_cat_features,
            seed,
        )
        ran_hpo = True
    else:
        tuned_args = current_model_args
        best_params = {}

    _merge_missing_config_values(tuned_args, current_model_args)
    return tuned_args, best_params, ran_hpo, has_search_space


def tune_model_with_optuna(
    model_class,
    trial_params: Dict[str, Any],
    X_train,
    y_train,
    X_val,
    y_val,
    n_trials=25,
):
    """
    Tunes a model's hyperparameters using Optuna.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            name: getattr(trial, f"suggest_{sugg_type}")(name, **kwargs)
            for name, (sugg_type, kwargs) in trial_params.items()
        }
        model = model_class(**params)

        if "verbose" in model.get_params():
            model.set_params(verbose=-1)
        if "verbosity" in model.get_params():
            model.set_params(verbosity=0)

        model.fit(X_train, y_train)
        y_pred_val = model.predict(X_val)
        score = accuracy_score(y_val, y_pred_val)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params


def get_talent_args(dataset_name, dataset_path, model_type, seed, n_trials):
    """Creates a mock args object for TALENT functions."""
    args = Bunch()
    args.dataset = dataset_name
    args.dataset_path = dataset_path
    args.model_type = model_type
    args.seed = seed
    args.save_path = "./"
    args.use_float = True
    args.n_trials = n_trials

    import importlib.resources as pkg_resources
    import json

    default_para_model = {}
    opt_space_model = {}
    model_config = {}

    config_found = False
    for config_type in ["deep", "classical"]:
        try:
            default_path = f"configs/default/{model_type}.json"
            opt_space_path = f"configs/opt_space/{model_type}.json"

            with open(default_path, "r") as f:
                default_para_model = json.load(f)
            with open(opt_space_path, "r") as f:
                opt_space_model = json.load(f)

            model_config = default_para_model[model_type]
            config_found = True
            if config_type == "classical":
                classical_path = "configs/classical_configs.json"

                with classical_path.open("r") as f:
                    classical_configs = json.load(f)
                classical_configs.update(model_config)
                model_config = classical_configs
            elif config_type == "deep":
                deep_path = "configs/deep_configs.json"
                with open(deep_path, "r") as f:
                    deep_configs = json.load(f)
                deep_configs.update(model_config)
                model_config = deep_configs
            break
        except FileNotFoundError:
            continue

    if not config_found:
        print(
            f"Warning: Config files for model '{model_type}' not found in TALENT's default/opt_space. HPO might be skipped."
        )

    model_config.update(args)

    namespace_config = Namespace(**{"config": model_config, **args, **model_config})
    return namespace_config, opt_space_model


def evaluate_talent_dataset(
    dataset_name,
    dataset_path,
    models_to_run,
    device,
    confidence_level=0.9,
    conformity_score="lac",
    seed=42,
    n_trials=25,
    logger: Optional[logging.Logger] = None,
    on_result: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_error: Optional[Callable[[Dict[str, Any]], None]] = None,
    mock_run: bool = False,
):
    """Evaluates models on a single TALENT dataset."""
    dataset_splits = _prepare_talent_dataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        seed=seed,
        n_trials=n_trials,
        mock_run=mock_run,
    )
    log = logger or LOGGER

    all_results = []
    errors = []
    for model_name in models_to_run:
        log.info("Training %s on %s", model_name, dataset_name)
        try:
            tuned_args, best_params, ran_hpo, has_search_space = (
                _prepare_model_arguments(
                    dataset_name=dataset_name,
                    dataset_path=dataset_path,
                    model_name=model_name,
                    info=dataset_splits.info,
                    X_train=dataset_splits.X_train,
                    y_train=dataset_splits.y_train,
                    n_cat_features=dataset_splits.n_categorical_features,
                    seed=seed,
                    n_trials=n_trials,
                    mock_run=mock_run,
                )
            )

            if ran_hpo:
                log.info("Completed TALENT HPO for %s", model_name)
            elif not has_search_space:
                log.info("Skipping HPO for %s (no search space found).", model_name)
            elif mock_run:
                log.info("Skipping HPO for %s (mock run).", model_name)

            is_regression = dataset_splits.info["task_type"] == "regression"
            talent_method = get_method(tuned_args.model_type)(tuned_args, is_regression)

            model = TalentScikitWrapper(
                talent_method,
                dataset_splits.info,
                dataset_splits.n_categorical_features,
            )

            log.info(
                "Fitting %s on the full training data (train + HPO validation)",
                model_name,
            )
            model.fit(dataset_splits.X_train, dataset_splits.y_train)
            y_test = talent_method.label_encoder.transform(dataset_splits.y_test_raw)

            log.info("Conformalizing %s on the calibration data", model_name)
            mapie_clf = SplitConformalClassifier(
                estimator=model,
                confidence_level=confidence_level,
                prefit=True,
                conformity_score=conformity_score,
                random_state=seed,
            )
            mapie_clf.conformalize(dataset_splits.X_calib, dataset_splits.y_calib)

            y_prob = model.predict_proba(dataset_splits.X_test)
            y_pred = np.argmax(y_prob, axis=1)
            _, y_pred_set = mapie_clf.predict_set(dataset_splits.X_test)

            test_metrics = evaluate_classification(y_pred, y_test, y_pred_set, y_prob)

            prefixed_metrics = {f"{model_name}_{k}": v for k, v in test_metrics.items()}
            prefixed_params = {
                f"{model_name}_best_{k}": v for k, v in best_params.items()
            }

            result = {
                "dataset": dataset_name,
                "seed": seed,
                "model": model_name,
                "confidence_level": confidence_level,
                "conformity_score": conformity_score,
                **test_metrics,
                "wandb_log": {
                    **prefixed_metrics,
                    **prefixed_params,
                    "dataset": dataset_name,
                    "seed": seed,
                    "conformity_score": conformity_score,
                },
            }
            all_results.append(result)
            if on_result:
                on_result(result)
            log.info("Successfully evaluated %s on %s", model_name, dataset_name)
        except Exception as exc:  # noqa: BLE001
            error_record = {
                "dataset": dataset_name,
                "seed": seed,
                "model": model_name,
                "conformity_score": conformity_score,
                "error": repr(exc),
            }
            errors.append(error_record)
            if on_error:
                on_error(error_record)
            log.exception(
                "Error while evaluating %s on %s. Continuing with next model.",
                model_name,
                dataset_name,
            )
            continue

    return all_results, errors
