#!/usr/bin/env python3
"""
Generate synthetic datasets.

Based on analysis of real datasets, we target:
- Low SNR (0.02 - 0.10): Weak signal, but not degenerate
- Low Fisher ratio (≈0.5 - 2.5): Poor linear separability
- Mildly non-Gaussian distributions: skewed/heavy-tailed but bounded
- Moderate correlation (0.25 - 0.45)
- Intrinsic dimensionality ~0.5 (redundant features)
- Medium size: 1,000 - 5,000 samples
- Moderate features: 15 - 40
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
from scipy import stats
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler


class SyntheticGenerator:
    def __init__(
        self,
        n_samples: int = 2400,
        n_informative_features: int = 12,
        n_redundant_features: int = 10,
        n_noise_features: int = 8,
        target_snr: float = 0.10,
        target_fisher_ratio: float = 2.0,
        class_sep: float = 0.5,
        n_classes: int = 2,
        random_state: int = 42,
        is_baseline: bool = False,
    ):
        """
        Initialize generator.

        Args:
            n_samples: Number of samples to generate
            n_informative_features: Number of truly informative features
            n_redundant_features: Number of redundant (correlated) features
            n_noise_features: Number of pure noise features
            target_snr: Target signal-to-noise ratio (≈0.02 - 0.10)
            target_fisher_ratio: Target Fisher discriminant ratio (≈0.5 - 2.5)
            class_sep: Class separation parameter (lower = harder)
            n_classes: Number of classes (2 for binary)
            random_state: Random seed
            is_baseline: If True, generate baseline dataset (Gaussian, low noise)
        """
        self.n_samples = n_samples
        self.n_informative = n_informative_features
        self.n_redundant = n_redundant_features
        self.n_noise = n_noise_features
        self.n_features = (
            n_informative_features + n_redundant_features + n_noise_features
        )
        self.target_snr = target_snr
        self.target_fisher_ratio = target_fisher_ratio
        self.class_sep = class_sep
        self.n_classes = n_classes
        self.random_state = random_state
        self.is_baseline = is_baseline
        self.rng = np.random.RandomState(random_state)

    def _make_base_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate base classification data."""

        if self.is_baseline:
            X, y = make_classification(
                n_samples=self.n_samples,
                n_features=self.n_informative + self.n_redundant,
                n_informative=self.n_informative,
                n_redundant=self.n_redundant,
                n_classes=self.n_classes,
                n_clusters_per_class=1,
                class_sep=self.class_sep,
                flip_y=0.0,
                random_state=self.random_state,
            )
        else:
            X, y = make_classification(
                n_samples=self.n_samples,
                n_features=self.n_informative + self.n_redundant,
                n_informative=self.n_informative,
                n_redundant=self.n_redundant,
                n_classes=self.n_classes,
                n_clusters_per_class=2,
                class_sep=self.class_sep,
                flip_y=0.1,
                random_state=self.random_state,
            )

        return X, y

    def _add_heavy_noise(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        current_snr = self._compute_snr(X, y)

        if current_snr <= self.target_snr:
            return X

        overall_mean = np.mean(X, axis=0)
        signal_var = np.var(X - overall_mean, axis=0)

        required_noise_var = signal_var / (self.target_snr + 1e-10)

        within_class_var = np.zeros(X.shape[1])
        for cls in np.unique(y):
            X_cls = X[y == cls]
            class_mean = np.mean(X_cls, axis=0)
            within_class_var += np.var(X_cls - class_mean, axis=0)
        within_class_var /= len(np.unique(y))

        additional_noise_var = np.maximum(0, required_noise_var - within_class_var)
        per_feature_var = np.var(X, axis=0)
        noise_cap = 0.8 * per_feature_var
        additional_noise_var = np.minimum(additional_noise_var, noise_cap)

        X_noisy = X.copy()
        if self.is_baseline:
            noise_scale = np.sqrt(additional_noise_var + 1e-12)
            noise = self.rng.randn(X.shape[0], X.shape[1]) * noise_scale
            X_noisy += noise
        else:
            for cls in np.unique(y):
                mask = y == cls
                noise_scale = np.sqrt(additional_noise_var + 1e-12)
                noise = self.rng.exponential(
                    scale=noise_scale, size=(mask.sum(), X.shape[1])
                )
                noise -= np.mean(noise, axis=0)
                X_noisy[mask] += noise

        min_allowed_snr = max(0.03, 0.7 * self.target_snr)
        for _ in range(2):
            final_snr = self._compute_snr(X_noisy, y)
            if final_snr >= min_allowed_snr:
                break
            blend = min(0.8, 1 - (final_snr / (min_allowed_snr + 1e-12)))
            X_noisy = (1 - blend) * X_noisy + blend * X

        return X_noisy

    def _apply_non_gaussian_transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features to be non-Gaussian (skewed, heavy-tailed)."""
        if self.is_baseline:
            return X

        X_transformed = X.copy()

        transform_types = ["identity", "exp", "log", "power", "tanh"]
        transform_probs = [0.4, 0.15, 0.15, 0.15, 0.15]

        for i in range(X.shape[1]):
            feature = X[:, i]

            feature = (feature - np.mean(feature)) / (np.std(feature) + 1e-10)

            transform_type = self.rng.choice(transform_types, p=transform_probs)

            if transform_type == "exp":
                feature = np.exp(0.2 * feature)
            elif transform_type == "log":
                feature = feature - np.min(feature) + 1.01
                feature = np.log(feature)
            elif transform_type == "power":
                power = self.rng.uniform(1.2, 1.8)
                feature = np.sign(feature) * (np.abs(feature) ** power)
            elif transform_type == "tanh":
                feature = np.tanh(feature)

            feature = (feature - np.mean(feature)) / (np.std(feature) + 1e-10)
            X_transformed[:, i] = feature

        return X_transformed

    def _add_correlated_features(self, X: np.ndarray) -> np.ndarray:
        """Add redundant features through correlation."""

        X_with_redundant = []

        for i in range(self.n_informative):
            X_with_redundant.append(X[:, i])

        for i in range(self.n_redundant):
            if self.rng.rand() < 0.5:
                source_idx = self.rng.choice(len(X_with_redundant))
                base_feature = X_with_redundant[source_idx]
                noise = 0.1 * np.std(base_feature) * self.rng.randn(X.shape[0])
                redundant_feature = base_feature + noise
            else:
                n_to_combine = self.rng.choice([2, 3])
                pool_size = len(X_with_redundant)
                features_to_combine = self.rng.choice(
                    pool_size, n_to_combine, replace=False
                )

                weights = self.rng.randn(n_to_combine)
                weights /= np.sum(np.abs(weights)) + 1e-10

                redundant_feature = np.zeros(X.shape[0])
                for j, feat_idx in enumerate(features_to_combine):
                    redundant_feature += weights[j] * X_with_redundant[feat_idx]

                noise_level = self.rng.uniform(0.05, 0.15)
                noise = self.rng.randn(X.shape[0]) * np.std(redundant_feature)
                redundant_feature = (
                    1 - noise_level
                ) * redundant_feature + noise_level * noise

            X_with_redundant.append(redundant_feature)

        X_combined = np.column_stack(X_with_redundant)
        return X_combined

    def _add_pure_noise_features(self, X: np.ndarray) -> np.ndarray:
        """Add pure noise features (mix of structured and white noise)."""

        noise_features = []
        shared_latent = self.rng.randn(X.shape[0])
        for i in range(self.n_noise):
            if self.rng.rand() < 0.85:
                base_idxs = self.rng.choice(
                    X.shape[1], self.rng.choice([2, 3, 4]), replace=False
                )
                weights = self.rng.randn(len(base_idxs))
                weights /= np.sum(np.abs(weights)) + 1e-10
                structured = np.zeros(X.shape[0])
                for w, idx in zip(weights, base_idxs):
                    structured += w * X[:, idx]
                structured += 0.15 * shared_latent
                structured += 0.15 * self.rng.randn(X.shape[0])
                noise_features.append(structured)
            else:
                noise_features.append(self.rng.randn(X.shape[0]))

        noise_features = (
            np.column_stack(noise_features)
            if noise_features
            else np.empty((X.shape[0], 0))
        )
        X_with_noise = np.column_stack([X, noise_features])

        return X_with_noise

    def _compute_snr(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute signal-to-noise ratio."""

        overall_mean = np.mean(X, axis=0)

        between_var = 0.0
        for cls in np.unique(y):
            X_cls = X[y == cls]
            class_mean = np.mean(X_cls, axis=0)
            n_cls = len(X_cls)
            between_var += n_cls * np.sum((class_mean - overall_mean) ** 2)
        between_var /= len(X)

        within_var = 0.0
        for cls in np.unique(y):
            X_cls = X[y == cls]
            class_mean = np.mean(X_cls, axis=0)
            within_var += np.sum((X_cls - class_mean) ** 2)
        within_var /= len(X)

        snr = between_var / (within_var + 1e-10)
        return snr

    def _compute_fisher_ratio(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute Fisher discriminant ratio."""

        overall_mean = np.mean(X, axis=0)

        S_B = np.zeros((X.shape[1], X.shape[1]))
        for cls in np.unique(y):
            X_cls = X[y == cls]
            n_cls = len(X_cls)
            class_mean = np.mean(X_cls, axis=0)
            mean_diff = (class_mean - overall_mean).reshape(-1, 1)
            S_B += n_cls * (mean_diff @ mean_diff.T)

        S_W = np.zeros((X.shape[1], X.shape[1]))
        for cls in np.unique(y):
            X_cls = X[y == cls]
            class_mean = np.mean(X_cls, axis=0)
            S_W += (X_cls - class_mean).T @ (X_cls - class_mean)

        try:
            fisher_ratio = np.trace(np.linalg.pinv(S_W) @ S_B)
        except:
            fisher_ratio = 0.0

        return fisher_ratio

    def generate(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Generate a synthetic dataset with high-noise, poor-separability properties.
        """

        print(f"Generating synthetic dataset (seed={self.random_state})...")

        X_base, y = self._make_base_data()

        X_redundant = self._add_correlated_features(X_base)

        X_transformed = self._apply_non_gaussian_transform(X_redundant)

        if not self.is_baseline:
            X_noisy = self._add_heavy_noise(X_transformed, y)
        else:
            X_noisy = X_transformed

        X_final = self._add_pure_noise_features(X_noisy)

        props = self._compute_properties(X_final, y)

        print(f"  Generated: {X_final.shape[0]} samples, {X_final.shape[1]} features")
        print(f"  SNR: {props['snr']:.4f} (target: {self.target_snr:.4f})")
        print(
            f"  Fisher ratio: {props['fisher_ratio']:.4f} (target: {self.target_fisher_ratio:.4f})"
        )
        print(f"  Mean skewness: {props['mean_skewness']:.4f}")
        print(f"  Mean kurtosis: {props['mean_kurtosis']:.4f}")

        return X_final, y, props

    def _compute_properties(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Compute dataset properties."""

        props = {
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "n_informative": self.n_informative,
            "n_redundant": self.n_redundant,
            "n_noise": self.n_noise,
            "snr": self._compute_snr(X, y),
            "fisher_ratio": self._compute_fisher_ratio(X, y),
            "mean_skewness": np.mean(np.abs(stats.skew(X, axis=0))),
            "max_skewness": np.max(np.abs(stats.skew(X, axis=0))),
            "mean_kurtosis": np.mean(np.abs(stats.kurtosis(X, axis=0))),
            "max_kurtosis": np.max(np.abs(stats.kurtosis(X, axis=0))),
        }

        corr_matrix = np.corrcoef(X.T)
        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        off_diag_corr = np.abs(corr_matrix[mask])
        props["mean_abs_correlation"] = np.mean(off_diag_corr)
        props["max_abs_correlation"] = np.max(off_diag_corr)

        return props


def generate_dataset_suite(
    output_dir: Path,
    n_datasets: int = 20,
    dataset_type: str = "high_noise_poor_separability",
):
    """
    Generate a suite of synthetic datasets with varying properties.

    Args:
        output_dir: Directory to save datasets
        n_datasets: Number of datasets to generate
        dataset_type: Either "high_noise_poor_separability" or "baseline"
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    all_props = []

    print("=" * 80)
    print(f"GENERATING {n_datasets} SYNTHETIC {dataset_type.upper()} DATASETS")
    print("=" * 80)
    print()

    for i in range(n_datasets):
        print(f"\n[{i+1}/{n_datasets}] ", end="")

        if dataset_type == "baseline":
            n_samples = np.random.randint(1500, 5000)
            n_informative = np.random.randint(28, 38)
            n_redundant = np.random.randint(1, 2)
            n_noise = 0

            target_snr = np.random.uniform(10.0, 20.0)

            class_sep = np.random.uniform(3.5, 5.0)

        else:
            n_samples = np.random.randint(1500, 5000)
            n_informative = np.random.randint(8, 20)
            n_redundant = np.random.randint(5, 15)
            n_noise = np.random.randint(3, 10)

            log_snr_min = np.log10(0.03)
            log_snr_max = np.log10(0.09)
            log_snr = np.random.uniform(log_snr_min, log_snr_max)
            target_snr = 10**log_snr

            class_sep = np.random.uniform(0.55, 1.2)

        is_baseline = dataset_type == "baseline"

        generator = SyntheticGenerator(
            n_samples=n_samples,
            n_informative_features=n_informative,
            n_redundant_features=n_redundant,
            n_noise_features=n_noise,
            target_snr=target_snr,
            class_sep=class_sep,
            random_state=42 + i,
            is_baseline=is_baseline,
        )

        X, y, props = generator.generate()

        dataset_name = f"synthetic_{dataset_type}_{i:03d}"
        props["dataset_name"] = dataset_name
        props["dataset_type"] = dataset_type

        df = pd.DataFrame(X, columns=[f"feature_{j}" for j in range(X.shape[1])])
        df["target"] = y

        output_path = output_dir / f"{dataset_name}.csv"
        df.to_csv(output_path, index=False)

        all_props.append(props)

    summary_df = pd.DataFrame(all_props)
    summary_path = output_dir / f"synthetic_datasets_{dataset_type}_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 80)
    print(f"Generated {n_datasets} synthetic {dataset_type} datasets")
    print(f"Saved to: {output_dir}")
    print(f"Summary: {summary_path}")
    print("=" * 80)

    print("\nAVERAGE PROPERTIES:")
    print(
        f"  SNR:              {summary_df['snr'].mean():.4f} ± {summary_df['snr'].std():.4f}"
    )
    print(
        f"  Fisher ratio:     {summary_df['fisher_ratio'].mean():.4f} ± {summary_df['fisher_ratio'].std():.4f}"
    )
    print(
        f"  Mean skewness:    {summary_df['mean_skewness'].mean():.4f} ± {summary_df['mean_skewness'].std():.4f}"
    )
    print(
        f"  Mean kurtosis:    {summary_df['mean_kurtosis'].mean():.4f} ± {summary_df['mean_kurtosis'].std():.4f}"
    )
    print(
        f"  Mean |corr|:      {summary_df['mean_abs_correlation'].mean():.4f} ± {summary_df['mean_abs_correlation'].std():.4f}"
    )

    return summary_df


def main():
    """Main function."""

    output_dir_high_noise = Path("synthetic_datasets_high_noise_poor_separability")
    summary_high_noise = generate_dataset_suite(
        output_dir_high_noise,
        n_datasets=20,
        dataset_type="high_noise_poor_separability",
    )

    output_dir_baseline = Path("synthetic_datasets_baseline")
    summary_baseline = generate_dataset_suite(
        output_dir_baseline, n_datasets=20, dataset_type="baseline"
    )


if __name__ == "__main__":
    main()
