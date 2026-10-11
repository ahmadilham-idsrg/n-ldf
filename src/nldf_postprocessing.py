"""Post-process a completed N-LDF benchmark without retraining any model.

Google Colab example
--------------------
!pip install -q scipy pandas numpy matplotlib scikit-posthocs
!python nldf_postprocessing.py \
    --input nldf_results_41_v2_final/fold_level_results.csv \
    --audit nldf_results_41_v2_final/dataset_audit.csv \
    --output-dir nldf_paper_analysis

The script never modifies the input CSV files. All derived artifacts are written
to ``--output-dir``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import combinations, product
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    import scikit_posthocs as sp
except ImportError:
    sp = None


DEFAULT_METHODS = (
    "Baseline",
    "SMOTE",
    "B-SMOTE",
    "ADASYN",
    "SMOTE-Tomek",
    "N-LDF",
)
DEFAULT_CLASSIFIERS = ("RF", "SVM-Linear", "SVM-RBF", "XGBoost")
METRICS = (
    "F1",
    "Precision",
    "Recall",
    "Specificity",
    "G-Mean",
    "AUC-ROC",
    "AUPRC",
)
KEY_COLUMNS = ("Dataset", "Repeat", "Fold", "Method", "Classifier")


@dataclass(frozen=True)
class AnalysisConfig:
    input_path: str
    audit_path: str | None
    output_dir: str
    nominal_folds: int
    nominal_repeats: int
    bootstrap_iterations: int
    seed: int
    methods: tuple[str, ...]
    classifiers: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correct, auditable post-processing for the N-LDF benchmark"
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="Completed fold_level_results.csv")
    parser.add_argument("--audit", type=Path,
                        help="Optional dataset_audit.csv from the same run")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("nldf_paper_analysis"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def validate_results(results: pd.DataFrame) -> None:
    required = set(KEY_COLUMNS) | {"Severity"} | set(METRICS)
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"fold-level file is missing required columns: {missing}")
    if results.empty:
        raise ValueError("fold-level file contains no rows")
    for metric in METRICS:
        results[metric] = pd.to_numeric(results[metric], errors="coerce")
    if results[list(METRICS)].isna().all(axis=None):
        raise ValueError("all metric values are missing")


def normalized_methods(results: pd.DataFrame) -> tuple[str, ...]:
    observed = list(dict.fromkeys(results["Method"].astype(str)))
    ordered = [method for method in DEFAULT_METHODS if method in observed]
    ordered.extend(method for method in observed if method not in ordered)
    return tuple(ordered)


def normalized_classifiers(results: pd.DataFrame) -> tuple[str, ...]:
    observed = list(dict.fromkeys(results["Classifier"].astype(str)))
    ordered = [name for name in DEFAULT_CLASSIFIERS if name in observed]
    ordered.extend(name for name in observed if name not in ordered)
    return tuple(ordered)


def infer_dataset_design(
    results: pd.DataFrame,
    audit: pd.DataFrame | None,
    nominal_folds: int,
    nominal_repeats: int,
) -> pd.DataFrame:
    observed = (
        results.groupby("Dataset", as_index=False)
        .agg(
            ObservedFolds=("Fold", lambda values: int(pd.Series(values).nunique())),
            ObservedRepeats=("Repeat", lambda values: int(pd.Series(values).nunique())),
        )
    )
    if audit is None:
        design = observed.rename(
            columns={"ObservedFolds": "ExpectedFolds", "ObservedRepeats": "ExpectedRepeats"}
        )
        design["DesignSource"] = "observed maxima; audit unavailable"
        return design

    audit = audit.copy()
    name_column = "name" if "name" in audit.columns else "Dataset"
    minority_column = "n_minority" if "n_minority" in audit.columns else None
    problematic_column = "is_problematic" if "is_problematic" in audit.columns else None
    if minority_column is None:
        raise ValueError("dataset audit must contain n_minority")

    audit = audit.rename(columns={name_column: "Dataset"})
    audit[minority_column] = pd.to_numeric(audit[minority_column], errors="raise").astype(int)
    if problematic_column:
        audit[problematic_column] = (
            audit[problematic_column].astype(str).str.lower().isin({"true", "1", "yes"})
        )
    else:
        audit["is_problematic"] = False
        problematic_column = "is_problematic"

    rows = []
    for row in audit.itertuples(index=False):
        record = row._asdict()
        n_minority = int(record[minority_column])
        problematic = bool(record[problematic_column])
        if problematic:
            folds = min(nominal_folds, max(2, n_minority // 5))
            repeats = min(nominal_repeats, 2)
        elif n_minority < nominal_folds * 2:
            folds = max(2, n_minority // 2)
            repeats = nominal_repeats
        else:
            folds = nominal_folds
            repeats = nominal_repeats
        rows.append({
            "Dataset": str(record["Dataset"]),
            "ExpectedFolds": int(folds),
            "ExpectedRepeats": int(repeats),
            "DesignSource": "benchmark protocol and dataset audit",
        })
    design = pd.DataFrame(rows)
    return design.merge(observed, on="Dataset", how="left")


def build_completeness(
    results: pd.DataFrame,
    design: pd.DataFrame,
    methods: Sequence[str],
    classifiers: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    duplicates = (
        results.groupby(list(KEY_COLUMNS), as_index=False)
        .size()
        .query("size > 1")
        .rename(columns={"size": "DuplicateRows"})
    )

    actual_counts = (
        results.groupby(["Dataset", "Method", "Classifier"], as_index=False)
        .size()
        .rename(columns={"size": "CompletedEvaluations"})
    )
    expected_rows = []
    missing_rows = []
    actual_keys = set(map(tuple, results[list(KEY_COLUMNS)].itertuples(index=False, name=None)))
    for row in design.itertuples(index=False):
        dataset = row.Dataset
        folds = int(row.ExpectedFolds)
        repeats = int(row.ExpectedRepeats)
        expected_count = folds * repeats
        for method, classifier in product(methods, classifiers):
            expected_rows.append({
                "Dataset": dataset,
                "Method": method,
                "Classifier": classifier,
                "ExpectedEvaluations": expected_count,
            })
            for repeat, fold in product(range(repeats), range(folds)):
                key = (dataset, repeat, fold, method, classifier)
                if key not in actual_keys:
                    missing_rows.append(dict(zip(KEY_COLUMNS, key)))

    completeness = pd.DataFrame(expected_rows).merge(
        actual_counts, on=["Dataset", "Method", "Classifier"], how="left"
    )
    completeness["CompletedEvaluations"] = (
        completeness["CompletedEvaluations"].fillna(0).astype(int)
    )
    completeness["MissingEvaluations"] = (
        completeness["ExpectedEvaluations"] - completeness["CompletedEvaluations"]
    )
    completeness["CompletionPct"] = (
        100 * completeness["CompletedEvaluations"] / completeness["ExpectedEvaluations"]
    )
    missing = pd.DataFrame(missing_rows, columns=KEY_COLUMNS)
    return completeness, missing, duplicates


def bootstrap_mean_ci(
    values: Iterable[float], iterations: int, seed: int
) -> tuple[float, float]:
    data = np.asarray(list(values), dtype=float)
    data = data[np.isfinite(data)]
    if len(data) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(data), size=(iterations, len(data)))
    means = data[indices].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]).astype(float))


def dataset_level_tables(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    direct = (
        results.groupby(["Dataset", "Severity", "Method"], as_index=False)[list(METRICS)]
        .mean()
    )
    classifier_blocks = (
        results.groupby(
            ["Dataset", "Severity", "Method", "Classifier"], as_index=False
        )[list(METRICS)]
        .mean()
    )
    equal_classifier = (
        classifier_blocks.groupby(["Dataset", "Severity", "Method"], as_index=False)[
            list(METRICS)
        ]
        .mean()
    )
    classifier_counts = (
        classifier_blocks.groupby(["Dataset", "Method"], as_index=False)["Classifier"]
        .nunique()
        .rename(columns={"Classifier": "AvailableClassifiers"})
    )
    equal_classifier = equal_classifier.merge(
        classifier_counts, on=["Dataset", "Method"], how="left"
    )
    return direct, equal_classifier, classifier_blocks


def global_summary(
    dataset_method: pd.DataFrame,
    methods: Sequence[str],
    iterations: int,
    seed: int,
    aggregation: str,
) -> pd.DataFrame:
    pivot = dataset_method.pivot(index="Dataset", columns="Method", values="F1")
    available = [method for method in methods if method in pivot.columns]
    complete = pivot[available].dropna(axis=0, how="any")
    ranks = complete.rank(axis=1, ascending=False, method="average")
    rows = []
    for method_index, method in enumerate(available):
        block = dataset_method.loc[dataset_method["Method"] == method]
        record = {
            "Aggregation": aggregation,
            "Method": method,
            "N_Datasets": int(block["Dataset"].nunique()),
            "N_CompleteRankDatasets": int(len(complete)),
            "MeanF1Rank": float(ranks[method].mean()),
        }
        for metric_index, metric in enumerate(METRICS):
            values = block[metric].to_numpy(dtype=float)
            low, high = bootstrap_mean_ci(
                values, iterations, seed + 1000 * method_index + metric_index
            )
            record[metric] = float(np.nanmean(values))
            record[f"{metric}_CI95_Low"] = low
            record[f"{metric}_CI95_High"] = high
        rows.append(record)
    return pd.DataFrame(rows).sort_values("MeanF1Rank")


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    total = len(p_values)
    for position, index in enumerate(order):
        candidate = min(1.0, (total - position) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def rank_biserial(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if not np.any(differences != 0):
        return 0.0
    ranks = stats.rankdata(np.abs(differences))
    positive = float(np.sum(ranks[differences > 0]))
    negative = float(np.sum(ranks[differences < 0]))
    denominator = positive + negative
    return (positive - negative) / denominator if denominator else 0.0


def paired_test_row(
    pivot: pd.DataFrame,
    method_a: str,
    method_b: str,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    paired = pivot[[method_a, method_b]].dropna()
    differences = (paired[method_a] - paired[method_b]).to_numpy(dtype=float)
    try:
        statistic, raw_p = stats.wilcoxon(
            paired[method_a], paired[method_b], zero_method="pratt", alternative="two-sided"
        )
    except ValueError:
        statistic, raw_p = 0.0, 1.0
    ci_low, ci_high = bootstrap_mean_ci(differences, iterations, seed)
    return {
        "MethodA": method_a,
        "MethodB": method_b,
        "N_Datasets": len(paired),
        "MeanDifference": float(np.mean(differences)),
        "MeanDifference_CI95_Low": ci_low,
        "MeanDifference_CI95_High": ci_high,
        "MedianDifference": float(np.median(differences)),
        "RankBiserialCorrelation": rank_biserial(differences),
        "WilcoxonStatistic": float(statistic),
        "RawP": float(raw_p),
    }


def inferential_analysis(
    dataset_method: pd.DataFrame,
    methods: Sequence[str],
    iterations: int,
    seed: int,
    aggregation: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    pivot = dataset_method.pivot(index="Dataset", columns="Method", values="F1")
    available = [method for method in methods if method in pivot.columns]
    complete = pivot[available].dropna(axis=0, how="any")
    if len(complete) < 2:
        raise ValueError(f"insufficient complete datasets for {aggregation} statistics")

    omnibus_rows = []
    families = [("All strategies", available)]
    oversamplers = [method for method in available if method != "Baseline"]
    if len(oversamplers) >= 3:
        families.append(("Oversamplers only", oversamplers))
    for family, family_methods in families:
        matrix = complete[family_methods]
        statistic, p_value = stats.friedmanchisquare(
            *[matrix[method].to_numpy() for method in family_methods]
        )
        omnibus_rows.append({
            "Aggregation": aggregation,
            "Family": family,
            "N_Datasets": len(matrix),
            "N_Methods": len(family_methods),
            "Statistic": float(statistic),
            "DF": len(family_methods) - 1,
            "PValue": float(p_value),
        })

    planned_rows = []
    if "N-LDF" in available:
        comparators = [method for method in available if method != "N-LDF"]
        for index, comparator in enumerate(comparators):
            planned_rows.append(
                paired_test_row(
                    complete, "N-LDF", comparator, iterations, seed + 10000 + index
                )
            )
        adjusted = holm_adjust([row["RawP"] for row in planned_rows])
        for row, adjusted_p in zip(planned_rows, adjusted):
            row["Aggregation"] = aggregation
            row["HolmP_PlannedFamily"] = float(adjusted_p)
            row["SignificantAt0.05"] = bool(adjusted_p < 0.05)

    exploratory_rows = []
    for index, (method_a, method_b) in enumerate(combinations(available, 2)):
        exploratory_rows.append(
            paired_test_row(
                complete, method_a, method_b, iterations, seed + 20000 + index
            )
        )
    adjusted = holm_adjust([row["RawP"] for row in exploratory_rows])
    for row, adjusted_p in zip(exploratory_rows, adjusted):
        row["Aggregation"] = aggregation
        row["HolmP_ExploratoryFamily"] = float(adjusted_p)

    nemenyi = None
    if sp is not None and len(available) >= 3:
        nemenyi = sp.posthoc_nemenyi_friedman(complete[available].to_numpy())
        nemenyi.index = available
        nemenyi.columns = available

    return (
        pd.DataFrame(omnibus_rows),
        pd.DataFrame(planned_rows),
        pd.DataFrame(exploratory_rows),
        nemenyi,
    )


def classifier_summary(
    blocks: pd.DataFrame, iterations: int, seed: int
) -> pd.DataFrame:
    rows = []
    for group_index, ((classifier, method), group) in enumerate(
        blocks.groupby(["Classifier", "Method"], sort=False)
    ):
        record = {
            "Classifier": classifier,
            "Method": method,
            "N_Datasets": int(group["Dataset"].nunique()),
        }
        for metric_index, metric in enumerate(METRICS):
            values = group[metric].to_numpy(dtype=float)
            low, high = bootstrap_mean_ci(
                values, iterations, seed + group_index * 100 + metric_index
            )
            record[metric] = float(np.nanmean(values))
            record[f"{metric}_CI95_Low"] = low
            record[f"{metric}_CI95_High"] = high
        rows.append(record)
    return pd.DataFrame(rows)


def severity_summary(
    dataset_method: pd.DataFrame, iterations: int, seed: int
) -> pd.DataFrame:
    rows = []
    for group_index, ((severity, method), group) in enumerate(
        dataset_method.groupby(["Severity", "Method"], sort=False)
    ):
        record = {
            "Severity": severity,
            "Method": method,
            "N_Datasets": int(group["Dataset"].nunique()),
        }
        for metric_index, metric in enumerate(METRICS):
            values = group[metric].to_numpy(dtype=float)
            low, high = bootstrap_mean_ci(
                values, iterations, seed + group_index * 100 + metric_index
            )
            record[metric] = float(np.nanmean(values))
            record[f"{metric}_CI95_Low"] = low
            record[f"{metric}_CI95_High"] = high
        rows.append(record)
    return pd.DataFrame(rows)


def nldf_diagnostics(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    diagnostic_candidates = [
        "IR", "delta", "q_boundary", "lambda_prior", "n_requested",
        "target_minority", "n_generated", "n_accepted", "n_attempted",
        "target_shortfall", "n_rejected_distance_original",
        "n_rejected_distance_synthetic", "n_rejected_distance",
        "n_rejected_density", "n_rejected_majority", "n_rejected_maj",
        "n_covariance_failures", "sampling_ratio", "achieved_ratio",
        "acceptance_rate", "eta_min", "eta_max", "eta_mean", "weights_max",
        "accepted_eta_mean",
    ]
    available = [column for column in diagnostic_candidates if column in results.columns]
    base_columns = [
        column for column in ["Dataset", "Severity", "Repeat", "Fold", "Method"]
        if column in results.columns
    ]
    nldf = results.loc[results["Method"] == "N-LDF", base_columns + available].copy()
    fold_diagnostics = nldf.drop_duplicates(base_columns).reset_index(drop=True)
    numeric = [column for column in available if pd.api.types.is_numeric_dtype(fold_diagnostics[column])]
    if not numeric:
        return fold_diagnostics, pd.DataFrame(), pd.DataFrame()
    by_dataset = fold_diagnostics.groupby(["Dataset", "Severity"], as_index=False)[numeric].mean()
    by_severity = fold_diagnostics.groupby("Severity", as_index=False)[numeric].agg(
        ["mean", "median", "min", "max"]
    )
    by_severity.columns = [
        "Severity" if column[0] == "Severity" else f"{column[0]}_{column[1]}"
        for column in by_severity.columns
    ]
    return fold_diagnostics, by_dataset, by_severity


def aggregation_sensitivity(
    direct_summary: pd.DataFrame, equal_summary: pd.DataFrame
) -> pd.DataFrame:
    columns = ["Method", "F1", "MeanF1Rank"]
    direct = direct_summary[columns].rename(
        columns={"F1": "DirectF1", "MeanF1Rank": "DirectMeanRank"}
    )
    equal = equal_summary[columns].rename(
        columns={"F1": "EqualClassifierF1", "MeanF1Rank": "EqualClassifierMeanRank"}
    )
    merged = direct.merge(equal, on="Method", how="outer")
    merged["F1Difference_EqualMinusDirect"] = (
        merged["EqualClassifierF1"] - merged["DirectF1"]
    )
    merged["RankDifference_EqualMinusDirect"] = (
        merged["EqualClassifierMeanRank"] - merged["DirectMeanRank"]
    )
    return merged


def method_order(frame: pd.DataFrame, methods: Sequence[str]) -> pd.DataFrame:
    result = frame.copy()
    result["Method"] = pd.Categorical(result["Method"], categories=methods, ordered=True)
    return result.sort_values("Method")


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def create_figures(
    direct_summary: pd.DataFrame,
    severity: pd.DataFrame,
    classifiers: pd.DataFrame,
    output_dir: Path,
    methods: Sequence[str],
) -> None:
    ordered = method_order(direct_summary, methods)
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    errors = np.vstack((
        ordered["F1"] - ordered["F1_CI95_Low"],
        ordered["F1_CI95_High"] - ordered["F1"],
    ))
    axis.bar(ordered["Method"].astype(str), ordered["F1"], color="#4472C4")
    axis.errorbar(
        np.arange(len(ordered)), ordered["F1"], yerr=errors,
        fmt="none", ecolor="black", capsize=4,
    )
    axis.set_ylabel("Dataset-level mean F1")
    axis.tick_params(axis="x", rotation=25)
    save_figure(fig, output_dir, "fig_global_f1")

    severity_order = [value for value in ("Mild", "High", "Extreme")
                      if value in severity["Severity"].unique()]
    fig, axes = plt.subplots(1, max(len(severity_order), 1), figsize=(14, 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, level in zip(axes, severity_order):
        block = method_order(severity.loc[severity["Severity"] == level], methods)
        axis.bar(block["Method"].astype(str), block["F1"], color="#70AD47")
        axis.set_title(f"{level} (n={int(block['N_Datasets'].max())})")
        axis.tick_params(axis="x", rotation=60)
    axes[0].set_ylabel("Dataset-level mean F1")
    save_figure(fig, output_dir, "fig_f1_by_severity")

    classifier_order = [name for name in DEFAULT_CLASSIFIERS
                        if name in classifiers["Classifier"].unique()]
    fig, axes = plt.subplots(1, max(len(classifier_order), 1), figsize=(16, 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, classifier in zip(axes, classifier_order):
        block = method_order(
            classifiers.loc[classifiers["Classifier"] == classifier], methods
        )
        axis.bar(block["Method"].astype(str), block["F1"], color="#ED7D31")
        axis.set_title(classifier)
        axis.tick_params(axis="x", rotation=60)
    axes[0].set_ylabel("Dataset-level mean F1")
    save_figure(fig, output_dir, "fig_f1_by_classifier")

    rank_block = direct_summary.sort_values("MeanF1Rank")
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.barh(rank_block["Method"], rank_block["MeanF1Rank"], color="#A5A5A5")
    axis.invert_yaxis()
    axis.set_xlabel("Mean F1 rank (lower is better)")
    save_figure(fig, output_dir, "fig_average_f1_ranks")


def summarize_runtime(results: pd.DataFrame) -> pd.DataFrame:
    runtime_columns = [
        column for column in results.columns
        if any(token in column.lower() for token in ("time", "duration", "seconds"))
        and pd.api.types.is_numeric_dtype(results[column])
    ]
    if not runtime_columns:
        return pd.DataFrame()
    return results.groupby("Method", as_index=False)[runtime_columns].agg(["mean", "median", "sum"])


def write_manifest(config: AnalysisConfig, input_path: Path, audit_path: Path | None,
                   output_dir: Path, results: pd.DataFrame) -> None:
    packages = {}
    for package in ("numpy", "pandas", "scipy", "matplotlib", "scikit-posthocs"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not-installed"
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "config": asdict(config),
        "input_sha256": sha256_file(input_path),
        "audit_sha256": sha256_file(audit_path) if audit_path else None,
        "input_rows": int(len(results)),
        "input_datasets": int(results["Dataset"].nunique()),
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if args.audit is not None and not args.audit.exists():
        raise FileNotFoundError(args.audit)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(args.input)
    validate_results(results)
    audit = pd.read_csv(args.audit) if args.audit else None
    methods = normalized_methods(results)
    classifiers = normalized_classifiers(results)
    config = AnalysisConfig(
        input_path=str(args.input),
        audit_path=str(args.audit) if args.audit else None,
        output_dir=str(args.output_dir),
        nominal_folds=args.folds,
        nominal_repeats=args.repeats,
        bootstrap_iterations=args.bootstrap,
        seed=args.seed,
        methods=methods,
        classifiers=classifiers,
    )

    design = infer_dataset_design(results, audit, args.folds, args.repeats)
    completeness, missing, duplicates = build_completeness(
        results, design, methods, classifiers
    )
    save_csv(design, args.output_dir / "expected_design_by_dataset.csv")
    save_csv(completeness, args.output_dir / "completeness_report.csv")
    save_csv(missing, args.output_dir / "missing_evaluations.csv")
    save_csv(duplicates, args.output_dir / "duplicate_evaluations.csv")
    overall_completeness = pd.DataFrame([{
        "ExpectedEvaluations": int(completeness["ExpectedEvaluations"].sum()),
        "CompletedEvaluations": int(completeness["CompletedEvaluations"].sum()),
        "MissingEvaluations": int(completeness["MissingEvaluations"].sum()),
        "CompletionPct": float(
            100 * completeness["CompletedEvaluations"].sum()
            / completeness["ExpectedEvaluations"].sum()
        ),
        "DuplicateKeyCount": int(len(duplicates)),
    }])
    save_csv(overall_completeness, args.output_dir / "overall_completeness.csv")

    direct, equal_classifier, classifier_blocks = dataset_level_tables(results)
    save_csv(direct, args.output_dir / "dataset_method_direct_pooling.csv")
    save_csv(equal_classifier, args.output_dir / "dataset_method_equal_classifier.csv")
    save_csv(classifier_blocks, args.output_dir / "dataset_method_classifier_means.csv")

    direct_global = global_summary(
        direct, methods, args.bootstrap, args.seed, "direct_pooling"
    )
    equal_global = global_summary(
        equal_classifier, methods, args.bootstrap, args.seed + 1, "equal_classifier"
    )
    save_csv(pd.concat([direct_global, equal_global], ignore_index=True),
             args.output_dir / "global_summary_corrected.csv")

    classifier_table = classifier_summary(
        classifier_blocks, args.bootstrap, args.seed + 2
    )
    severity_table = severity_summary(direct, args.bootstrap, args.seed + 3)
    save_csv(classifier_table, args.output_dir / "classifier_method_summary.csv")
    save_csv(severity_table, args.output_dir / "severity_summary_corrected.csv")
    save_csv(
        aggregation_sensitivity(direct_global, equal_global),
        args.output_dir / "aggregation_sensitivity.csv",
    )

    all_omnibus = []
    all_planned = []
    all_exploratory = []
    for offset, (label, table) in enumerate((
        ("direct_pooling", direct),
        ("equal_classifier", equal_classifier),
    )):
        omnibus, planned, exploratory, nemenyi = inferential_analysis(
            table, methods, args.bootstrap, args.seed + 100 * offset, label
        )
        all_omnibus.append(omnibus)
        all_planned.append(planned)
        all_exploratory.append(exploratory)
        if nemenyi is not None:
            nemenyi.to_csv(args.output_dir / f"nemenyi_{label}.csv")
    save_csv(pd.concat(all_omnibus, ignore_index=True),
             args.output_dir / "friedman_tests.csv")
    save_csv(pd.concat(all_planned, ignore_index=True),
             args.output_dir / "nldf_pairwise_holm.csv")
    save_csv(pd.concat(all_exploratory, ignore_index=True),
             args.output_dir / "all_pairwise_exploratory.csv")

    fold_diag, dataset_diag, severity_diag = nldf_diagnostics(results)
    save_csv(fold_diag, args.output_dir / "nldf_diagnostics_by_fold.csv")
    save_csv(dataset_diag, args.output_dir / "nldf_diagnostics_by_dataset.csv")
    save_csv(severity_diag, args.output_dir / "nldf_diagnostics_by_severity.csv")

    runtime = summarize_runtime(results)
    if not runtime.empty:
        runtime.to_csv(args.output_dir / "runtime_summary.csv")

    create_figures(
        direct_global, severity_table, classifier_table, args.output_dir, methods
    )
    write_manifest(config, args.input, args.audit, args.output_dir, results)

    print("Post-processing complete")
    print(f"Input rows: {len(results):,}")
    print(f"Datasets: {results['Dataset'].nunique()}")
    print(f"Expected evaluations: {int(overall_completeness.iloc[0]['ExpectedEvaluations']):,}")
    print(f"Missing evaluations: {int(overall_completeness.iloc[0]['MissingEvaluations']):,}")
    print(f"Output directory: {args.output_dir}")
    if runtime.empty:
        print("Runtime summary was not generated because no timing column exists.")
    if sp is None:
        print("Nemenyi output skipped because scikit-posthocs is not installed.")


if __name__ == "__main__":
    main()