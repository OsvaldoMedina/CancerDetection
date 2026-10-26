"""Reusable functions for the Breast Cancer Wisconsin classification challenge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import calibration_curve
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    make_scorer,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TARGET_NAME = "malignant"


def load_dataset(save_path: Path | None = None) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Load the sklearn dataset and optionally save an auditable raw CSV.

    The sklearn target uses 0 for malignant and 1 for benign. This project
    converts the target to 1 for malignant so sensitivity refers to malignant
    observations throughout the analysis.
    """
    bunch = load_breast_cancer(as_frame=True)
    features = bunch.data.copy()
    target = (bunch.target == 0).astype(int).rename(TARGET_NAME)
    metadata = {
        "source": "scikit-learn load_breast_cancer",
        "dataset": "Breast Cancer Wisconsin Diagnostic",
        "rows": int(features.shape[0]),
        "features": int(features.shape[1]),
        "positive_class": "malignant",
        "target_mapping": {"benign": 0, "malignant": 1},
    }
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([features, target], axis=1).to_csv(save_path, index=False)
    return features, target, metadata


def validate_dataset(features: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
    """Validate the expected schema and return a compact quality report."""
    if features.empty:
        raise ValueError("The feature table is empty.")
    if len(features) != len(target):
        raise ValueError("Features and target have different row counts.")
    if features.columns.duplicated().any():
        raise ValueError("Duplicate feature names were found.")
    if not set(target.unique()).issubset({0, 1}):
        raise ValueError("The target must contain only 0 and 1.")
    if features.select_dtypes(exclude="number").shape[1] > 0:
        raise TypeError("All predictors must be numeric.")
    return {
        "rows": int(len(features)),
        "features": int(features.shape[1]),
        "missing_values": int(features.isna().sum().sum()),
        "duplicate_rows": int(features.duplicated().sum()),
        "infinite_values": int(np.isinf(features.to_numpy()).sum()),
        "class_counts": {str(k): int(v) for k, v in target.value_counts().sort_index().items()},
    }


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a fixed stratified train-test split."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=target,
    )


def build_models() -> dict[str, BaseEstimator]:
    """Return the baseline and candidate model pipelines."""
    scaled = lambda model: Pipeline([("scale", StandardScaler()), ("model", model)])
    return {
        "Dummy most frequent": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": scaled(
            LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)
        ),
        "Support vector machine": scaled(
            SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE)
        ),
        "K nearest neighbors": scaled(KNeighborsClassifier(n_neighbors=7)),
        "Decision tree": DecisionTreeClassifier(
            max_depth=5, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "Gradient boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def compare_models(
    models: dict[str, BaseEstimator],
    features: pd.DataFrame,
    target: pd.Series,
    folds: int = 5,
) -> pd.DataFrame:
    """Compare candidate models using only training data."""
    if folds < 2:
        raise ValueError("folds must be at least 2.")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "recall": make_scorer(recall_score, zero_division=0),
        "precision": make_scorer(precision_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
    }
    rows = []
    for name, model in models.items():
        scores = cross_validate(model, features, target, cv=cv, scoring=scoring, n_jobs=1)
        row: dict[str, Any] = {"model": name}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"mean_{metric}"] = float(np.mean(values))
            row[f"std_{metric}"] = float(np.std(values, ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_roc_auc", ascending=False).reset_index(drop=True)


def tune_model(
    name: str,
    model: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
    folds: int = 5,
) -> GridSearchCV:
    """Tune the selected candidate using training data only.

    Each grid is intentionally compact so the notebook remains practical on a
    laptop while still demonstrating nested preprocessing and model selection.
    """
    parameter_grids: dict[str, dict[str, list[Any]]] = {
        "Logistic regression": {"model__C": [0.1, 1.0, 10.0]},
        "Support vector machine": {
            "model__C": [0.5, 1.0, 5.0],
            "model__gamma": ["scale", 0.01],
        },
        "K nearest neighbors": {
            "model__n_neighbors": [5, 7, 11],
            "model__weights": ["uniform", "distance"],
        },
        "Decision tree": {"max_depth": [3, 5, 8], "min_samples_leaf": [1, 3, 5]},
        "Random forest": {"max_features": ["sqrt", 0.7], "min_samples_leaf": [1, 3]},
        "Gradient boosting": {"learning_rate": [0.05, 0.1], "max_depth": [1, 2]},
    }
    if name not in parameter_grids:
        raise ValueError(f"No tuning grid is defined for {name!r}.")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=model,
        param_grid=parameter_grids[name],
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        refit=True,
        return_train_score=False,
    )
    search.fit(features, target)
    return search


def positive_probabilities(model: BaseEstimator, features: pd.DataFrame) -> np.ndarray:
    """Return probabilities for the malignant class."""
    if hasattr(model, "predict_proba"):
        positions = np.flatnonzero(np.asarray(model.classes_) == 1)
        if len(positions) != 1:
            raise ValueError("The model must expose class 1 (malignant).")
        return np.asarray(model.predict_proba(features))[:, positions[0]]
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(features))
        return 1 / (1 + np.exp(-scores))
    raise TypeError("The selected model does not provide probabilities or decision scores.")


def evaluate_model(
    model: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[dict[str, float | int | list[list[int]]], np.ndarray, np.ndarray]:
    """Evaluate a fitted model on held-out observations."""
    predicted = np.asarray(model.predict(features))
    probabilities = positive_probabilities(model, features)
    tn, fp, fn, tp = confusion_matrix(target, predicted, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    metrics: dict[str, float | int | list[list[int]]] = {
        "accuracy": float(accuracy_score(target, predicted)),
        "sensitivity_recall": float(recall_score(target, predicted, zero_division=0)),
        "specificity": float(specificity),
        "precision": float(precision_score(target, predicted, zero_division=0)),
        "f1": float(f1_score(target, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "average_precision": float(average_precision_score(target, probabilities)),
        "brier_score": float(brier_score_loss(target, probabilities)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    return metrics, predicted, probabilities


def compute_permutation_importance(
    model: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    """Compute model-agnostic feature importance on held-out data."""
    result = permutation_importance(
        model,
        features,
        target,
        scoring="roc_auc",
        n_repeats=20,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    return (
        pd.DataFrame(
            {
                "feature": features.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def calibration_points(target: pd.Series, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return observed and predicted probability points for a calibration plot."""
    observed, predicted = calibration_curve(target, probabilities, n_bins=8, strategy="quantile")
    return observed, predicted


def save_json(data: dict[str, Any], path: Path) -> None:
    """Save JSON with stable formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def save_model(model: BaseEstimator, path: Path) -> None:
    """Persist the fitted sklearn pipeline."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
