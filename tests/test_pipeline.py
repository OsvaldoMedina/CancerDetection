from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cancer_detection import (  # noqa: E402
    build_models,
    compare_models,
    evaluate_model,
    load_dataset,
    split_dataset,
    validate_dataset,
)


def test_dataset_schema_and_target_mapping():
    features, target, metadata = load_dataset()
    quality = validate_dataset(features, target)
    assert features.shape == (569, 30)
    assert set(target.unique()) == {0, 1}
    assert target.sum() == 212
    assert metadata["positive_class"] == "malignant"
    assert quality["missing_values"] == 0


def test_split_is_reproducible_and_stratified():
    features, target, _ = load_dataset()
    first = split_dataset(features, target)
    second = split_dataset(features, target)
    assert first[0].index.equals(second[0].index)
    assert np.isclose(first[2].mean(), target.mean(), atol=0.01)
    assert np.isclose(first[3].mean(), target.mean(), atol=0.01)


def test_small_model_comparison_and_evaluation():
    features, target, _ = load_dataset()
    x_train, x_test, y_train, y_test = split_dataset(features, target)
    models = build_models()
    selected = {name: models[name] for name in ["Dummy most frequent", "Logistic regression"]}
    comparison = compare_models(selected, x_train, y_train, folds=3)
    assert set(comparison["model"]) == set(selected)
    model = selected["Logistic regression"].fit(x_train, y_train)
    metrics, predicted, probabilities = evaluate_model(model, x_test, y_test)
    assert metrics["roc_auc"] > 0.90
    assert len(predicted) == len(y_test)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
