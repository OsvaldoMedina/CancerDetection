import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inference import (FEATURES, load_assets, validate_frame, predict,
                           read_csv_bytes, batch_predict, threshold_classes,
                           classification_metrics, range_warnings)
from src.cancer_detection import evaluate_model, positive_probabilities


@pytest.fixture(scope="module")
def assets():
    return load_assets()


def test_schema_order_metadata_and_mapping(assets):
    model, train, test, report = assets
    assert len(FEATURES) == 30
    assert tuple(model.feature_names_in_) == FEATURES
    numeric, errors, metadata = validate_frame(train[train.columns[::-1]])
    assert tuple(numeric.columns) == FEATURES
    assert metadata == ["source_index", "malignant"]
    assert not errors.size
    assert model.classes_.tolist() == [0, 1]
    bunch = load_breast_cancer(as_frame=True)
    for frame in (train, test):
        ids = frame.source_index.to_numpy()
        np.testing.assert_array_equal(frame.malignant, (bunch.target.loc[ids] == 0).astype(int))
        np.testing.assert_allclose(frame[list(FEATURES)], bunch.data.loc[ids])
    assert report["target_mapping"] == {"benign": 0, "malignant": 1}


@pytest.mark.parametrize("bad", [None, np.nan, np.inf, -np.inf, "texto", ""])
def test_bad_values_rejected(assets, bad):
    model, train, _, _ = assets
    frame = train.iloc[:1].astype(object)
    frame.loc[frame.index[0], FEATURES[0]] = bad
    _, errors, _ = validate_frame(frame)
    assert errors.iloc[0].fila == 1
    assert errors.iloc[0].columna == FEATURES[0]
    with pytest.raises(ValueError):
        predict(model, frame)


def test_schema_errors(assets):
    _, train, _, _ = assets
    for frame in (train.drop(columns=FEATURES[0]), train.assign(extra=1),
                  pd.concat([train, train[[FEATURES[0]]]], axis=1)):
        with pytest.raises(ValueError, match="Columnas"):
            validate_frame(frame)
    with pytest.raises(ValueError, match="duplicadas"):
        read_csv_bytes(b"mean radius,mean radius\n1,2\n")
    with pytest.raises(ValueError, match="campos"):
        read_csv_bytes((",".join(FEATURES) + "\n1,2\n").encode())


def test_pipeline_equality_and_no_fit(assets, monkeypatch):
    model, train, _, _ = assets
    frame = train.iloc[:5]
    expected = model.predict_proba(frame[list(FEATURES)])[:, list(model.classes_).index(1)]
    expected_class = model.predict(frame[list(FEATURES)])
    def forbidden(*args, **kwargs):
        raise AssertionError("Inference must never fit")
    monkeypatch.setattr(model, "fit", forbidden)
    monkeypatch.setattr(model.named_steps["scale"], "fit", forbidden)
    monkeypatch.setattr(model.named_steps["scale"], "fit_transform", forbidden)
    classes, p = predict(model, frame[frame.columns[::-1]])
    np.testing.assert_allclose(p, expected, rtol=0, atol=0)
    np.testing.assert_array_equal(classes, expected_class)


def test_class_probability_position():
    class ReversedModel:
        classes_ = np.array([1, 0])
        def predict_proba(self, x):
            return np.array([[.8, .2]])
    np.testing.assert_allclose(positive_probabilities(ReversedModel(), pd.DataFrame()), [.8])


def test_reports_reproduce(assets):
    model, _, test, report = assets
    actual, _, _ = evaluate_model(model, test[list(FEATURES)], test.malignant)
    for key, expected in report["test_metrics"].items():
        np.testing.assert_allclose(actual[key], expected, rtol=1e-6, atol=1e-8, err_msg=key)


def test_threshold_boundary_and_metrics():
    np.testing.assert_array_equal(threshold_classes([.49, .50, .51], .50), [0, 1, 1])
    np.testing.assert_array_equal(threshold_classes([0, 1], 0), [1, 1])
    np.testing.assert_array_equal(threshold_classes([0, 1], 1), [0, 1])
    with pytest.raises(ValueError):
        threshold_classes([.1], 1.1)
    values = classification_metrics([0, 0, 1, 1], [0, 1, 0, 1], [.1, .9, .2, .8])
    assert values["confusion_matrix"] == [[1, 1], [1, 1]]
    assert all(values[k] == .5 for k in ("accuracy", "precision", "specificity", "sensitivity_recall", "f1"))


def test_batch_partial_validation(assets):
    model, train, _, _ = assets
    frame = train.iloc[:3].reset_index(drop=True).astype(object)
    frame.loc[1, FEATURES[0]] = "incorrecto"
    frame.loc[2, "malignant"] = "etiqueta inválida"
    out, errors = batch_predict(model, frame, train)
    assert out.loc[1, "clase_predicha"] is None
    assert np.isnan(out.loc[1, "probabilidad_maligno"])
    assert out.loc[0, "clase_predicha"] in ("benigno", "maligno")
    assert "inválida" in out.loc[2, "estado_etiqueta"]
    assert errors.fila.tolist() == [2]
    parsed = read_csv_bytes(train.iloc[:2].to_csv(index=False).encode("utf-8-sig"))
    assert len(parsed) == 2


def test_reference_ranges_only_training(assets):
    _, train, _, _ = assets
    frame = train.iloc[:1].copy()
    frame.loc[frame.index[0], FEATURES[0]] = train[FEATURES[0]].max() + 1
    warnings = range_warnings(frame, train)
    assert warnings.iloc[0]["máximo train"] == train[FEATURES[0]].max()
