"""Local inference, strict CSV validation and educational threshold evaluation.

No training, user-file persistence, or user-supplied model deserialization.
"""
from __future__ import annotations

import csv
import io
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.cancer_detection import positive_probabilities

ROOT = Path(__file__).resolve().parents[1]
FEATURES = tuple(load_breast_cancer().feature_names)
METADATA = ("source_index", "malignant")
CLASSES = {0: "benigno", 1: "maligno"}
NAMES = dict(zip(
    ("radius", "texture", "perimeter", "area", "smoothness", "compactness",
     "concavity", "concave points", "symmetry", "fractal dimension"),
    ("Radio", "Textura", "Perímetro", "Área", "Suavidad", "Compacidad",
     "Concavidad", "Puntos cóncavos", "Simetría", "Dimensión fractal")))


def label(feature):
    base = feature.removeprefix("mean ").removeprefix("worst ").removesuffix(" error")
    return NAMES[base]


def load_assets(root=ROOT):
    """Fail closed if versions, class mapping, schema or pipeline disagree."""
    root = Path(root)
    train = pd.read_csv(root / "data/processed/train.csv")
    test = pd.read_csv(root / "data/processed/test.csv")
    report = json.loads((root / "reports/metrics.json").read_text(encoding="utf-8"))
    if report["target_mapping"] != {"benign": 0, "malignant": 1}:
        raise ValueError("El reporte no confirma benigno=0 y maligno=1.")
    original = load_breast_cancer(as_frame=True)
    for name, data in (("train", train), ("test", test)):
        if tuple(c for c in data.columns if c not in METADATA) != FEATURES:
            raise ValueError(f"Esquema incompatible en {name}.csv.")
        numeric, errors, _ = validate_frame(data)
        if len(errors) or numeric.empty or not data.malignant.isin([0, 1]).all():
            raise ValueError(f"Datos originales inválidos en {name}.csv.")
        ids = data.source_index.to_numpy()
        if not np.isin(ids, original.data.index).all():
            raise ValueError("source_index no corresponde al dataset original.")
        expected = (original.target.loc[ids].to_numpy() == 0).astype(int)
        if not np.array_equal(expected, data.malignant.to_numpy()):
            raise ValueError("Las etiquetas no coinciden con la recodificación original.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            model = joblib.load(root / "models/best_model.joblib")
    except Exception as exc:
        raise RuntimeError(
            "No se pudo cargar el modelo local compatible. Instala requirements.txt "
            "en Python 3.12 y comprueba el artefacto original. No se reentrenó el modelo. "
            f"Detalle: {exc}") from exc
    if not isinstance(model, Pipeline) or not any(
        isinstance(step, StandardScaler) for _, step in model.steps
    ):
        raise ValueError("Se requiere el pipeline original completo con StandardScaler.")
    if set(model.classes_) != {0, 1}:
        raise ValueError("El modelo debe contener las clases 0 y 1.")
    if getattr(model, "n_features_in_", None) != 30:
        raise ValueError("El modelo no espera 30 variables.")
    if hasattr(model, "feature_names_in_") and tuple(model.feature_names_in_) != FEATURES:
        raise ValueError("El orden de feature_names_in_ no coincide con el dataset.")
    predict(model, test.iloc[:1])  # check inference compatibility, never fit
    return model, train, test, report


def validate_frame(frame):
    """Return ordered numbers, cell errors (1-based data rows), metadata columns.

    Structural errors reject the file; bad cells reject only their rows.
    Optional observed labels are checked separately, never used as predictors.
    """
    duplicates = frame.columns[frame.columns.duplicated()].tolist()
    missing = [c for c in FEATURES if c not in frame]
    extras = [c for c in frame if c not in FEATURES + METADATA]
    if duplicates or missing or extras:
        raise ValueError(f"Columnas faltantes: {missing}; duplicadas: {duplicates}; adicionales: {extras}.")
    raw = frame.loc[:, list(FEATURES)].reset_index(drop=True)
    numeric = raw.apply(pd.to_numeric, errors="coerce").astype(float)
    errors = []
    for r, c in zip(*np.where(~np.isfinite(numeric.to_numpy()))):
        value = raw.iat[r, c]
        reason = "Dato faltante" if pd.isna(value) or str(value).strip() == "" else "Se requiere un número finito"
        errors.append({"fila": int(r + 1), "columna": FEATURES[c], "error": reason})
    return numeric, pd.DataFrame(errors, columns=["fila", "columna", "error"]), [c for c in METADATA if c in frame]


def read_csv_bytes(content):
    try:
        text = content.decode("utf-8-sig")
        rows = list(csv.reader(io.StringIO(text), strict=True))
    except (UnicodeError, csv.Error) as exc:
        raise ValueError("CSV inválido: usa UTF-8, comas y punto decimal.") from exc
    if not rows or not rows[0]:
        raise ValueError("El CSV está vacío.")
    header = rows[0]
    if len(header) != len(set(header)):
        raise ValueError("El CSV tiene columnas duplicadas; corrige su encabezado.")
    for n, row in enumerate(rows[1:], 1):
        if len(row) != len(header):
            raise ValueError(f"Fila de datos {n}: contiene {len(row)} campos; se esperaban {len(header)}.")
    frame = pd.DataFrame(rows[1:], columns=header)
    if frame.empty:
        raise ValueError("El CSV no contiene registros.")
    validate_frame(frame)
    return frame


def predict(model, frame):
    numeric, errors, _ = validate_frame(frame)
    if len(errors):
        raise ValueError("Hay mediciones faltantes, no numéricas o no finitas.")
    with np.errstate(over="raise", invalid="raise"):
        probabilities = positive_probabilities(model, numeric)
        classes = np.asarray(model.predict(numeric), dtype=int)
    if not np.isfinite(probabilities).all():
        raise ValueError("El modelo produjo probabilidades no finitas.")
    return classes, probabilities


def range_warnings(frame, train):
    numbers = frame.loc[:, list(FEATURES)].astype(float)
    low, high = train[list(FEATURES)].min(), train[list(FEATURES)].max()
    records = []
    for r, c in zip(*np.where(((numbers < low) | (numbers > high)).to_numpy())):
        feature = FEATURES[c]
        records.append({"fila": int(r + 1), "variable": feature,
                        "valor": numbers.iat[r, c], "mínimo train": low[feature],
                        "máximo train": high[feature]})
    return pd.DataFrame(records)


def batch_predict(model, frame, train):
    numeric, errors, metadata = validate_frame(frame)
    result = frame.reset_index(drop=True).copy()
    result["clase_predicha"] = pd.Series([None] * len(frame), dtype=object)
    result["probabilidad_benigno"] = np.nan
    result["probabilidad_maligno"] = np.nan
    result["estado_validacion"] = "válida"
    bad = set(errors.fila - 1)
    for row in bad:
        details = errors[errors.fila == row + 1]
        result.at[row, "estado_validacion"] = "; ".join(details.columna + ": " + details.error)
    valid = [i for i in range(len(frame)) if i not in bad]
    if valid:
        classes, p = predict(model, numeric.iloc[valid])
        result.loc[valid, "clase_predicha"] = [CLASSES[c] for c in classes]
        result.loc[valid, "probabilidad_benigno"] = 1 - p
        result.loc[valid, "probabilidad_maligno"] = p
        out = range_warnings(numeric, train)
        if not out.empty:
            for row in set(out.fila - 1) - bad:
                result.at[row, "estado_validacion"] = "válida; fuera del rango de entrenamiento"
    if "malignant" in metadata:
        observed = pd.to_numeric(result.malignant, errors="coerce")
        result["estado_etiqueta"] = np.where(observed.isin([0, 1]), "válida", "etiqueta inválida o ausente; excluida de evaluación")
    return result, errors


def threshold_classes(probabilities, threshold=0.5):
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("El umbral debe estar entre 0 y 1.")
    return (np.asarray(probabilities) >= threshold).astype(int)


def classification_metrics(target, predicted, probabilities):
    tn, fp, fn, tp = confusion_matrix(target, predicted, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(target, predicted)),
        "sensitivity_recall": float(recall_score(target, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.,
        "precision": float(precision_score(target, predicted, zero_division=0)),
        "f1": float(f1_score(target, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probabilities)) if len(set(target)) == 2 else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]}
