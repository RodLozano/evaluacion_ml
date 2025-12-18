# evaluar modelos

# src/evaluator.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)


@dataclass
class EvalConfig:
    artifacts_dir: str = "artifacts"
    reports_dirname: str = "reports"
    figures_dirname: str = "figures"
    models_dirname: str = "models"
    threshold: float = 0.5  # para convertir probas a clase


def _ensure_dirs(cfg: EvalConfig) -> Tuple[Path, Path, Path]:
    artifacts = Path(cfg.artifacts_dir)
    reports = artifacts / cfg.reports_dirname
    figures = artifacts / cfg.figures_dirname
    models = artifacts / cfg.models_dirname

    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    return reports, figures, models


def _safe_predict_proba(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Devuelve probas de clase 1 si el modelo lo soporta.
    Si no, intenta decision_function y lo reescala a (0,1) de forma simple.
    """
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)[:, 1]
        return np.asarray(proba)

    # Fallback: decision_function -> sigmoid-ish
    if hasattr(pipe, "decision_function"):
        scores = np.asarray(pipe.decision_function(X))
        # normalización suave a [0,1]
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
        return scores

    raise AttributeError("El modelo no soporta predict_proba ni decision_function.")


def plot_and_save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])

    plt.figure()
    disp.plot(values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_and_save_roc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    out_path: Path,
    title: str,
) -> Tuple[np.ndarray, np.ndarray, float]:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return fpr, tpr, float(auc)


def evaluate_model(
    name: str,
    model: Any,
    preprocessor: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cfg: Optional[EvalConfig] = None,
    fit: bool = True,
) -> Dict[str, Any]:
    """
    Entrena (opcional) y evalúa un modelo sklearn usando Pipeline(preprocess + model).
    Guarda matriz de confusión y ROC en artifacts/figures.

    Devuelve un dict con métricas + info para ROC comparativa.
    """
    cfg = cfg or EvalConfig()
    reports_dir, figures_dir, _ = _ensure_dirs(cfg)

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])

    if fit:
        pipe.fit(X_train, y_train)

    y_proba = _safe_predict_proba(pipe, X_test)
    y_pred = (y_proba >= cfg.threshold).astype(int)

    y_true = np.asarray(y_test).astype(int)

    metrics: Dict[str, Any] = {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "threshold": float(cfg.threshold),
    }

    # Plots por modelo
    cm_path = figures_dir / f"cm_{name}.png"
    roc_path = figures_dir / f"roc_{name}.png"

    plot_and_save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        out_path=cm_path,
        title=f"Confusion Matrix — {name}",
    )

    fpr, tpr, auc = plot_and_save_roc(
        y_true=y_true,
        y_proba=y_proba,
        out_path=roc_path,
        title=f"ROC Curve — {name}",
    )

    # Guardamos para ROC comparativa
    metrics["_fpr"] = fpr
    metrics["_tpr"] = tpr
    metrics["_auc"] = auc
    metrics["_pipeline"] = pipe  # por si luego quieres guardar el mejor

    return metrics


def save_reports(
    results: List[Dict[str, Any]],
    cfg: Optional[EvalConfig] = None,
    filename: str = "metrics.csv",
) -> Path:
    """
    Guarda tabla de métricas como CSV (sin arrays internos).
    """
    cfg = cfg or EvalConfig()
    reports_dir, _, _ = _ensure_dirs(cfg)

    # limpiamos claves internas + payloads grandes (p.ej. y_proba/y_pred de Keras)
    cleaned = []
    drop_keys = {"y_proba", "y_pred"}
    for r in results:
        rr = {k: v for k, v in r.items() if (not k.startswith("_")) and (k not in drop_keys)}
        cleaned.append(rr)

    df = pd.DataFrame(cleaned).sort_values(by="roc_auc", ascending=False)
    out_path = reports_dir / filename
    df.to_csv(out_path, index=False)
    return out_path


def plot_roc_comparison(
    results: List[Dict[str, Any]],
    cfg: Optional[EvalConfig] = None,
    filename: str = "roc_compare.png",
    title: str = "ROC Comparison",
) -> Path:
    """
    Plotea una ROC comparativa para todos los modelos evaluados.
    """
    cfg = cfg or EvalConfig()
    _, figures_dir, _ = _ensure_dirs(cfg)

    plt.figure()
    for r in results:
        if "_fpr" not in r or "_tpr" not in r:
            continue
        name = r.get("model", "model")
        auc = r.get("_auc", None)
        label = f"{name}" + (f" (AUC={auc:.4f})" if isinstance(auc, (int, float)) else "")
        plt.plot(r["_fpr"], r["_tpr"], label=label)

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()

    out_path = figures_dir / filename
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path

# FORMA DE USO
# results = []

# for name, model in get_sklearn_models(seed=SEED).items():
#     r = evaluate_model(
#         name=name,
#         model=model,
#         preprocessor=preprocessor,
#         X_train=X_train, y_train=y_train,
#         X_test=X_test, y_test=y_test,
#     )
#     results.append(r)

# save_reports(results)
# plot_roc_comparison(results)
