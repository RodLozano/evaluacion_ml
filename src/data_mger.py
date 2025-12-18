# cargar y partir datos

# src/data_mger.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

# Si tienes src/config.py, intentamos usarlo; si no, ponemos defaults seguros.
try:
    from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR, TARGET_COL, SEED, TEST_SIZE  # type: ignore
except Exception:
    DATA_RAW_DIR = "data/raw"
    DATA_PROCESSED_DIR = "data/processed"
    TARGET_COL = "is_canceled"
    SEED = 42
    TEST_SIZE = 0.2


# Columnas con fuga de información típicas del dataset hotel bookings
LEAKAGE_COLS = ["reservation_status", "reservation_status_date"]


@dataclass(frozen=True)
class DataPaths:
    raw_dir: Path = Path(DATA_RAW_DIR)
    processed_dir: Path = Path(DATA_PROCESSED_DIR)

    def raw_file(self, filename: str) -> Path:
        return self.raw_dir / filename

    def processed_file(self, filename: str) -> Path:
        return self.processed_dir / filename


def ensure_dirs(paths: DataPaths) -> None:
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    paths.processed_dir.mkdir(parents=True, exist_ok=True)


def load_raw_csv(filename: str, paths: Optional[DataPaths] = None) -> pd.DataFrame:
    """
    Carga el dataset desde data/raw.
    """
    paths = paths or DataPaths()
    ensure_dirs(paths)

    fpath = paths.raw_file(filename)
    if not fpath.exists():
        raise FileNotFoundError(
            f"No encuentro el archivo raw en: {fpath}. "
            f"Colócalo en {paths.raw_dir} o ajusta la ruta."
        )

    df = pd.read_csv(fpath)
    return df


def split_X_y(df: pd.DataFrame, target_col: str = TARGET_COL) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separa features y target.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target '{target_col}' no está en el DataFrame. Columnas: {list(df.columns)}")

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])
    return X, y


def make_splits(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = TEST_SIZE,
    val_size: Optional[float] = None,
    seed: int = SEED,
    stratify: bool = True,
) -> dict:
    """
    Devuelve splits en un dict:
    - X_train, X_test, y_train, y_test
    - (opcional) X_val, y_val

    stratify=True es lo habitual para clasificación binaria.
    """
    X, y = split_X_y(df, target_col=target_col)

    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=strat
    )

    out = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }

    if val_size is not None and val_size > 0:
        # val_size es relativo a train (p.ej 0.2 => 20% del train para validación)
        strat2 = y_train if stratify else None
        X_train2, X_val, y_train2, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=seed, stratify=strat2
        )
        out.update({"X_train": X_train2, "X_val": X_val, "y_train": y_train2, "y_val": y_val})

    return out


def save_splits(splits: dict, paths: Optional[DataPaths] = None, prefix: str = "") -> None:
    """
    Guarda splits como CSV en data/processed para reproducibilidad.
    """
    paths = paths or DataPaths()
    ensure_dirs(paths)

    def p(name: str) -> Path:
        fname = f"{prefix}{name}.csv" if prefix else f"{name}.csv"
        return paths.processed_file(fname)

    # X
    splits["X_train"].to_csv(p("X_train"), index=False)
    splits["X_test"].to_csv(p("X_test"), index=False)
    # y
    splits["y_train"].to_frame("y").to_csv(p("y_train"), index=False)
    splits["y_test"].to_frame("y").to_csv(p("y_test"), index=False)

    if "X_val" in splits:
        splits["X_val"].to_csv(p("X_val"), index=False)
    if "y_val" in splits:
        splits["y_val"].to_frame("y").to_csv(p("y_val"), index=False)


def load_splits(paths: Optional[DataPaths] = None, prefix: str = "") -> dict:
    """
    Carga splits guardados en data/processed.
    """
    paths = paths or DataPaths()

    def p(name: str) -> Path:
        fname = f"{prefix}{name}.csv" if prefix else f"{name}.csv"
        return paths.processed_file(fname)

    out = {
        "X_train": pd.read_csv(p("X_train")),
        "X_test": pd.read_csv(p("X_test")),
        "y_train": pd.read_csv(p("y_train"))["y"],
        "y_test": pd.read_csv(p("y_test"))["y"],
    }

    # opcional
    xval = p("X_val")
    yval = p("y_val")
    if xval.exists() and yval.exists():
        out["X_val"] = pd.read_csv(xval)
        out["y_val"] = pd.read_csv(yval)["y"]

    return out


def get_data(
    raw_filename: str,
    target_col: str = TARGET_COL,
    save_processed: bool = True,
    prefix: str = "",
    paths: Optional[DataPaths] = None,
    test_size: float = TEST_SIZE,
    val_size: Optional[float] = None,
    seed: int = SEED,
) -> dict:
    """
    Orquestador de datos:
    - carga raw
    - crea splits
    - (opcional) guarda en processed
    """
    df = load_raw_csv(raw_filename, paths=paths)
    splits = make_splits(
        df=df,
        target_col=target_col,
        test_size=test_size,
        val_size=val_size,
        seed=seed,
        stratify=True,
    )
    if save_processed:
        save_splits(splits, paths=paths, prefix=prefix)
    return splits

def normalize_results(
    results: list[dict],
    y_true: pd.Series | np.ndarray,
) -> list[dict]:
    """Normaliza la salida de `results` para incluir modelos no-sklearn en plots.

    Objetivo: que `plot_roc_comparison()` pueda pintar TODOS los modelos.

    - No muta `results` (devuelve una lista nueva).
    - Si un resultado ya trae `_fpr/_tpr`, se deja igual.
    - Si trae `y_proba`, calcula `_fpr/_tpr/_auc` con `y_true`.
    """
    y_true_np = np.asarray(y_true).astype(int)

    normalized: list[dict] = []
    for r in results:
        rr = dict(r)  # copia superficial

        # Caso sklearn: ya viene listo para ROC
        if "_fpr" in rr and "_tpr" in rr:
            normalized.append(rr)
            continue

        # Caso Keras (u otros): tenemos probas y generamos ROC
        if rr.get("y_proba", None) is not None:
            y_proba = np.asarray(rr["y_proba"]).reshape(-1)
            fpr, tpr, _ = roc_curve(y_true_np, y_proba)
            auc = roc_auc_score(y_true_np, y_proba)

            rr["_fpr"] = fpr
            rr["_tpr"] = tpr
            rr["_auc"] = float(auc)

        normalized.append(rr)

    return normalized
