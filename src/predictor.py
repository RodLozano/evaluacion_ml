# utilizar mejor modelo para predecir


# src/predict.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.selector import load_best_model

# Si tienes config, lo usamos; si no, defaults
try:
    from src.config import TARGET_COL  # type: ignore
except Exception:
    TARGET_COL = "is_canceled"

LEAKAGE_COLS = ["reservation_status", "reservation_status_date"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inferencia con el mejor modelo (Pipeline sklearn).")
    parser.add_argument("--input", "-i", required=True, help="Ruta al CSV de entrada con features.")
    parser.add_argument("--output", "-o", required=True, help="Ruta al CSV de salida con predicciones.")
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.5,
        help="Umbral para convertir probas a clase (default: 0.5).",
    )
    parser.add_argument(
        "--keep-target",
        action="store_true",
        help="Si el CSV incluye la columna target, no la borres (por defecto se elimina).",
    )
    return parser.parse_args()


def clean_input_df(df: pd.DataFrame, keep_target: bool = False) -> pd.DataFrame:
    """
    Limpia columnas que no deben entrar al modelo:
    - leakage
    - target (si viene incluida)
    """
    df = df.copy()

    for col in LEAKAGE_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])

    if (not keep_target) and (TARGET_COL in df.columns):
        df = df.drop(columns=[TARGET_COL])

    return df


def predict(
    input_path: str | Path,
    output_path: str | Path,
    threshold: float = 0.5,
    keep_target: bool = False,
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {input_path}")

    df = pd.read_csv(input_path)
    X = clean_input_df(df, keep_target=keep_target)

    pipe = load_best_model()

    # Probabilidades (clase 1)
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)[:, 1]
    elif hasattr(pipe, "decision_function"):
        scores = pipe.decision_function(X)
        # normalizar a [0,1] para poder usar umbral (fallback)
        scores = np.asarray(scores)
        proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    else:
        raise AttributeError("El modelo no soporta predict_proba ni decision_function.")

    pred = (proba >= threshold).astype(int)

    out = df.copy()
    out["proba"] = proba
    out["pred"] = pred

    out.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    args = parse_args()
    out_path = predict(
        input_path=args.input,
        output_path=args.output,
        threshold=args.threshold,
        keep_target=args.keep_target,
    )
    print(f"✅ Predicciones guardadas en: {out_path}")


if __name__ == "__main__":
    main()


# FORMAS DE USO

# SIN TARGET EN DATASET
# python -m src.predict \
#   --input data/processed/new_samples.csv \
#   --output artifacts/reports/predictions.csv \
#   --threshold 0.5

# CON TARGET EN DATASET
# python -m src.predict \
#   -i data/processed/test_with_target.csv \
#   -o artifacts/reports/predictions.csv \
#   --keep-target

# Este predictor.py funciona solo si ya guardaste el modelo con save_best_model() en:
