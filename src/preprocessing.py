# preprocesar datos para ingesta

# src/preprocessing.py
from __future__ import annotations

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# Columnas que NO deben usarse como features porque "filtran" el resultado final
LEAKAGE_COLS = ["reservation_status", "reservation_status_date"]


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variables derivadas útiles y normaliza algunas columnas.
    No toca la variable objetivo.
    """
    df = df.copy()

    # Total de noches
    if "stays_in_weekend_nights" in df.columns and "stays_in_week_nights" in df.columns:
        df["total_nights"] = df["stays_in_weekend_nights"].fillna(0) + df["stays_in_week_nights"].fillna(0)

    # Total de huéspedes (ojo: children puede venir NaN)
    for col in ["adults", "children", "babies"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    if set(["adults", "children", "babies"]).issubset(df.columns):
        df["total_guests"] = df["adults"] + df["children"] + df["babies"]

    # Fecha de llegada (si existen columnas)
    # arrival_date_month suele venir como string ("January"...)
    if set(["arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"]).issubset(df.columns):
        # Convertimos month a número si viene en inglés
        month_map = {
            "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
            "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
        }
        if df["arrival_date_month"].dtype == "object":
            df["arrival_month_num"] = df["arrival_date_month"].map(month_map)
        else:
            df["arrival_month_num"] = df["arrival_date_month"]

        # Construir datetime (si algo falla, quedará NaT)
        df["arrival_date"] = pd.to_datetime(
            dict(
                year=df["arrival_date_year"],
                month=df["arrival_month_num"],
                day=df["arrival_date_day_of_month"],
            ),
            errors="coerce",
        )

        # Features temporales extra
        df["arrival_dayofweek"] = df["arrival_date"].dt.dayofweek
        df["arrival_is_weekend"] = df["arrival_dayofweek"].isin([5, 6]).astype(int)

    return df


def drop_leakage_and_unused(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Elimina columnas con fuga de información (leakage) y la target si aparece en X.
    """
    df = df.copy()

    # Drop leakage
    for col in LEAKAGE_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Por si acaso target está dentro
    if target_col in df.columns:
        df = df.drop(columns=[target_col])

    return df


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """
    Construye un ColumnTransformer para:
    - num: imputación mediana + escalado
    - cat: imputación moda + one-hot
    """
    # Detectar tipos
    numeric_features = X_train.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    # Pipelines
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def prepare_X(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Conveniencia: aplica feature engineering + elimina leakage + devuelve X listo
    (todavía sin transformar, para que el preprocessor lo procese en pipeline).
    """
    df_fe = add_feature_engineering(df)
    X = drop_leakage_and_unused(df_fe, target_col=target_col)
    return X


#  FORMA DE USO 
# from src.preprocessing import prepare_X, build_preprocessor
# from src.config import TARGET_COL

# df = load_data()

# X = prepare_X(df, target_col=TARGET_COL)
# y = df[TARGET_COL]

# X_train, X_test, y_train, y_test = split_data_xy(X, y)

# preprocessor = build_preprocessor(X_train)
# --------------------------------
# from sklearn.pipeline import Pipeline

# pipe = Pipeline(steps=[
#     ("preprocess", preprocessor),
#     ("model", model)
# ])

# pipe.fit(X_train, y_train)
