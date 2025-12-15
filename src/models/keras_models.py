#entrenamiento de modelos de keras

# src/models/keras_model.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@dataclass
class KerasTrainConfig:
    seed: int = 42
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 1e-3
    patience: int = 5
    validation_split: float = 0.15
    model_dir: str = "artifacts/models"
    model_name: str = "keras_model.keras"
    threshold: float = 0.5  # para convertir probas a clase


def set_tf_seed(seed: int) -> None:
    # Reproducibilidad razonable (no perfecta al 100% en GPU)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def build_mlp(input_dim: int, cfg: KerasTrainConfig) -> keras.Model:
    """
    MLP simple pero sólida para tabular + one-hot.
    """
    inputs = keras.Input(shape=(input_dim,), name="features")

    x = layers.Dense(256)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(1, activation="sigmoid", name="prob")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="mlp_binary_classifier")

    opt = keras.optimizers.Adam(learning_rate=cfg.learning_rate)

    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def _to_numpy_matrix(X) -> np.ndarray:
    """
    ColumnTransformer puede devolver np.ndarray o sparse.
    Keras necesita matriz densa (np.ndarray).
    """
    # Si es sparse, convertir a dense (cuidado con memoria si one-hot enorme)
    if hasattr(X, "toarray"):
        return X.toarray().astype(np.float32)
    return np.asarray(X, dtype=np.float32)


def train_keras_model(
    preprocessor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cfg: Optional[KerasTrainConfig] = None,
    fit_preprocessor: bool = True,
) -> Dict[str, Any]:
    """
    Entrena una red neuronal multicapa (Keras) para clasificación binaria.

    Parámetros:
    - preprocessor: ColumnTransformer de sklearn (num+cat)
    - fit_preprocessor: True si quieres fit_transform en train (lo normal).
      Si ya lo has fiteado antes, pon False.

    Devuelve un dict con métricas, ruta del modelo y probabilidades.
    """
    cfg = cfg or KerasTrainConfig()
    set_tf_seed(cfg.seed)

    # Preprocess -> matrices para Keras
    if fit_preprocessor:
        Xtr = preprocessor.fit_transform(X_train)
    else:
        Xtr = preprocessor.transform(X_train)

    Xte = preprocessor.transform(X_test)

    Xtr = _to_numpy_matrix(Xtr)
    Xte = _to_numpy_matrix(Xte)

    ytr = np.asarray(y_train).astype(np.float32)
    yte = np.asarray(y_test).astype(np.float32)

    model = build_mlp(input_dim=Xtr.shape[1], cfg=cfg)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=cfg.patience,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        Xtr,
        ytr,
        validation_split=cfg.validation_split,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    # Predicción (probas + clases)
    y_proba = model.predict(Xte, batch_size=cfg.batch_size, verbose=0).reshape(-1)
    y_pred = (y_proba >= cfg.threshold).astype(int)

    # Métricas
    metrics = {
        "model": "keras_mlp",
        "accuracy": float(accuracy_score(yte, y_pred)),
        "precision": float(precision_score(yte, y_pred, zero_division=0)),
        "recall": float(recall_score(yte, y_pred, zero_division=0)),
        "f1": float(f1_score(yte, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(yte, y_proba)),
        "threshold": cfg.threshold,
        "epochs_trained": len(history.history.get("loss", [])),
    }

    # Guardar modelo
    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / cfg.model_name
    model.save(model_path)

    metrics["model_path"] = str(model_path)

    # Devolvemos también probas por si evaluator quiere ROC
    metrics["y_proba"] = y_proba
    metrics["y_pred"] = y_pred

    return metrics


def load_keras_model(model_path: str | Path) -> keras.Model:
    """
    Carga el modelo Keras guardado.
    """
    return keras.models.load_model(model_path)


#  FORMA DE USO 
# from src.models.keras_model import train_keras_model, KerasTrainConfig

# keras_metrics = train_keras_model(
#     preprocessor=preprocessor,
#     X_train=X_train, y_train=y_train,
#     X_test=X_test, y_test=y_test,
#     cfg=KerasTrainConfig(epochs=40, patience=6, batch_size=256)
# )
# results.append(keras_metrics)
