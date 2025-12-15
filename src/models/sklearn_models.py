# src/models/sklearn_models.py
from __future__ import annotations

from typing import Dict, Any, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier


def get_sklearn_models(
    seed: int = 42,
    class_weight: Optional[str] = "balanced",
) -> Dict[str, Any]:
    """
    Devuelve un diccionario de modelos sklearn (sin pipelines)
    para clasificación binaria.

    Modelos incluidos:
    - Regresión Logística
    - Árbol de Decisión
    - Random Forest
    - Gradient Boosting (HistGradientBoostingClassifier)

    Parámetros:
    - seed: reproducibilidad
    - class_weight: balanceo de clases para modelos que lo soportan
      ("balanced" o None)
    """
    models: Dict[str, Any] = {}

    # 1) Logistic Regression
    # 'saga' funciona bien con one-hot y datasets grandes
    models["logreg"] = LogisticRegression(
        solver="saga",
        penalty="l2",
        C=1.0,
        max_iter=500,
        n_jobs=-1,
        class_weight=class_weight,
        random_state=seed,
    )

    # 2) Decision Tree
    models["decision_tree"] = DecisionTreeClassifier(
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight=class_weight,
        random_state=seed,
    )

    # 3) Random Forest
    models["random_forest"] = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        class_weight=class_weight,
        random_state=seed,
    )

    # 4) Gradient Boosting (100% sklearn)
    models["hist_gradient_boosting"] = HistGradientBoostingClassifier(
        max_depth=None,
        learning_rate=0.08,
        max_iter=400,
        random_state=seed,
    )

    return models


def get_model_names(models: Dict[str, Any]) -> list[str]:
    """
    Devuelve la lista de nombres de los modelos.
    """
    return list(models.keys())
