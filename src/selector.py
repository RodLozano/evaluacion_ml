# elegir el mejor modelo

# src/selector.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib


@dataclass
class SelectorConfig:
    artifacts_dir: str = "artifacts"
    models_dirname: str = "models"
    best_model_filename: str = "best_model.joblib"
    best_meta_filename: str = "best_model.json"


def _ensure_models_dir(cfg: SelectorConfig) -> Path:
    models_dir = Path(cfg.artifacts_dir) / cfg.models_dirname
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def _assert_metric_exists(results: List[Dict[str, Any]], metric: str) -> None:
    missing = [r.get("model", "<unknown>") for r in results if metric not in r]
    if missing:
        raise KeyError(
            f"La métrica '{metric}' no existe en los resultados para: {missing}. "
            f"Métricas disponibles ejemplo: {list(results[0].keys()) if results else '[]'}"
        )


def _sort_key(
    r: Dict[str, Any],
    primary: str,
    tie_breakers: Sequence[str],
) -> Tuple:
    """
    Devuelve una tupla para ordenar resultados:
    (primary, tie1, tie2, ...)
    Ordenaremos descendente, así que esta tupla serán valores numéricos.
    """
    key = [r.get(primary, float("-inf"))]
    for m in tie_breakers:
        key.append(r.get(m, float("-inf")))
    return tuple(key)


def select_best_model(
    results: List[Dict[str, Any]],
    metric: str = "roc_auc",
    tie_breakers: Optional[Sequence[str]] = ("f1", "recall", "precision", "accuracy"),
) -> Dict[str, Any]:
    """
    Selecciona el mejor resultado según:
    - métrica principal (metric)
    - y, si hay empate, métricas secundarias (tie_breakers)

    Devuelve el dict completo del mejor (incluyendo _pipeline si existe).
    """
    if not results:
        raise ValueError("results está vacío. No hay modelos para seleccionar.")

    _assert_metric_exists(results, metric)

    tie_breakers = tuple(tie_breakers) if tie_breakers is not None else tuple()

    # Orden descendente por la tupla (primary, ties...)
    best = sorted(
        results,
        key=lambda r: _sort_key(r, metric, tie_breakers),
        reverse=True,
    )[0]

    return best


def save_best_model(
    best_result: Dict[str, Any],
    cfg: Optional[SelectorConfig] = None,
) -> Tuple[Path, Path]:
    """
    Guarda el pipeline ganador (si viene) y un JSON con resumen del mejor modelo.

    Espera que evaluator.py haya añadido:
    - best_result["_pipeline"] : sklearn Pipeline entrenado
    """
    cfg = cfg or SelectorConfig()
    models_dir = _ensure_models_dir(cfg)

    model_path = models_dir / cfg.best_model_filename
    meta_path = models_dir / cfg.best_meta_filename

    # Guardar pipeline (sklearn)
    pipe = best_result.get("_pipeline", None)
    if pipe is None:
        raise KeyError(
            "No encuentro '_pipeline' en best_result. "
            "Asegúrate de que evaluate_model guarda el pipeline en metrics['_pipeline']."
        )

    joblib.dump(pipe, model_path)

    # Guardar metadata limpia (sin arrays ni pipeline)
    clean = {}
    for k, v in best_result.items():
        if k.startswith("_"):
            continue
        # Convertibles a JSON
        if isinstance(v, (int, float, str, bool)) or v is None:
            clean[k] = v
        else:
            # fallback: convertir a string (por ejemplo, paths)
            clean[k] = str(v)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)

    return model_path, meta_path


def load_best_model(
    cfg: Optional[SelectorConfig] = None,
):
    """
    Carga el pipeline ganador desde artifacts/models/best_model.joblib
    """
    cfg = cfg or SelectorConfig()
    models_dir = _ensure_models_dir(cfg)
    model_path = models_dir / cfg.best_model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"No existe el modelo guardado: {model_path}")
    return joblib.load(model_path)


# FORMA DE USO
# from src.selector import select_best_model, save_best_model

# best = select_best_model(results, metric="roc_auc")  # o "f1"
# model_path, meta_path = save_best_model(best)

# print("Best model:", best["model"])
# print("Saved to:", model_path)
# print("Meta:", meta_path)
