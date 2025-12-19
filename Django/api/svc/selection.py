from __future__ import annotations

from api.models import PipelineRun, Artifact
from api.svc.base import capture_stdout
from selector import select_best_model, save_best_model


def _json_safe(obj):
    """
    Convierte recursivamente objetos no serializables (numpy, Path, etc.)
    a tipos JSON-safe: dict/list/str/int/float/bool/None.
    """
    # Lazy import para no depender de numpy si no está instalado en runtime
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None  # type: ignore

    # None / primitives
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # dict
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}

    # list/tuple/set
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]

    # numpy scalars / arrays
    if np is not None:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # numpy scalar types (np.float64, np.int64, etc.)
        if isinstance(obj, np.generic):
            return obj.item()

    # pathlib.Path
    try:
        from pathlib import Path
        if isinstance(obj, Path):
            return str(obj)
    except Exception:
        pass

    # fallback: string
    return str(obj)


def run_selection_step(run: PipelineRun, results):
    # Nota: idealmente aquí pondrías un estado intermedio SELECTION si lo tienes.
    run.status = PipelineRun.Status.SELECTED
    run.save(update_fields=["status"])

    sklearn_results = [r for r in results if "_pipeline" in r]

    with capture_stdout(run, step="SELECTION"):
        best = select_best_model(
            sklearn_results,
            metric=(run.config or {}).get("metric_main", "roc_auc"),
        )
        model_path, meta_path = save_best_model(best)

    # IMPORTANT: JSONField necesita tipos serializables
    run.metrics = _json_safe(best)
    run.save(update_fields=["metrics"])

    Artifact.objects.create(
        run=run,
        name="best_model.joblib",
        kind=Artifact.Kind.MODEL,
        path=str(model_path),
    )

    # Opcional: guarda también meta si existe
    if meta_path:
        Artifact.objects.create(
            run=run,
            name="best_model_meta.json",
            kind=Artifact.Kind.REPORT,
            path=str(meta_path),
        )
