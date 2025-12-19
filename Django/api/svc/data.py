from __future__ import annotations

from pathlib import Path

from api.models import PipelineRun
from api.svc.base import capture_stdout

# IMPORTANTE:
# - como metes /mlpontia/src en sys.path, aquí se importa SIN prefijo "src."
from data_mger import get_data


def run_data_step(run: PipelineRun):
    run.status = PipelineRun.Status.DATA
    run.save(update_fields=["status"])

    cfg = run.config or {}
    if not isinstance(cfg, dict):
        cfg = {}

    raw = cfg.get("raw_filename") or cfg.get("dataset") or run.dataset
    raw = (raw or "").strip()
    raw_filename = Path(raw.replace("\\", "/")).name if raw else ""

    if not raw_filename:
        # Esto deja un mensaje claro en UI (FAILED: 'raw_filename')
        raise KeyError("raw_filename")

    with capture_stdout(run, step="DATA"):
        splits = get_data(
            raw_filename=raw_filename,               # <-- SIEMPRE solo "x.csv"
            seed=cfg.get("seed", 42),
            test_size=cfg.get("test_size", 0.2),
            val_size=cfg.get("val_size"),
            save_processed=True,
        )

    return splits
