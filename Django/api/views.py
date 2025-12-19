from __future__ import annotations

import io
import logging
import mimetypes
import sys
import threading
import traceback as tb
import warnings
from pathlib import Path

from django.conf import settings
from django.http import FileResponse, Http404
from django.shortcuts import render
from django.utils import timezone
from django.utils.dateparse import parse_datetime

from rest_framework import status, viewsets
from rest_framework.decorators import action, api_view
from rest_framework.response import Response

from api.models import Artifact, PipelineRun, StepLog
from api.serializers import (
    ArtifactSerializer,
    PipelineRunCreateSerializer,
    PipelineRunSerializer,
    StepLogSerializer,
)
from api.svc.runner import run_full_pipeline


# -----------------------------
# Artifacts helpers
# -----------------------------

def _artifacts_roots() -> list[Path]:
    """
    Busca artifacts en:
      1) Django/artifacts
      2) repo_root/artifacts (un nivel por encima de Django/)
    """
    base = Path(settings.BASE_DIR)
    return [
        (base / "artifacts").resolve(),
        (base.parent / "artifacts").resolve(),
    ]


def _artifacts_root() -> Path:
    return _artifacts_roots()[0]


def _safe_join_under(root: Path, rel: str) -> Path:
    rel = (rel or "").replace("\\", "/").lstrip("/")
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root)
    except Exception:
        raise Http404("Invalid path")
    return candidate


def _find_artifact_file(relpath: str) -> Path:
    """
    Dado un relpath tipo:
      - runs/1/figures/roc.png
      - reports/metrics.csv
      - artifacts/reports/metrics.csv
    intenta localizarlo en cualquiera de los roots válidos.
    """
    rel = (relpath or "").replace("\\", "/").lstrip("/")
    if rel.startswith("artifacts/"):
        rel = rel[len("artifacts/") :]

    for root in _artifacts_roots():
        p = _safe_join_under(root, rel)
        if p.exists() and p.is_file():
            return p

    raise Http404("Artifact not found")


# -----------------------------
# DB log sink (stdout/stderr)
# -----------------------------

class _DBLogStream(io.TextIOBase):
    """
    Stream estilo archivo para capturar stdout/stderr y persistirlo en StepLog.

    Nota: convertimos '\r' a '\n' para capturar progresos tipo tqdm.
    """
    def __init__(self, run_id: int, step: str = "console", level: str = StepLog.Level.INFO):
        super().__init__()
        self.run_id = run_id
        self.step = step
        self.level = level
        self._buf = ""

    def write(self, s: str) -> int:
        if not s:
            return 0

        # Captura progress bars / carriage returns
        s = s.replace("\r", "\n")

        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip()
            if line:
                StepLog.objects.create(
                    run_id=self.run_id,
                    step=self.step,
                    level=self.level,
                    message=line,
                )
        return len(s)

    def flush(self) -> None:
        if self._buf.strip():
            StepLog.objects.create(
                run_id=self.run_id,
                step=self.step,
                level=self.level,
                message=self._buf.strip(),
            )
        self._buf = ""


class _DBLogHandler(logging.Handler):
    """
    Handler de logging que vuelca a nuestros streams persistentes.
    (INFO y por debajo -> stdout; WARNING+ -> stderr)
    """
    def __init__(self, out_stream: _DBLogStream, err_stream: _DBLogStream):
        super().__init__(level=logging.INFO)
        self.out_stream = out_stream
        self.err_stream = err_stream
        self.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            stream = self.err_stream if record.levelno >= logging.WARNING else self.out_stream
            stream.write(msg + "\n")
        except Exception:
            # Nunca rompas el pipeline por logging
            pass


# -----------------------------
# Pipeline async runner
# -----------------------------

def _run_pipeline_async(run_id: int) -> None:
    run = PipelineRun.objects.get(pk=run_id)

    run.status = PipelineRun.Status.RUNNING
    run.started_at = timezone.now()
    run.finished_at = None
    run.error_message = None
    run.traceback = None
    run.save(update_fields=["status", "started_at", "finished_at", "error_message", "traceback"])

    out_stream = _DBLogStream(run_id=run.id, step="stdout", level=StepLog.Level.INFO)
    err_stream = _DBLogStream(run_id=run.id, step="stderr", level=StepLog.Level.ERROR)

    # Redirección stdout/stderr (print, etc.)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out_stream, err_stream

    # Captura de logging (librerías incluidas)
    root_logger = logging.getLogger()
    old_root_level = root_logger.level
    old_root_handlers = list(root_logger.handlers)
    old_propagate = root_logger.propagate

    db_handler = _DBLogHandler(out_stream=out_stream, err_stream=err_stream)

    # Captura warnings vía logging (py.warnings)
    old_capture_warnings = logging.captureWarnings(True)
    old_warning_filters = warnings.filters[:]

    try:
        # Asegura que warnings se vean (no solo "default" silenciado por librerías)
        warnings.simplefilter("default")

        # Configura root logger: INFO y solo nuestro handler (evita duplicados a consola)
        root_logger.handlers = []
        root_logger.addHandler(db_handler)
        root_logger.setLevel(logging.INFO)
        root_logger.propagate = False

        # Algunos loggers ruidosos: los dejamos en INFO, pero si te molesta alguno, aquí se ajusta.
        # Ejemplos:
        # logging.getLogger("matplotlib").setLevel(logging.INFO)
        # logging.getLogger("tensorflow").setLevel(logging.INFO)

        StepLog.objects.create(run=run, step="system", level=StepLog.Level.INFO, message="Pipeline started")

        run_full_pipeline(run)

        StepLog.objects.create(run=run, step="system", level=StepLog.Level.INFO, message="Pipeline finished")

        run.finished_at = timezone.now()
        if run.status == PipelineRun.Status.RUNNING:
            run.status = PipelineRun.Status.SELECTED
        run.save(update_fields=["status", "finished_at"])

    except Exception as e:
        run.status = PipelineRun.Status.FAILED
        run.error_message = str(e)
        run.traceback = tb.format_exc()
        run.finished_at = timezone.now()
        run.save(update_fields=["status", "error_message", "traceback", "finished_at"])
        StepLog.objects.create(run=run, step="system", level=StepLog.Level.ERROR, message=f"FAILED: {e}")

    finally:
        # Restaura warnings
        try:
            warnings.filters[:] = old_warning_filters
        except Exception:
            pass

        # Restaura logging
        try:
            logging.captureWarnings(old_capture_warnings)
        except Exception:
            pass

        try:
            root_logger.handlers = old_root_handlers
            root_logger.setLevel(old_root_level)
            root_logger.propagate = old_propagate
        except Exception:
            pass

        # Flush streams
        try:
            out_stream.flush()
            err_stream.flush()
        except Exception:
            pass

        # Restaura stdout/stderr
        sys.stdout, sys.stderr = old_stdout, old_stderr


# -----------------------------
# ViewSet
# -----------------------------

class PipelineRunViewSet(viewsets.ModelViewSet):
    queryset = PipelineRun.objects.all().order_by("-created_at")

    def get_serializer_class(self):
        if self.action == "create":
            return PipelineRunCreateSerializer
        return PipelineRunSerializer

    def perform_create(self, serializer):
        run = serializer.save(status=PipelineRun.Status.CREATED)

        # Normaliza por si alguien mete "data/raw/x.csv"
        run.dataset = Path((run.dataset or "").replace("\\", "/")).name

        cfg = run.config or {}
        if not isinstance(cfg, dict):
            cfg = {}

        # Garantiza raw_filename siempre
        cfg.setdefault("raw_filename", run.dataset)
        cfg.setdefault("dataset", run.dataset)
        run.config = cfg

        # Directorio de artifacts por run
        base = _artifacts_root() / "runs" / str(run.id)
        base.mkdir(parents=True, exist_ok=True)

        run.artifacts_path = str(base)
        run.save(update_fields=["dataset", "config", "artifacts_path"])

    @action(detail=True, methods=["post"])
    def start(self, request, pk=None):
        run = self.get_object()

        if run.status not in [PipelineRun.Status.CREATED, PipelineRun.Status.FAILED]:
            return Response({"error": "Pipeline ya ejecutado o en ejecución"}, status=status.HTTP_400_BAD_REQUEST)

        t = threading.Thread(target=_run_pipeline_async, args=(run.id,), daemon=True)
        t.start()

        return Response({"status": "Pipeline started", "run_id": run.id}, status=status.HTTP_202_ACCEPTED)

    @action(detail=True, methods=["get"])
    def logs(self, request, pk=None):
        run = self.get_object()
        since = request.query_params.get("since")

        qs = StepLog.objects.filter(run=run).order_by("created_at", "id")

        if since:
            dt = parse_datetime(since)
            if dt is not None:
                if timezone.is_naive(dt):
                    dt = timezone.make_aware(dt, timezone=timezone.utc)
                qs = qs.filter(created_at__gt=dt)

        serializer = StepLogSerializer(qs, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=["get"])
    def artifacts(self, request, pk=None):
        run = self.get_object()
        qs = Artifact.objects.filter(run=run).order_by("created_at", "id")
        serializer = ArtifactSerializer(qs, many=True, context={"request": request})
        return Response(serializer.data)


# -----------------------------
# Datasets (filesystem)
# -----------------------------

@api_view(["GET"])
def list_datasets(request):
    """
    Lista ../data/raw/*.csv (data está al mismo nivel que Django y src).
    IMPORTANTE: devuelve SOLO nombres ("x.csv"), no "data/raw/x.csv".
    """
    base = Path(settings.BASE_DIR).parent / "data" / "raw"
    if not base.exists():
        return Response([])
    files = sorted(p.name for p in base.glob("*.csv") if p.is_file())
    return Response(files)


# -----------------------------
# Artifacts serving
# -----------------------------

@api_view(["GET"])
def download_artifact(request, artifact_id: int):
    art = Artifact.objects.get(pk=artifact_id)

    raw = (art.path or "").replace("\\", "/")

    pp = Path(raw)
    if pp.is_absolute() and pp.exists() and pp.is_file():
        p = pp
    else:
        rel = raw
        if rel.startswith("artifacts/"):
            rel = rel[len("artifacts/") :]
        p = _find_artifact_file(rel)

    ctype, _ = mimetypes.guess_type(str(p))
    return FileResponse(p.open("rb"), content_type=ctype or "application/octet-stream")


def serve_artifact(request, relpath: str):
    p = _find_artifact_file(relpath)
    ctype, _ = mimetypes.guess_type(str(p))
    return FileResponse(p.open("rb"), content_type=ctype or "application/octet-stream")


# -----------------------------
# Renders (HTML)
# -----------------------------

def pipeline_dashboard(request):
    return render(request, "pipeline/dashboard.html")


def pipeline_run_detail(request, run_id):
    return render(request, "pipeline/run_detail.html", {"run_id": run_id})
