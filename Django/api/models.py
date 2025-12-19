from __future__ import annotations

from django.core.serializers.json import DjangoJSONEncoder
from django.db import models
from django.utils import timezone


class NumpyJSONEncoder(DjangoJSONEncoder):
    """
    Permite guardar en JSONField objetos típicos de numpy (ndarray, np.int64, np.float32, etc.)
    sin tocar la lógica del pipeline (src/).
    """

    def default(self, obj):
        try:
            import numpy as np
        except Exception:
            np = None

        if np is not None:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)

        return super().default(obj)


class PipelineRun(models.Model):
    class Status(models.TextChoices):
        CREATED = "CREATED", "Created"
        RUNNING = "RUNNING", "Running"
        DATA = "DATA", "Data loaded"
        PREPROCESS = "PREPROCESS", "Preprocessed"
        TRAINING = "TRAINING", "Training"
        EVALUATED = "EVALUATED", "Evaluated"
        SELECTED = "SELECTED", "Best model selected"
        FAILED = "FAILED", "Failed"

    # Identidad
    name = models.CharField(
        max_length=200,
        blank=True,
        help_text="Nombre opcional de la ejecución",
    )

    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.CREATED,
    )

    # Dataset seleccionado (relativo a la raíz del repo; p.ej. "data/raw/foo.csv")
    dataset = models.CharField(
        max_length=255,
        blank=True,
        default="",
        help_text="Ruta relativa del dataset (p.ej. data/raw/hotel_bookings.csv)",
    )

    # Configuración (seed, métricas, etc.)
    config = models.JSONField(
        default=dict,
        encoder=NumpyJSONEncoder,
        help_text="Configuración usada en la ejecución (seed, metric, dataset, etc.)",
    )

    # Métricas finales
    metrics = models.JSONField(
        null=True,
        blank=True,
        encoder=NumpyJSONEncoder,
        help_text="Resumen de métricas finales (best model)",
    )

    # Paths
    artifacts_path = models.CharField(
        max_length=500,
        blank=True,
        default="",
        help_text="Ruta base donde se guardan los artifacts de esta run",
    )

    # Tiempos
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    # Error handling
    error_message = models.TextField(null=True, blank=True)
    traceback = models.TextField(null=True, blank=True)

    def mark_started(self):
        self.started_at = timezone.now()
        self.save(update_fields=["started_at"])

    def mark_finished(self):
        self.finished_at = timezone.now()
        self.save(update_fields=["finished_at"])

    def __str__(self):
        return f"PipelineRun #{self.id} [{self.status}]"


class Artifact(models.Model):
    class Kind(models.TextChoices):
        DATA = "DATA", "Data"
        MODEL = "MODEL", "Model"
        REPORT = "REPORT", "Report"
        FIGURE = "FIGURE", "Figure"
        OTHER = "OTHER", "Other"

    run = models.ForeignKey(
        PipelineRun,
        on_delete=models.CASCADE,
        related_name="artifacts",
    )

    name = models.CharField(max_length=200)
    kind = models.CharField(max_length=20, choices=Kind.choices)
    path = models.CharField(max_length=500)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.kind}: {self.name}"


class StepLog(models.Model):
    class Level(models.TextChoices):
        INFO = "INFO", "Info"
        WARN = "WARN", "Warn"
        ERROR = "ERROR", "Error"
        DEBUG = "DEBUG", "Debug"

    run = models.ForeignKey(
        PipelineRun,
        on_delete=models.CASCADE,
        related_name="logs",
    )

    step = models.CharField(max_length=50)
    level = models.CharField(max_length=10, choices=Level.choices, default=Level.INFO)
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"[{self.step}] {self.level} {self.message[:40]}"


# Modelo Dataset
class Dataset(models.Model):
    name = models.CharField(max_length=200)
    filename = models.CharField(max_length=200)
    description = models.TextField(blank=True)

    target_column = models.CharField(max_length=100)
    rows = models.IntegerField(null=True, blank=True)
    columns = models.IntegerField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name
