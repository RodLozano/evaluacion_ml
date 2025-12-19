from __future__ import annotations

from pathlib import Path
from rest_framework import serializers

from api.models import PipelineRun, Artifact, StepLog


def _normalize_csv_name(v: str) -> str:
    """
    Acepta:
      - "dataset.csv"
      - "data/raw/dataset.csv"
      - "C:\\...\\data\\raw\\dataset.csv"
    y devuelve:
      - "dataset.csv"
    """
    v = (v or "").strip()
    if not v:
        return ""
    # Normaliza a "solo nombre"
    return Path(v.replace("\\", "/")).name


class PipelineRunCreateSerializer(serializers.ModelSerializer):
    """
    Crea una run.
    Reglas:
      - dataset: requerido (CSV name). Si llega como "data/raw/x.csv", se normaliza a "x.csv".
      - config: opcional. Siempre se fuerza config["raw_filename"] = dataset (si no existe).
    """

    class Meta:
        model = PipelineRun
        fields = ("id", "name", "dataset", "config")
        extra_kwargs = {
            "dataset": {"required": True, "allow_blank": False},
            "name": {"required": False, "allow_blank": True},
            "config": {"required": False},
        }

    def validate_dataset(self, value: str) -> str:
        value = _normalize_csv_name(value)
        if not value.lower().endswith(".csv"):
            raise serializers.ValidationError("dataset debe ser un .csv")
        return value

    def validate(self, attrs):
        cfg = attrs.get("config") or {}
        if not isinstance(cfg, dict):
            raise serializers.ValidationError({"config": "config debe ser un JSON object"})

        dataset = attrs.get("dataset") or ""
        dataset = _normalize_csv_name(dataset)
        if not dataset:
            raise serializers.ValidationError({"dataset": "dataset es obligatorio"})

        # Garantiza la clave que espera el pipeline
        cfg.setdefault("raw_filename", dataset)

        # Compatibilidad con implementaciones previas (si alguien usa cfg["dataset"])
        cfg.setdefault("dataset", dataset)

        attrs["dataset"] = dataset
        attrs["config"] = cfg
        return attrs


class PipelineRunSerializer(serializers.ModelSerializer):
    class Meta:
        model = PipelineRun
        fields = (
            "id",
            "name",
            "status",
            "dataset",
            "config",
            "metrics",
            "artifacts_path",
            "created_at",
            "started_at",
            "finished_at",
            "error_message",
            "traceback",
        )


class ArtifactSerializer(serializers.ModelSerializer):
    url = serializers.SerializerMethodField()
    download_url = serializers.SerializerMethodField()

    class Meta:
        model = Artifact
        fields = ("id", "name", "kind", "path", "url", "download_url", "created_at")

    def _artifacts_roots(self) -> list[Path]:
        from django.conf import settings
        base = Path(settings.BASE_DIR)
        return [
            (base / "artifacts").resolve(),
            (base.parent / "artifacts").resolve(),
        ]

    def _to_relpath(self, p: str) -> str:
        raw = (p or "").replace("\\", "/").strip()
        if not raw:
            return ""

        # Si ya es relativo con prefijo artifacts/, lo quitamos
        if raw.startswith("artifacts/"):
            return raw[len("artifacts/"):].lstrip("/")

        pp = Path(raw)

        # Si es absoluto, intentamos relativizar contra CUALQUIER root válido
        try:
            if pp.is_absolute():
                resolved = pp.resolve()
                for root in self._artifacts_roots():
                    try:
                        return str(resolved.relative_to(root)).replace("\\", "/").lstrip("/")
                    except Exception:
                        continue
        except Exception:
            pass

        # Si es relativo sin prefijo, lo devolvemos limpio
        return raw.lstrip("/")

    def get_url(self, obj: Artifact) -> str:
        rel = self._to_relpath(obj.path)
        return f"/artifacts/{rel}"

    def get_download_url(self, obj: Artifact) -> str:
        return f"/api/artifacts/{obj.id}/download/"



class StepLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = StepLog
        fields = ("id", "step", "level", "message", "created_at")
