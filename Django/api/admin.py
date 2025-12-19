from django.contrib import admin
from api.models import PipelineRun, Artifact, StepLog, Dataset

@admin.register(PipelineRun)
class PipelineRunAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "status", "created_at", "finished_at")
    list_filter = ("status",)
    readonly_fields = ("created_at", "started_at", "finished_at")

@admin.register(Artifact)
class ArtifactAdmin(admin.ModelAdmin):
    list_display = ("id", "run", "name", "kind", "created_at")
    list_filter = ("kind",)

@admin.register(StepLog)
class StepLogAdmin(admin.ModelAdmin):
    list_display = ("id", "run", "step", "created_at")
    list_filter = ("step",)

@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "filename", "target_column", "created_at")
