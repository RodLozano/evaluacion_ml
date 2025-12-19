from django.urls import path
from rest_framework.routers import DefaultRouter

from api.views import (
    PipelineRunViewSet,
    list_datasets,
    download_artifact,
)

router = DefaultRouter()
router.register(r"runs", PipelineRunViewSet, basename="runs")

urlpatterns = [
    path("datasets/", list_datasets, name="list-datasets"),
    path("artifacts/<int:artifact_id>/download/", download_artifact, name="download-artifact"),
]

urlpatterns += router.urls
