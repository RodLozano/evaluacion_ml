from django.contrib import admin
from django.shortcuts import redirect
from django.urls import include, path

from api.views import pipeline_dashboard, pipeline_run_detail, serve_artifact

urlpatterns = [
    path("", lambda request: redirect("/pipeline/")),

    # Front
    path("pipeline/", pipeline_dashboard, name="pipeline-dashboard"),
    path("pipeline/<int:run_id>/", pipeline_run_detail, name="pipeline-run-detail"),

    # Artifacts
    path("artifacts/<path:relpath>", serve_artifact, name="serve-artifact"),

    # Admin + API
    path("admin/", admin.site.urls),
    path("api/", include("api.urls")),
]
