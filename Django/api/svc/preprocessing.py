import joblib
from pathlib import Path

from api.models import PipelineRun, Artifact
from api.svc.base import capture_stdout
from preprocessing import build_preprocessor


def run_preprocessing_step(run: PipelineRun, X_train):
    run.status = PipelineRun.Status.PREPROCESS
    run.save(update_fields=["status"])

    artifacts_dir = Path(run.artifacts_path) / "preprocess"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    with capture_stdout(run, step="PREPROCESS"):
        preprocessor = build_preprocessor(X_train)

    path = artifacts_dir / "preprocessor.joblib"
    joblib.dump(preprocessor, path)

    Artifact.objects.create(
        run=run,
        name="preprocessor.joblib",
        kind=Artifact.Kind.OTHER,
        path=str(path),
    )

    return preprocessor
