from django.utils import timezone

from api.models import PipelineRun
from api.svc.data import run_data_step
from api.svc.preprocessing import run_preprocessing_step
from api.svc.training import run_training_step
from api.svc.evaluation import run_evaluation_step
from api.svc.selection import run_selection_step


def run_full_pipeline(run: PipelineRun):
    try:
        run.mark_started()

        splits = run_data_step(run)
        preprocessor = run_preprocessing_step(run, splits["X_train"])
        results = run_training_step(run, preprocessor, splits)
        run_evaluation_step(run, results, y_test=splits["y_test"])
        run_selection_step(run, results)

        run.mark_finished()

    except Exception as e:
        run.status = PipelineRun.Status.FAILED
        run.error_message = str(e)
        run.save(update_fields=["status", "error_message"])
        raise
