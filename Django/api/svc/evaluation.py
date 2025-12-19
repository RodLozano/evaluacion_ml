from api.models import PipelineRun, Artifact
from api.svc.base import capture_stdout
from evaluator import save_reports, plot_roc_comparison
from data_mger import normalize_results


def run_evaluation_step(run: PipelineRun, results, y_test):
    run.status = PipelineRun.Status.EVALUATED
    run.save(update_fields=["status"])

    with capture_stdout(run, step="EVALUATION"):
        metrics_path = save_reports(results)
        results_for_plots = normalize_results(results, y_true=y_test)
        roc_path = plot_roc_comparison(results_for_plots)

    Artifact.objects.create(
        run=run,
        name="metrics.csv",
        kind=Artifact.Kind.REPORT,
        path=str(metrics_path),
    )

    Artifact.objects.create(
        run=run,
        name="roc_compare.png",
        kind=Artifact.Kind.FIGURE,
        path=str(roc_path),
    )