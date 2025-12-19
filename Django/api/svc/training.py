from api.models import PipelineRun
from api.svc.base import capture_stdout
from models.sklearn_models import get_sklearn_models
from models.keras_models import train_keras_model, KerasTrainConfig
from evaluator import evaluate_model, EvalConfig


def run_training_step(run, preprocessor, splits):
    run.status = PipelineRun.Status.TRAINING
    run.save(update_fields=["status"])

    results = []
    models = get_sklearn_models(seed=run.config.get("seed", 42))
    cfg = EvalConfig(threshold=run.config.get("threshold", 0.5))

    with capture_stdout(run, step="TRAINING"):
        # 1) sklearn
        for name, model in models.items():
            r = evaluate_model(
                name=name,
                model=model,
                preprocessor=preprocessor,
                X_train=splits["X_train"],
                y_train=splits["y_train"],
                X_test=splits["X_test"],
                y_test=splits["y_test"],
                cfg=cfg,
                fit=True,
            )
            results.append(r)

        # 2) keras
        keras_metrics = train_keras_model(
            preprocessor=preprocessor,
            X_train=splits["X_train"],
            y_train=splits["y_train"],
            X_test=splits["X_test"],
            y_test=splits["y_test"],
            cfg=KerasTrainConfig(
                seed=run.config.get("seed", 42),
                epochs=run.config.get("keras_epochs", 40),
                patience=run.config.get("keras_patience", 6),
                batch_size=run.config.get("keras_batch_size", 256),
            ),
            fit_preprocessor=True,
        )
        results.append(keras_metrics)

    return results