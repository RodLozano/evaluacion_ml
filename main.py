# orquestacion del flujo 
# preprocessing --> trainer --> evaluator --> selector --> predictor --> explainer??

# main.py
from __future__ import annotations

from typing import List, Dict, Any

# Datos
from src.data_mger import get_data

# Preprocessing
from src.preprocessing import build_preprocessor

# Modelos
from src.models.sklearn_models import get_sklearn_models
from src.models.keras_models import train_keras_model, KerasTrainConfig

# Evaluación y selección
from src.evaluator import evaluate_model, save_reports, plot_roc_comparison, EvalConfig
from src.selector import select_best_model, save_best_model

# Config (si existe)
try:
    from src.config import SEED, METRIC_MAIN, RAW_FILENAME  # type: ignore
except Exception:
    SEED = 42
    METRIC_MAIN = "roc_auc"  # o "f1"
    RAW_FILENAME = "hotel_bookings.csv"


def run_pipeline() -> None:
    # 1) Cargar y split de datos
    print(f"\n Cargando datos raw")
    splits = get_data(
        raw_filename=RAW_FILENAME,
        save_processed=True,
        seed=SEED,
        # val_size=0.2,  # si quieres validación separada
    )
    print(f"\n Datos raw cargados")
    print(f"\n Splitting de datos raw")

    X_train, y_train = splits["X_train"], splits["y_train"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    print(f"\n Datos raw partidos")    

    # 2) Preprocessor (ColumnTransformer)    
    print(f"\n Preprocsando datos raw partidos, x_train")    
    preprocessor = build_preprocessor(X_train)
    
    print(f"\n Datos raw partidos x_train preprocesados")  

    # 3) Evaluar modelos sklearn
    print(f"\n Obteniendo modelos predefinidos")  
    results: List[Dict[str, Any]] = []
    models = get_sklearn_models(seed=SEED)
    
    print(f"\n Modelos predefinidos obtenidos") 

    eval_cfg = EvalConfig(threshold=0.5)

    for name, model in models.items():
        print(f"\n Entrenando sklearn: {name}")
        r = evaluate_model(
            name=name,
            model=model,
            preprocessor=preprocessor,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            cfg=eval_cfg,
            fit=True,
        )
        results.append(r)
        print(f"\n Modelo {name} entrenado")

    # 4) Evaluar Keras (MLP)
    # Nota: aquí usamos el MISMO preprocessor, pero Keras entrena fuera de sklearn Pipeline.
    print("\n Entrenando Keras: keras_mlp")
    keras_metrics = train_keras_model(
        preprocessor=preprocessor,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        cfg=KerasTrainConfig(seed=SEED, epochs=40, patience=6, batch_size=256),
        fit_preprocessor=True,  # Keras hará fit_transform en train
    )
    # Para integrarlo con el selector, añadimos campos esperados (sin pipeline sklearn).
    # OJO: selector.py guarda best_model.joblib, así que keras NO se puede guardar ahí tal cual.
    # Lo incluimos en report pero NO compite por "best pipeline" (ver más abajo).
    results.append(keras_metrics)
    print(f"\n Modelo keras_mlp entrenado")

    # 5) Guardar tabla de métricas + ROC comparativa (solo sklearn)
    metrics_path = save_reports(results, filename="metrics.csv")
    print(f"\n Métricas guardadas en: {metrics_path}")

    # ROC comparativa: solo modelos sklearn (los que tengan _fpr/_tpr)
    roc_path = plot_roc_comparison(results, filename="roc_compare.png", title="ROC Comparison (sklearn models)")
    print(f" ROC comparativa guardada en: {roc_path}")

    # 6) Selección del mejor modelo sklearn y guardado
    # Filtramos resultados que tengan _pipeline (solo sklearn)
    sklearn_results = [r for r in results if "_pipeline" in r]

    best = select_best_model(
        sklearn_results,
        metric=METRIC_MAIN,
        tie_breakers=("f1", "recall", "precision", "accuracy"),
    )

    model_path, meta_path = save_best_model(best)
    print("\n Mejor modelo (sklearn):", best["model"])
    print(" Guardado en:", model_path)
    print(" Metadata:", meta_path)

    # Info extra: Keras se guarda aparte en artifacts/models/keras_model.keras
    if "model_path" in keras_metrics:
        print("\n Modelo Keras guardado en:", keras_metrics["model_path"])
        print(" Nota: el 'best_model.joblib' es sklearn. Keras se usa como comparación.")


def main() -> None:
    print("Iniciando pipeline de entrenamiento/evaluación...")
    run_pipeline()
    print("\n Pipeline finalizado.")


if __name__ == "__main__":
    main()
