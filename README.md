# Evaluación ML — Pipeline de entrenamiento, evaluación y serving (Django + CLI)

Proyecto orientado a **clasificación binaria** sobre un dataset tabular (por defecto, `dataset_practica_final.csv`) con un pipeline reproducible que:
1) carga y parte datos,  
2) construye un preprocesador (numérico + categórico),  
3) entrena varios modelos (sklearn + un MLP en Keras),  
4) evalúa (métricas + figuras),  
5) selecciona y guarda el mejor modelo sklearn,  
6) expone el flujo vía **Django REST** (ejecución asíncrona, logs en BD y descarga de artifacts).

---

## 1. Componentes principales

### 1.1 Pipeline “standalone” (CLI/Script)
El flujo de referencia está orquestado en `main.py` y ejecuta el pipeline completo (datos → preproceso → entrenamiento → evaluación → selección).

**Qué hace en alto nivel:**
- Carga dataset raw desde `data/raw/<RAW_FILENAME>`.
- Genera `train/test` (y opcionalmente `val`) y guarda splits en `data/processed/`.
- Construye `ColumnTransformer` para numéricas/categóricas.
- Evalúa un conjunto de modelos sklearn y un modelo Keras MLP.
- Guarda un CSV con métricas y una ROC comparativa.
- Selecciona el mejor modelo y lo guarda en `artifacts/models/best_model.joblib`. 

---

### 1.2 Capa “core” (src/)
#### Datos (`src/data_mger.py`)
- Define rutas “raw” y “processed”, asegura directorios y carga CSV.
- Hace split estratificado (por defecto) y guarda/recupera splits a CSV.
- Incluye utilidad `normalize_results(...)` para completar info ROC en resultados que traen `y_proba` (Keras), de cara a la ROC comparativa.

#### Preprocesado (`src/preprocessing.py`)
- Feature engineering (noches totales, huéspedes totales, fecha de llegada y variables derivadas).
- Elimina columnas con **fuga de información** (“leakage”) típicas (`reservation_status`, `reservation_status_date`).
- Construye un `ColumnTransformer`:
  - numéricas: imputación mediana + `StandardScaler`
  - categóricas: imputación moda + `OneHotEncoder(handle_unknown="ignore")`

#### Modelos sklearn (`src/models/sklearn_models.py`)
Provee un set base de modelos (sin pipeline) que luego se envuelven con `Pipeline(preprocess + model)` en evaluación:
- Logistic Regression
- Decision Tree
- Random Forest
- HistGradientBoostingClassifier

#### Modelo Keras (`src/models/keras_model.py`)
- Entrena un MLP para tabular ya transformado a matriz (dense).
- Usa early stopping monitorizando `val_auc`.
- Devuelve métricas y también `y_proba`/`y_pred` para reporting/ROC.
- Guarda el modelo como `.keras` en `artifacts/models/`.

#### Evaluación y reporting (`src/evaluator.py`)
- Entrena y evalúa un modelo sklearn dentro de un `Pipeline`.
- Calcula métricas: accuracy, precision, recall, f1, roc_auc.
- Guarda por modelo:
  - matriz de confusión (`cm_<model>.png`)
  - curva ROC (`roc_<model>.png`)
- Genera:
  - `metrics.csv` (limpia claves internas y payloads grandes como `y_proba/y_pred`)
  - `roc_compare.png` (ROC comparativa multi-modelo)

#### Selección y persistencia (`src/selector.py`)
- Selecciona el mejor resultado por una métrica principal (default `roc_auc`) y desempates (f1, recall, precision, accuracy).
- Guarda el pipeline ganador en `artifacts/models/best_model.joblib` y un JSON con metadata (`best_model.json`).

#### Predicción “offline” (`src/predict.py`)
- Carga `best_model.joblib` y predice sobre un CSV de entrada.
- Añade columnas `proba` y `pred` al CSV de salida.
- Limpia columnas de leakage y opcionalmente elimina la target si venía incluida.

---

### 1.3 Configuración (`src/config.py`)
Parámetros de ejecución por defecto:
- `SEED = 42`
- `METRIC_MAIN = "roc_auc"`
- `RAW_FILENAME = "dataset_practica_final.csv"`

> Si `src/config.py` no existe o falla el import, el código tiene defaults seguros embebidos (seed 42, roc_auc, etc.). 

---

## 2. Capa Django (API + ejecución asíncrona + artifacts + logs)

### 2.1 Modelos de BD (tracking del pipeline)
- `PipelineRun`: representa una ejecución completa (estado, dataset, config, métricas finales, artifacts_path, timestamps, errores).
- `Artifact`: referencia a archivos generados (reportes, figuras, modelos, etc.).
- `StepLog`: log estructurado por paso y nivel.
- `NumpyJSONEncoder`: permite guardar en JSONField tipos numpy sin romper la serialización.

### 2.2 Endpoints (router + rutas)
Rutas principales:
- `GET /datasets/`: lista datasets disponibles en `../data/raw/*.csv` devolviendo **solo nombres de archivo**.
- CRUD runs: ` /runs/` (DRF ViewSet).
- `POST /runs/{id}/start/`: arranca el pipeline en un thread daemon.
- `GET /runs/{id}/logs/?since=...`: recupera logs incrementalmente.
- `GET /runs/{id}/artifacts/`: lista artifacts asociados.
- `GET /artifacts/{artifact_id}/download/`: descarga un artifact. 

### 2.3 Ejecución asíncrona + captura de logs
La ejecución del pipeline en la API se lanza en background (thread) y se capturan:
- `stdout` / `stderr` (incluye `print()` y mensajes de consola) persistiendo líneas en `StepLog`.
- logging de librerías (root logger) a INFO, con WARNING+ al stream de error.
- warnings capturados vía `logging.captureWarnings(True)` y forzados a `simplefilter("default")` para no “perder” avisos.

Además, el servido de artifacts protege contra path traversal mediante `_safe_join_under(...)` y busca el archivo en roots permitidos.

---

## 3. “Service files” (api/svc/*): qué son y por qué existen
El proyecto incluye módulos “de servicio” en `api/svc/` que **adaptan** el pipeline core (`src/`) al mundo Django:
- gestionan estados (`PipelineRun.Status.*`),
- envuelven pasos con captura de salida,
- crean y registran `Artifact` en BD,
- coordinan el flujo `run_full_pipeline(...)`. 

---

## 4. Estructura de carpetas relevantes

```text
.
├─ main.py
├─ data/
│  └─ raw/
│     └─ dataset_practica_final.csv
├─ artifacts/
│  ├─ models/
│  │  ├─ best_model.joblib
│  │  ├─ best_model.json
│  │  └─ keras_model.keras
│  ├─ reports/
│  │  └─ metrics.csv
│  └─ figures/
│     ├─ roc_compare.png
│     ├─ cm_<model>.png
│     └─ roc_<model>.png
├─ src/
│  ├─ config.py
│  ├─ data_mger.py
│  ├─ preprocessing.py
│  ├─ evaluator.py
│  ├─ selector.py
│  ├─ predict.py
│  └─ models/
│     ├─ sklearn_models.py
│     └─ keras_model.py
└─ api/
   ├─ models.py
   ├─ views.py
   ├─ urls.py
   └─ svc/
      ├─ runner.py
      ├─ base.py
      ├─ data.py
      ├─ preprocessing.py
      ├─ training.py
      ├─ evaluation.py
      └─ selection.py
```

Notas de rutas:

* En la API, el dataset seleccionado se normaliza para quedarse con el **nombre de archivo** (p.ej. `"x.csv"`, no `"data/raw/x.csv"`), porque el loader espera buscarlo en `data/raw/`.
* Cada ejecución API crea un directorio por run: `artifacts/runs/<run_id>/...` y guarda ahí artifacts de esa run (además de los “globales” si tu core los genera). 

---

## 5. Cómo ejecutar

### 5.1 Requisitos

El proyecto usa, entre otros:

* Python (entorno virtual recomendado)
* pandas, numpy
* scikit-learn
* tensorflow/keras (para el MLP)
* Django + Django REST Framework (para la API)

### 5.2 Preparar el dataset

1. Coloca el CSV raw en:

```text
data/raw/dataset_practica_final.csv
```

(ó el nombre que configures en `src/config.py`).

2. Asegúrate de que la columna target existe (por defecto se espera `is_canceled` si no se redefine).

### 5.3 Ejecutar pipeline standalone

```bash
python main.py
```

Generará splits (si `save_processed=True`), reportes y artifacts.

### 5.4 Predicción offline con el mejor modelo sklearn

Requiere haber generado previamente `artifacts/models/best_model.joblib`.

```bash
python -m src.predict \
  --input data/processed/X_test.csv \
  --output artifacts/reports/predictions.csv \
  --threshold 0.5
```

---

## 6. Uso vía API (Django)

Se detalla a continuación el procedimiento para usar la API de producción desplegada ahora mismo en servidores propios de los autores. También es posible ejecutar Django en local y utilizar los mismos endpoints. Para hacer esto, ejecuta

```bash
docker compose up --build
```
Y sustituye las URLs expuestas a continuación por `localhost`.

### 6.1 Listar datasets disponibles

```bash
curl -s https://ml.spark-ops.com/datasets/
```

Devuelve nombres `*.csv` presentes en `../data/raw/`.

### 6.2 Crear una run

La creación (`POST /runs/`) guarda:

* `status=CREATED`
* normaliza el dataset a nombre de archivo
* inicializa `run.config.raw_filename`
* crea directorio `artifacts/runs/<id>/` y lo registra como `run.artifacts_path` 

### 6.3 Lanzar la run

```bash
curl -X POST https://ml.spark-ops.com/runs/<id>/start/
```

Esto inicia `_run_pipeline_async(...)` y el pipeline completo.

### 6.4 Ver logs (incremental)

```bash
curl -s "https://ml.spark-ops.com/runs/<id>/logs/"
# o incremental:
curl -s "https://ml.spark-ops.com/runs/<id>/logs/?since=2025-12-19T18:00:00Z"
```

Los logs se guardan por línea y por “step”.

### 6.5 Ver y descargar artifacts

* Listado:

```bash
curl -s https://ml.spark-ops.com/runs/<id>/artifacts/
```

* Descarga:

```bash
curl -L -o metrics.csv https://ml.spark-ops.com/artifacts/<artifact_id>/download/
```

---

## 7. Artefactos generados

### 7.1 Reportes

* `artifacts/reports/metrics.csv`: tabla de métricas por modelo, limpiando payloads grandes y claves internas. 

### 7.2 Figuras

* `artifacts/figures/cm_<model>.png`: matrices de confusión.
* `artifacts/figures/roc_<model>.png`: ROC por modelo.
* `artifacts/figures/roc_compare.png`: ROC comparativa multi-modelo.

### 7.3 Modelos

* `artifacts/models/best_model.joblib`: el **pipeline sklearn** ganador (preprocess + modelo).
* `artifacts/models/best_model.json`: metadata “limpia” del ganador.
* `artifacts/models/keras_model.keras`: modelo Keras (comparativo).