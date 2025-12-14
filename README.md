# Práctica Rodrigo Lozano — AutoML “from scratch” (Clasificación binaria)

Sistema modular para **entrenar, evaluar y comparar modelos de clasificación binaria**, seleccionar el mejor según una **métrica principal**, y permitir **inferencia** de forma reproducible.

> Basado en el guion de entrega final del módulo de *Machine Learning y Deep Learning* del Máster en IA, Cloud Computing y DevOps. :contentReference[oaicite:0]{index=0}

---

## 🧠 Contexto y objetivo

El objetivo de esta práctica es diseñar e implementar un sistema automático tipo “AutoML” que:

- Entrene, evalúe y compare distintos modelos de **clasificación binaria**
- Seleccione el **mejor modelo** según una **métrica principal** (y reporte métricas secundarias)
- Automatice el flujo completo: **datos → preprocesado → entrenamiento → evaluación → selección → persistencia → predicción**

---

## 📦 Dataset

- **Fuente**: dataset proporcionado por el profesor (ver carpeta `data/`).
- **Variable objetivo**: binaria (0/1).
- **Motivación del problema**: *(completa aquí con el caso del dataset: qué predice y por qué es relevante).*

> Nota: si el dataset no puede subirse por privacidad, se incluye un ejemplo de estructura y un script de descarga/preparación.

---

## ✅ Requisitos cubiertos

Modelos implementados (mínimo obligatorio):

- Regresión Logística
- Árbol de Decisión
- Random Forest
- Gradient Boosting (XGBoost / LightGBM / CatBoost)
- Red neuronal multicapa (Keras - TensorFlow)

Evaluación:

- Métrica principal: **(e.g., F1-score / AUC-ROC / Recall...)** *(justificar abajo)*
- Matriz de confusión
- Curva ROC

Automatización:

- Pipeline estructurado para carga de datos, preprocesado, entrenamiento, evaluación y selección.

---

## 🧪 Métricas y criterio de selección

**Métrica principal elegida:** `TODO` (por ejemplo `F1-score`)

**Justificación:**  
`TODO` (ejemplo: “El dataset está desbalanceado y queremos equilibrio entre precisión y recall, por eso F1.”)

Métricas reportadas:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

---

## 🏗️ Estructura del repositorio

```text
.
├── data/
│   ├── raw/                # datos originales (si aplica)
│   └── processed/          # datos procesados / splits
├── notebooks/
│   └── 01_eda.ipynb        # análisis exploratorio (EDA)
├── src/
│   ├── config.py           # configuración global (paths, semilla, etc.)
│   ├── data_loader.py      # carga + particionado
│   ├── preprocessing.py    # pipeline de preprocesado (sklearn)
│   ├── models/
│   │   ├── sklearn_models.py  # LR, DT, RF, Boosting
│   │   └── keras_model.py     # MLP (Keras)
│   ├── trainer.py          # entrenamiento
│   ├── evaluator.py        # métricas + plots (ROC, confusion matrix)
│   ├── selector.py         # selección del mejor modelo
│   ├── predict.py          # inferencia con modelo final
│   └── utils.py            # utilidades (logging, seeds, etc.)
├── artifacts/
│   ├── models/             # modelos entrenados (joblib / keras)
│   ├── reports/            # tablas de métricas
│   └── figures/
