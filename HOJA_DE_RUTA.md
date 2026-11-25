# Hoja de Ruta — Taller Final Integrador de Machine Learning (601)

## Contexto y Reglas del Proyecto

- **Curso / Actividad:** Taller Final Integrador de Machine Learning - 601 (Semana 15).
- **Dominio:** Comercio electrónico de cosméticos / fragancias (catálogo Sephora).
- **Enfoque de trabajo:** Flujo lineal sobre la rama local `main`, ejecutando y verificando cada fase por terminal antes de avanzar a la siguiente.
- **Restricciones estrictas:**
  - **Prohibido usar Random Forest** (`RandomForestRegressor` / `RandomForestClassifier`).
  - **Prohibido usar cuadernos Jupyter (`.ipynb`)** en el proyecto y entregable final; todo el código reside en scripts modulares `.py` (`1_` a `4_`).
  - **Cero métricas inventadas:** todas las cifras documentadas provienen de ejecuciones reales en `.venv`.
  - **Estructura raíz limpia:** scripts y carpeta `datos/` en la raíz del repositorio durante el desarrollo; en la Fase 6 se empaqueta dentro de `proyecto_machine_learning/` en el `.zip`.

---

## Entorno y Dataset Validados (Fase 0)

- **Entorno virtual (`.venv` con `uv`):**
  - Python: `3.11.14`
  - `pandas`: `3.1.1`
  - `scikit-learn`: `1.9.1`
  - `tensorflow`: `2.21.0` (`keras`: `3.15.1`)
- **Dataset oficial:** `datos/sephora_fragrance.csv` (subconjunto `primary_category == "Fragrance"` extraído del dataset de Kaggle *Sephora Products and Skincare Reviews*).
  - **Dimensiones:** `1,432` registros × `13` columnas (`12` variables de entrada `X` + `1` variable objetivo `y = price_usd`).
  - **Variables numéricas / binarias (8):** `loves_count`, `rating` (93 nulos, 6.49%), `reviews` (93 nulos, 6.49%), `child_count`, `limited_edition`, `new`, `online_only`, `sephora_exclusive`.
  - **Variables categóricas (4):** `brand_name` (79 valores únicos), `secondary_category` (2 nulos, 0.14%), `tertiary_category` (16 nulos, 1.12%), `variation_type` (193 nulos, 13.48%).
  - **Variable objetivo (`y`):** `price_usd` (`0` nulos, rango `10.0` – `395.0` USD, media `87.26` USD, mediana `80.00` USD, desviación estándar `63.31` USD).
  - **Calidad y nulos reales:** `397` celdas nulas en total (`2.13%` sobre el CSV completo de 18,616 celdas; `2.31%` sobre la matriz `X` de 17,184 celdas), con `186` nulos en numéricas y `211` en categóricas (dentro del rango `2%–5%` exigido).
  - **Prevención de *data leakage*:** excluidos `product_id`, `brand_id`, `product_name`, `value_price_usd`, `sale_price_usd`, `child_max_price`, `child_min_price`, `primary_category` (constante) y texto libre (`ingredients`, `highlights`, `variation_desc`, `size`, `variation_value`, `out_of_stock`).
  - **Duplicados exactos:** `3` filas exactas (derivadas de retirar `product_id`/`product_name` en variantes de un mismo catálogo).

---

## Fases del Proyecto

### Fase 0: Contextualización, Auditoría y Hoja de Ruta ✅
- **Implementación:**
  - [x] Auditar directorio inicial, leer `Taller Final Integrador de Machine Learning 601.pdf` completo y descartar el antiguo `sephora.csv` sin precios.
  - [x] Crear `.gitignore` (ignorando `.venv/`, `__pycache__/`, `*.py[cod]`, `*.zip`) y `requirements.txt` (`pandas`, `scikit-learn`, `tensorflow`).
  - [x] Crear entorno virtual `.venv` con `uv venv --python 3.11` e instalar dependencias.
  - [x] Descargar y guardar el dataset oficial filtrado en `datos/sephora_fragrance.csv`.
  - [x] Crear `HOJA_DE_RUTA.md`.
  - [x] Inicializar repositorio Git local en rama `main` y hacer commit limpio de Fase 0.
- **Verificación / Tests:**
  - [x] Importación real de `pandas`, `scikit-learn` y `tensorflow` en `.venv\Scripts\python.exe` con exit code `0`.
  - [x] Auditoría con `assert` sobre `datos/sephora_fragrance.csv`: dimensiones `(1432, 13)`, 8 numéricas + 4 categóricas, `0` nulos en `price_usd`, nulos reales en numéricas (`186`) y categóricas (`211`), porcentaje global `2.13%` (`2.31%` en `X`) dentro de `[2%, 5%]`, sin columnas ID ni de fuga de datos.
  - [ ] Confirmación del usuario sobre pregunta de negocio, modelo clásico elegido (sin Random Forest) y paso a Fase 1.

---

### Fase 1: Preprocesamiento (`1_preprocesamiento.py`) ⏳
- **Implementación:**
  - [ ] Cargar `datos/sephora_fragrance.csv` validando ausencia de IDs o fuga de datos.
  - [ ] Separar `X` (12 variables) e `y = price_usd` (continua para regresión) e imprimir resumen de variables y rango de `y`.
  - [ ] Dividir en 80% Train / 20% Test con `random_state=412` **antes** de imputar, codificar o escalar (sin data leakage).
  - [ ] Aplicar `SimpleImputer`: `strategy="median"` para numéricas y `strategy="most_frequent"` (moda) para categóricas, ajustado **solo** sobre `X_train`.
  - [ ] Aplicar One-Hot Encoding consistente a categóricas (`pd.Categorical` con categorías aprendidas en Train + `pd.get_dummies(..., drop_first=True, dtype=int)` + `reindex` para garantizar mismas columnas y tipo numérico en Pandas 3.x / TensorFlow).
  - [ ] Aplicar `StandardScaler` a las variables numéricas ajustado **solo** sobre `X_train`.
  - [ ] Exportar `datos/X_train.csv`, `datos/X_test.csv`, `datos/y_train.csv` y `datos/y_test.csv`.
- **Tests obligatorios Fase 1:**
  - [ ] Ejecutar `1_preprocesamiento.py` con exit code `0`.
  - [ ] Script de `assert` verificando: existencia de los 4 CSVs, partición 80/20 exacta (`1145` train / `287` test), `0` NaNs en los 4 archivos, columnas idénticas y 100% numéricas en `X_train` y `X_test`, media `~0` y desviación estándar `~1` en columnas numéricas escaladas de `X_train`, y distribución válida de `y_train` e `y_test`.

---

### Fase 2: Modelo Clásico Scikit-Learn (`2_modelo_clasico.py`, SIN Random Forest) ⏳
- **Implementación:**
  - [ ] Cargar `datos/X_train.csv`, `datos/X_test.csv`, `datos/y_train.csv`, `datos/y_test.csv`.
  - [ ] Instanciar y entrenar el modelo clásico acordado (sin Random Forest).
  - [ ] Exponer la función `evaluar_modelo_clasico(verbose=True)` retornando diccionario estándar (`mse`, `rmse`, `mae`, `r2`, `y_pred`, `tiempo_entrenamiento`) y mantener bloque `if __name__ == "__main__":`.
  - [ ] Imprimir control de overfitting (métricas Train vs Test), interpretación del modelo (coeficientes/pesos más influyentes) y métricas completas en Test.
- **Tests obligatorios Fase 2:**
  - [ ] Ejecución con exit code `0` sin warnings críticos.
  - [ ] Forma y rango válidos de `y_pred` (longitud `287`, sin `NaN`/`Inf`).
  - [ ] Superación clara del baseline ingenuo (predicción por media de `y_train`).
  - [ ] Brecha Train/Test controlada (sin sobreajuste severo).
  - [ ] Determinismo exacto en dos ejecuciones consecutivas.
  - [ ] Prueba de inferencia individual sobre 1 muestra de `X_test`.

---

### Fase 3: Red Neuronal TensorFlow/Keras (`3_red_neuronal.py`) ⏳
- **Implementación:**
  - [ ] Cargar los mismos archivos preprocesados desde `datos/`.
  - [ ] Fijar `tf.keras.utils.set_random_seed(412)` **dentro** de `evaluar_red_neuronal(verbose=True)` para garantizar determinismo también al importarse desde `4_comparacion.py`.
  - [ ] Definir modelo `Sequential` con capas ocultas `ReLU`, capa de salida `Dense(1, activation="linear")`, optimizador `Adam` y pérdida `mse`.
  - [ ] Entrenar monitoreando `validation_split=0.2` (sin tocar `X_test`) y evaluando finalmente sobre `X_test`.
  - [ ] Retornar exactamente las mismas llaves que `2_modelo_clasico.py` (`mse`, `rmse`, `mae`, `r2`, `y_pred`, `tiempo_entrenamiento`) y mantener `if __name__ == "__main__":`.
- **Tests obligatorios Fase 3:**
  - [ ] Forma y rango válidos de salida (sin `NaN`).
  - [ ] Convergencia real de la pérdida (`loss` / `val_loss`) durante las épocas.
  - [ ] Determinismo exacto en dos llamadas consecutivas a `evaluar_red_neuronal(verbose=False)`.
  - [ ] Superación del baseline ingenuo en Test.
  - [ ] Compatibilidad verificada con entradas `float32`.

---

### Fase 4: Comparación Final (`4_comparacion.py`) ⏳
- **Implementación:**
  - [ ] Configurar `os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")` al inicio del script.
  - [ ] Importar `2_modelo_clasico` y `3_red_neuronal` mediante `importlib.import_module`.
  - [ ] Evaluar ambos modelos sobre el mismo conjunto `X_test` / `y_test`.
  - [ ] Imprimir tabla comparativa limpia lado a lado (`MSE`, `RMSE`, `MAE`, `R²`, `Tiempo`) y conclusión técnica justificada sobre cuál modelo obtuvo mejor desempeño.
- **Tests obligatorios Fase 4 (End-to-End):**
  - [ ] Ejecutar en secuencia `1_preprocesamiento.py` -> `2_modelo_clasico.py` -> `3_red_neuronal.py` -> `4_comparacion.py` con exit code `0`.
  - [ ] Verificar coincidencia exacta de métricas entre las ejecuciones individuales y `4_comparacion.py`.

---

### Fase 5: Documentación (`README.md`) y Revisión de Presentación ⏳
- **Implementación:**
  - [ ] Redactar `README.md` completo: pregunta de negocio, resumen y fuente del dataset, pipeline de preprocesamiento, arquitectura de modelos, instrucciones de ejecución con `uv`, resultados reales verificados y conclusión técnica.
  - [ ] Auditar presentación `.pptx` o informe (si el usuario lo entrega) verificando consistencia exacta de cifras (`test` vs `val`) y cumplimiento al 100% de la rúbrica del PDF.
- **Verificación:**
  - [ ] Cero discrepancias entre los números de `README.md` / presentación y la salida real de `4_comparacion.py`.

---

### Fase 6: Empaquetado `.zip` + Prueba Limpia + Guía Maestra HTML ⏳
- **Implementación:**
  - [ ] Verificar que haya `0` archivos `.ipynb` en todo el proyecto.
  - [ ] Construir archivo `.zip` limpio con carpeta raíz `proyecto_machine_learning/` conteniendo únicamente los entregables oficiales (`datos/` con los 5 CSVs, `1_` a `4_`, `README.md`, `requirements.txt` e informe/presentación si aplica), excluyendo `.venv/`, `__pycache__/`, `.git/`, `HOJA_DE_RUTA.md` y `GUIA_SUSTENTACION.html`.
  - [ ] Crear `GUIA_SUSTENTACION.html` autocontenida, didáctica e interactiva (con diagramas, guion para sustentación en pareja de 5–8 min con placeholders *Integrante A* / *Integrante B*, diccionario técnico y banco interactivo de preguntas de código y teoría).
  - [ ] Actualizar `HOJA_DE_RUTA.md` y mantener historial de commits limpio en `main`.
- **Verificación (Prueba limpia):**
  - [ ] Descomprimir el `.zip` en una carpeta temporal fuera del workspace y ejecutar `python 4_comparacion.py` desde cero verificando exit code `0` y resultados reproducibles.
