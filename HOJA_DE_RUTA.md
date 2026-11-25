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
  - [x] Confirmación del usuario sobre pregunta de negocio, modelo clásico elegido (sin Random Forest) y paso a Fase 1:
    - **Pregunta de negocio aprobada:** *«¿Con qué precisión se puede predecir el precio en USD (`price_usd`) de una fragancia en Sephora según su marca, subcategoría, tipo de variación, popularidad y atributos comerciales?»*
    - **Modelo clásico elegido:** Regresión Lineal Múltiple (`LinearRegression`, Scikit-Learn; sin Random Forest).

---

### Fase 1: Preprocesamiento (`1_preprocesamiento.py`) ✅
- **Implementación:**
  - [x] Cargar `datos/sephora_fragrance.csv` validando ausencia de IDs o fuga de datos (`0` columnas excluidas en carga al estar ya limpio el archivo).
  - [x] Separar `X` (12 variables) e `y = price_usd` (regresión continua, sin mapeo de clases) e imprimir resumen de variables y rango de `y`.
  - [x] Dividir en 80% Train / 20% Test (`1,145` / `287` registros) con `random_state=412` **antes** de imputar, codificar o escalar (sin *data leakage*).
  - [x] Aplicar `SimpleImputer`: `strategy="median"` para numéricas (`rating=4.3333`, `reviews=49.0`) y `strategy="most_frequent"` (moda) para categóricas (`secondary_category="Women"`, `tertiary_category="Perfume"`, `variation_type="Size + Concentration + Formulation"`), ajustado **solo** sobre `X_train`.
  - [x] Aplicar One-Hot Encoding consistente a categóricas (`pd.Categorical` con categorías aprendidas en Train + `pd.get_dummies(..., drop_first=True, dtype=int)` + `reindex`), generando `92` columnas dummy sin warnings en Pandas 3.1.1 (2 categorías no vistas en Test —`brand_name: 'Eight & Bob'` y `variation_type: 'Formulation'`— codificadas limpiamente en 0).
  - [x] Aplicar `StandardScaler` a las 8 variables numéricas ajustado **solo** sobre `X_train`.
  - [x] Exportar `datos/X_train.csv`, `datos/X_test.csv`, `datos/y_train.csv` y `datos/y_test.csv`.
- **Tests obligatorios Fase 1:**
  - [x] Ejecutar `1_preprocesamiento.py` con exit code `0` y `0` warnings.
  - [x] Script de `assert` verificado: existencia de los 4 CSVs, partición 80/20 exacta (`1,145 × 100` train / `287 × 100` test), `0` NaNs e infinitos en los 4 archivos, columnas idénticas y 100% numéricas (`float32`), media `~0.0` y `std(ddof=0) = 1.00000000` en las 8 numéricas de `X_train`, dummies en `{0, 1}`, distribución válida de `y` (`y_train` media `86.77`, `y_test` media `89.23` USD) y determinismo exacto verificado por SHA256 en dos ejecuciones consecutivas.

---

### Fase 2: Modelo Clásico Scikit-Learn (`2_modelo_clasico.py`, SIN Random Forest) ✅
- **Implementación:**
  - [x] Cargar `datos/X_train.csv`, `datos/X_test.csv`, `datos/y_train.csv`, `datos/y_test.csv`.
  - [x] Instanciar y entrenar el modelo clásico acordado: `LinearRegression` (Regresión Lineal Múltiple, sin Random Forest) en `0.1000` s.
  - [x] Exponer `evaluar_modelo_clasico(verbose=True)` retornando diccionario estándar de 12 llaves (`modelo`, `mse`, `rmse`, `mae`, `r2`, `train_*`, `y_pred`, `tiempo_entrenamiento`, `historial_aprendizaje` con curva de aprendizaje 20%–100% en holdout interno) y mantener bloque `if __name__ == "__main__":`.
  - [x] Imprimir control de overfitting (Train vs. Test), comparación contra baseline ingenuo e interpretación del modelo (intercepto `β0 = +43.76` USD y Top 8 coeficientes positivos/negativos en USD).
- **Tests obligatorios Fase 2:**
  - [x] Ejecución directa con exit code `0` y `0` warnings.
  - [x] Forma y rango válidos de `y_pred` (`shape=(287,)`, sin `NaN`/`Inf`, min `-9.88`, mediana `94.93`, max `247.83` USD; `8` predicciones negativas en combinaciones extremas).
  - [x] Superación clara del baseline ingenuo (`R² = -0.0012`, `RMSE = 70.06` USD, `MAE = 53.56` USD vs. modelo en Test: `MSE = 1325.1320`, `RMSE = 36.4024` USD, `MAE = 24.7604` USD, `R² = 0.7297`).
  - [x] Brecha Train/Test controlada: Train (`MSE = 1106.54`, `RMSE = 33.26` USD, `MAE = 21.89` USD, `R² = 0.7072`) vs. Test (`R² = 0.7297`), brecha `Train R² - Test R² = -0.0225` (cero *overfitting*).
  - [x] Determinismo exacto verificado en dos llamadas consecutivas (`y_pred` y métricas idénticas).
  - [x] Prueba de inferencia individual sobre la muestra 0 de `X_test`: real `195.00` USD vs. predicho `157.75` USD (error `-37.25` USD).

---

### Fase 3: Red Neuronal TensorFlow/Keras (`3_red_neuronal.py`) ✅
- **Implementación:**
  - [x] Cargar los mismos archivos preprocesados desde `datos/` convertidos a `float32`.
  - [x] Fijar `tf.keras.utils.set_random_seed(412)` **dentro** de `evaluar_red_neuronal(verbose=True)` (y `enable_op_determinism()` en `entrenar_red_neuronal`) para garantizar determinismo también al importarse desde `4_comparacion.py`.
  - [x] Definir modelo `Sequential`: `Input(100) -> Dense(64, ReLU, L2=0.01) -> Dropout(0.10) -> Dense(32, ReLU, L2=0.01) -> Dropout(0.10) -> Dense(1, linear)` (`8,577` parámetros), optimizador `Adam(lr=0.003)` y pérdida `mse`.
  - [x] Entrenar monitoreando `validation_split=0.2` con `EarlyStopping(patience=20)` (sin tocar `X_test`): `72/200` épocas ejecutadas, mejor época `52`, tiempo `10.4251` s.
  - [x] Retornar exactamente las mismas 12 llaves que `2_modelo_clasico.py` (`modelo`, `mse`, `rmse`, `mae`, `r2`, `train_*`, `y_pred`, `tiempo_entrenamiento`, `historial_aprendizaje` con detalle época por época `1..72`) y mantener `if __name__ == "__main__":`.
- **Tests obligatorios Fase 3:**
  - [x] Forma, contrato de 12 llaves y rango válidos de salida (`shape=(287,)`, sin `NaN`/`Inf`, min `17.41`, mediana `77.13`, max `302.20` USD, `0` predicciones negativas).
  - [x] Convergencia real sin `NaN`: `loss` baja de `10519.08` (época 1) a `701.66` (época 52); `val_loss` baja de `9301.32` a `876.19` (`val_mae = 18.59` USD).
  - [x] Determinismo exacto en dos llamadas consecutivas a `evaluar_red_neuronal(verbose=False)` (`max(|y1 - y2|) = 0.00000000`).
  - [x] Superación del baseline ingenuo (`R² = -0.0012`) y brecha controlada en Test: Train (`MSE = 661.96`, `RMSE = 25.73` USD, `MAE = 15.77` USD, `R² = 0.8248`) vs. Test (`MSE = 1145.8261`, `RMSE = 33.8501` USD, `MAE = 21.0231` USD, `R² = 0.7662`, brecha `+0.0586`).
  - [x] Compatibilidad verificada con entradas `float32` e inferencia individual sobre muestra 0: real `195.00` USD vs. predicho `157.32` USD (error `-37.68` USD).

---

### Fase 4: Comparación Final (`4_comparacion.py`) ✅
- **Implementación:**
  - [x] Configurar `os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")` y `TF_ENABLE_ONEDNN_OPTS=0` al inicio del script.
  - [x] Importar `2_modelo_clasico` y `3_red_neuronal` mediante `importlib.import_module`.
  - [x] Evaluar ambos modelos sobre el mismo conjunto `X_test` / `y_test` (`287 × 100`) y verificar independientemente que todas las métricas coincidan con `sklearn.metrics`.
  - [x] Imprimir evolución época por época (`1..72`) de la Red Neuronal MLP, curva de aprendizaje (`20%..100%`) de `LinearRegression`, cuadro comparativo lado a lado y conclusión técnica justificada.
- **Tests obligatorios Fase 4 (End-to-End):**
  - [x] Ejecutar en secuencia `1_preprocesamiento.py` -> `2_modelo_clasico.py` -> `3_red_neuronal.py` -> `4_comparacion.py` todos con exit code `0`.
  - [x] Verificar coincidencia exacta (`atol=1e-6`) entre ejecuciones individuales y `4_comparacion.py`, contrato de 12 llaves, 0 NaNs en historiales y ganador consistente: **Red Neuronal MLP (TensorFlow/Keras)** (`MSE=1145.83`, `RMSE=33.85` USD, `MAE=21.02` USD, `R²=0.7662` vs. `LinearRegression`: `MSE=1325.13`, `RMSE=36.40` USD, `MAE=24.76` USD, `R²=0.7297`; mejora `ΔR²=+0.0366`, `-15.09%` en `MAE`, `-13.53%` en `MSE` y `0` predicciones negativas).

---

### Fase 5: Documentación (`README.md`) y Presentación (`Presentacion_Taller_ML.pptx`) ✅
- **Implementación:**
  - [x] Redactar `README.md` completo: pregunta de negocio, auditoría del dataset, pipeline de preprocesamiento sin *leakage*, modelos clásico y neuronal, instalación con `uv`, resultados reales verificados y conclusión técnica.
  - [x] Generar `Presentacion_Taller_ML.pptx` (10 diapositivas 16:9 con 4 gráficas reales de resultados, placeholders *Integrante A* / *Integrante B* y cobertura completa de la rúbrica PDF).
- **Verificación:**
  - [x] Auditoría visual de las 4 gráficas y auditoría imparcial automatizada de `README.md` y `Presentacion_Taller_ML.pptx`: `10` diapositivas, `4` gráficas embebidas dentro del lienzo, `0` mezcla `val` vs. `test` y `0` discrepancias frente a `4_comparacion.py`.

---

### Fase 6: Empaquetado `.zip` + Prueba Limpia + Guía Maestra HTML ✅
- **Implementación:**
  - [x] Verificado `0` archivos `.ipynb` en todo el proyecto.
  - [x] Construido `proyecto_machine_learning.zip` (`698,694` bytes) bajo la carpeta raíz `proyecto_machine_learning/` conteniendo únicamente los `12` entregables oficiales (`datos/` con los 5 CSVs, `1_` a `4_`, `README.md`, `requirements.txt` y `Presentacion_Taller_ML.pptx`), excluyendo `.venv/`, `__pycache__/`, `.git/`, `HOJA_DE_RUTA.md`, `GUIA_SUSTENTACION.html` y el PDF de instrucciones.
  - [x] Creada `GUIA_SUSTENTACION.html` 100% autocontenida (0 enlaces externos), didáctica e interactiva (3 diagramas SVG, cronómetro de 5–8 min, barra de progreso en `localStorage`, guion en pareja con placeholders *Integrante A* / *Integrante B*, diccionario buscable de 22 términos y banco interactivo de 18 preguntas con filtros).
  - [x] Incluida en `GUIA_SUSTENTACION.html` la justificación completa de selección del dataset (descarte del CSV previo sin precios, cumplimiento del PDF y prevención de *data leakage*) y el análisis imparcial de **Regresión Lineal Múltiple (`LinearRegression`) vs. Random Forest**.
  - [x] Explicadas en **lenguaje cotidiano muy sencillo** (además del técnico) en `GUIA_SUSTENTACION.html`: (a) por qué en **regresión** se usan `MSE`, `RMSE`, `MAE` y `R²` en lugar de `Accuracy`, `Precision`, `Recall` y `F1`, y (b) por qué la Red Neuronal entrena por **épocas** (`1..72`) mediante descenso de gradiente (`Adam`) mientras `LinearRegression` resuelve en **1 paso analítico cerrado (SVD)** sin épocas y se analiza mediante curva de aprendizaje (`20%..100%`).
  - [x] Actualizada `HOJA_DE_RUTA.md`, commit final en `main` y push a `origin/main` (`https://github.com/Bonifacho/machine_charlie_learning.git`).
- **Verificación (Prueba limpia):**
  - [x] Descomprimido `proyecto_machine_learning.zip` en carpeta temporal externa (`%TEMP%\prueba_limpia_ml_*`), ejecutados `1_preprocesamiento.py` y `4_comparacion.py` desde cero con `exit code 0` y coincidencia exacta de métricas (`R²=0.7662` vs. `0.7297`), eliminando la carpeta temporal al finalizar.
