# Taller Final Integrador de Machine Learning (601) — Predicción de Precios de Fragancias Sephora

Proyecto modular en Python que compara un modelo clásico de **Scikit-Learn** (**Regresión Lineal Múltiple**) frente a una **Red Neuronal Artificial (MLP)** implementada en **TensorFlow/Keras** para predecir el precio de venta en dólares (`price_usd`) de fragancias comercializadas en Sephora.

---

## 1. Pregunta de Negocio

> *«¿Con qué precisión se puede predecir el precio en USD (`price_usd`) de una fragancia en Sephora según su marca, subcategoría, tipo de variación, popularidad y atributos comerciales?»*

- **Tipo de problema:** Regresión supervisada continua.
- **Variable objetivo (`y`):** `price_usd` (precio de venta en dólares estadounidenses, USD).
- **Métricas de evaluación:** `MSE`, `RMSE`, `MAE` y $R^2$. *(Nota: al ser un problema de regresión continua y no de clasificación, métricas como Accuracy, Precision, Recall o F1-Score no aplican; el error se mide directamente en dólares y varianza explicada).*

---

## 2. Descripción y Auditoría del Dataset

- **Archivo utilizado:** [`datos/sephora_fragrance.csv`](datos/sephora_fragrance.csv)
- **Fuente original:** Subconjunto `primary_category == "Fragrance"` extraído del catálogo público de Kaggle *Sephora Products and Skincare Reviews* (`product_info.csv`, 8,494 productos).

| Criterio de la Rúbrica | Requisito del Taller | Dataset Auditado |
|---|---|---|
| Formato | `.csv` separado por comas | CSV UTF-8 estándar (`1,432 × 13`) |
| Volumen de registros | 1,000 a 5,000 filas | **`1,432`** registros |
| Variables de entrada | 6 a 15 columnas | **`12`** variables de entrada `X` (+ `price_usd`) |
| Tipos de variables | Numéricas y categóricas | **`8`** numéricas/binarias + **`4`** categóricas |
| Presencia de nulos | 2% a 5% real (justifica `SimpleImputer`) | **`2.13%`** global (`397 / 18,616` celdas; **`2.31%`** en `X`: `186` numéricos y `211` categóricos) |
| Nulos en objetivo `y` | 0% | **`0`** nulos en `price_usd` |

**Detalle de variables y prevención de fuga de datos (*data leakage*):**
- **Numéricas y binarias (8):** `loves_count`, `rating` (93 nulos, `6.49%`), `reviews` (93 nulos, `6.49%`), `child_count`, `limited_edition`, `new`, `online_only`, `sephora_exclusive`.
- **Categóricas (4):** `brand_name` (`79` marcas únicas), `secondary_category` (`2` nulos, `0.14%`), `tertiary_category` (`16` nulos, `1.12%`), `variation_type` (`193` nulos, `13.48%`).
- **Distribución de `price_usd`:** mínimo `10.00` USD, Q1 `34.00` USD, mediana `80.00` USD, media `87.26` USD, Q3 `123.00` USD, máximo `395.00` USD (desviación estándar `63.31` USD).
- **Columnas excluidas del archivo fuente:** identificadores (`product_id`, `brand_id`, `product_name`), precios derivados con fuga de datos o nulos extremos (`value_price_usd`, `sale_price_usd`, `child_max_price`, `child_min_price`), columna constante (`primary_category`) y texto no estructurado (`ingredients`, `highlights`, `variation_desc`, `size`, `variation_value`, `out_of_stock`).

---

## 3. Estructura del Proyecto

```text
proyecto_machine_learning/
├── datos/
│   ├── sephora_fragrance.csv      # Dataset crudo validado (1,432 x 13)
│   ├── X_train.csv                # Matriz de entrenamiento procesada (1,145 x 100)
│   ├── X_test.csv                 # Matriz de prueba no vista (287 x 100)
│   ├── y_train.csv                # Precios objetivo de entrenamiento (1,145 x 1)
│   └── y_test.csv                 # Precios objetivo de prueba (287 x 1)
├── 1_preprocesamiento.py          # Limpieza, split 80/20, imputación, OHE y escalado
├── 2_modelo_clasico.py            # Regresión Lineal Múltiple (Scikit-Learn)
├── 3_red_neuronal.py              # Red Neuronal MLP Sequential (TensorFlow/Keras)
├── 4_comparacion.py               # Evaluación comparativa y conclusión en consola
├── Presentacion_Taller_ML.pptx    # Presentación ejecutiva base (10 diapositivas)
├── requirements.txt               # Dependencias mínimas (pandas, scikit-learn, tensorflow)
└── README.md
```

---

## 4. Instalación y Ejecución con `uv`

Requisitos previos: Python `3.11` y [`uv`](https://docs.astral.sh/uv/).

```powershell
# 1. Crear el entorno virtual e instalar dependencias
uv venv --python 3.11
uv pip install -r requirements.txt

# 2. Ejecutar el pipeline completo paso a paso (Windows)
.\.venv\Scripts\python.exe 1_preprocesamiento.py
.\.venv\Scripts\python.exe 2_modelo_clasico.py
.\.venv\Scripts\python.exe 3_red_neuronal.py

# 3. Ejecutar la demostración comparativa final (prueba en vivo)
.\.venv\Scripts\python.exe 4_comparacion.py
```

*(En Linux/macOS, sustituye `.\.venv\Scripts\python.exe` por `.venv/bin/python`).*

---

## 5. Pipeline de Preprocesamiento (`1_preprocesamiento.py`)

Para evitar cualquier fuga de información (*data leakage*) entre entrenamiento y prueba, todas las estadísticas de transformación se ajustan **únicamente sobre Train**:

1. **Partición 80/20 previa (`random_state=412`):** `1,145` muestras de entrenamiento (`79.96%`) y `287` muestras de prueba (`20.04%`).
2. **Imputación (`SimpleImputer`):**
   - Numéricas con la **mediana** de Train (`rating = 4.3333`, `reviews = 49.0`, `loves_count = 4969.0`).
   - Categóricas con la **moda** de Train (`brand_name = "NEST New York"`, `secondary_category = "Women"`, `tertiary_category = "Perfume"`, `variation_type = "Size + Concentration + Formulation"`).
3. **One-Hot Encoding consistente (`drop_first=True`, `dtype=int`):** categorías aprendidas solo de Train (`pd.Categorical` + `pd.get_dummies` + `reindex`). Genera `92` columnas binarias dummy; las 2 categorías presentes únicamente en Test (`brand_name: 'Eight & Bob'` y `variation_type: 'Formulation'`) quedan codificadas en `0`.
4. **Escalado (`StandardScaler`):** aplicado únicamente a las `8` variables numéricas ajustado en Train (media `0.0` y desviación estándar poblacional `1.0`).
5. **Salida verificada:** matrices `100%` numéricas (`1,145 × 100` y `287 × 100`) con `0` NaNs y determinismo exacto verificado por SHA256.

---

## 6. Modelos Implementados

### Modelo Clásico — Regresión Lineal Múltiple (`2_modelo_clasico.py`)
- **Algoritmo:** `sklearn.linear_model.LinearRegression` (sin Random Forest).
- **Naturaleza del entrenamiento:** calcula la solución óptima global mediante descomposición matricial en **un solo paso analítico cerrado (SVD / mínimos cuadrados)**, sin requerir épocas.
- **Interpretabilidad en USD:** intercepto $\beta_0 = +43.76$ USD; mayores incrementos positivos en `tertiary_category_Diffusers` (`+99.77` USD), `secondary_category_Women` (`+90.94` USD) y `brand_name_TOM FORD` (`+85.68` USD); mayores ajustes negativos en `brand_name_The Phluid Project` (`-103.78` USD) y `variation_type_Type` (`-81.03` USD).

### Red Neuronal Artificial — MLP Secuencial (`3_red_neuronal.py`)
- **Arquitectura `Sequential` (8,577 parámetros entrenables):**
  `Input(100) → Dense(64, ReLU, L2=0.01) → Dropout(0.10) → Dense(32, ReLU, L2=0.01) → Dropout(0.10) → Dense(1, linear)`
- **Compilación y entrenamiento:** optimizador `Adam(learning_rate=0.003)`, pérdida `mse`, `batch_size=32`, `validation_split=0.20` (sin tocar `X_test`) y `EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)`.
- **Reproducibilidad:** semilla `tf.keras.utils.set_random_seed(412)` y `enable_op_determinism()` fijadas dentro de `evaluar_red_neuronal()` (diferencia máxima entre ejecuciones: `0.00000000`).

---

## 7. Resultados Reales Verificados

### Evaluación Comparativa sobre Test No Visto (`X_test`: 287 fragancias × 100 variables)

| Modelo | MSE | RMSE (USD) | MAE (USD) | $R^2$ | Train $R^2$ | Brecha $R^2$ | Pred. `< 0` | Tiempo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Baseline ingenuo** *(media Train = 86.77 USD)* | 4907.91 | 70.06 | 53.56 | -0.0012 | — | — | 0 | — |
| **Regresión Lineal Múltiple (`Scikit-Learn`)** | 1325.13 | 36.40 | 24.76 | 0.7297 | 0.7072 | -0.0225 | 8 | ~0.05–0.07 s |
| **Red Neuronal MLP (`TensorFlow/Keras`)** 🏆 | **1145.83** | **33.85** | **21.02** | **0.7662** | 0.8248 | +0.0586 | **0** | ~10.1–11.5 s |

### Evolución del Aprendizaje (Validación Interna — sin tocar `X_test`)

- **Red Neuronal MLP (época por época, `validation_split=0.20`):** ejecutó `72` de `200` épocas; el mejor punto guardado por `EarlyStopping` fue la **época 52**:
  - Época 1: `Train MSE = 10519.08` (`MAE = 82.23`) | **`Val MSE = 9301.32`** (`Val MAE = 73.63` USD)
  - Época 25: `Train MSE = 906.70` (`MAE = 18.77`) | **`Val MSE = 925.25`** (`Val MAE = 19.47` USD)
  - **Época 52 (Mejor Validación):** `Train MSE = 701.66` (`MAE = 16.47`) | **`Val MSE = 876.19`** (`Val RMSE = 29.60`, `Val MAE = 18.59` USD)
  - Época 72 (cierre): `Train MSE = 671.95` | **`Val MSE = 892.06`** (`Val MAE = 19.03` USD)
- **Regresión Lineal Múltiple (curva de aprendizaje incremental sobre el mismo holdout interno 20%):**
  - `20%` (`183` muestras): `Val MSE = 1631.56` | `Val RMSE = 40.39` | `Val MAE = 28.16` | `Val R² = 0.5855`
  - `60%` (`549` muestras): `Val MSE = 1226.32` | `Val RMSE = 35.02` | `Val MAE = 23.72` | `Val R² = 0.6885`
  - `100%` (`916` muestras): `Val MSE = 1137.80` | `Val RMSE = 33.73` | `Val MAE = 22.71` | `Val R² = 0.7110`

---

## 8. Conclusión Técnica

El mejor desempeño predictivo en el conjunto de prueba no visto lo obtuvo la **Red Neuronal MLP (`TensorFlow/Keras`)**:

1. **Mayor precisión y reducción de error:** alcanza **$R^2 = 0.7662$**, superando a la Regresión Lineal Múltiple ($R^2 = 0.7297$) con una mejora de **`+0.0366`** en varianza explicada, **`-15.09%`** en `MAE` (`21.02` vs. `24.76` USD) y **`-13.53%`** en `MSE` (`1145.83` vs. `1325.13`).
2. **Captura de no linealidades:** el precio en perfumería depende de interacciones entre marca y formato/subcategoría (el salto de un tamaño viaje a un perfume estándar varía según sea marca de lujo o comercial), relaciones que las capas ocultas `ReLU` modelan sin ingeniería manual de cruces.
3. **Coherencia económica:** mientras la ecuación lineal irrestricta generó `8` predicciones negativas en combinaciones económicas (mínimo `-9.88` USD), la Red Neuronal mantuvo todas sus predicciones en un rango físicamente válido (`17.41` a `302.20` USD, `0` negativas).
4. **Compromiso (*trade-off*):** la Regresión Lineal sigue siendo valiosa por su interpretabilidad directa en USD y velocidad (~`0.05–0.07 s` frente a ~`10–11.5 s`), mientras que la Red Neuronal ofrece la mejor capacidad predictiva.
