"""Script 3: Red Neuronal Artificial con TensorFlow / Keras.

Construye, entrena y evalúa una arquitectura secuencial (MLP) para predecir
el precio en USD de fragancias en Sephora a partir de las matrices
preprocesadas en ``datos/``, manteniendo determinismo y evaluando con
exactamente las mismas métricas que ``2_modelo_clasico.py``.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, layers, regularizers

tf.get_logger().setLevel("ERROR")

BASE_DIR = Path(__file__).resolve().parent
DATOS_DIR = BASE_DIR / "datos"
TARGET_COL = "price_usd"

RANDOM_STATE = 412
VALIDATION_SPLIT = 0.20
EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.003
L2_REG = 1e-2
DROPOUT_RATE = 0.10
EARLY_STOPPING_PATIENCE = 20
NOMBRE_MODELO = "Red Neuronal MLP (TensorFlow/Keras)"


def cargar_datos() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Carga los conjuntos preprocesados en formato ``float32`` para TensorFlow."""
    rutas = {
        "X_train": DATOS_DIR / "X_train.csv",
        "X_test": DATOS_DIR / "X_test.csv",
        "y_train": DATOS_DIR / "y_train.csv",
        "y_test": DATOS_DIR / "y_test.csv",
    }
    for nombre, ruta in rutas.items():
        if not ruta.exists():
            raise FileNotFoundError(
                f"No se encontró {nombre} en {ruta}. Ejecuta primero 1_preprocesamiento.py."
            )

    X_train = pd.read_csv(rutas["X_train"]).to_numpy(dtype=np.float32)
    X_test = pd.read_csv(rutas["X_test"]).to_numpy(dtype=np.float32)
    y_train = pd.read_csv(rutas["y_train"])[TARGET_COL].to_numpy(dtype=np.float32)
    y_test = pd.read_csv(rutas["y_test"])[TARGET_COL].to_numpy(dtype=np.float32)
    return X_train, X_test, y_train, y_test


def construir_red_neuronal(n_features: int) -> keras.Sequential:
    """Define y compila la arquitectura secuencial para regresión."""
    modelo = keras.Sequential(
        [
            layers.Input(shape=(n_features,)),
            layers.Dense(
                64,
                activation="relu",
                kernel_regularizer=regularizers.l2(L2_REG),
            ),
            layers.Dropout(DROPOUT_RATE),
            layers.Dense(
                32,
                activation="relu",
                kernel_regularizer=regularizers.l2(L2_REG),
            ),
            layers.Dropout(DROPOUT_RATE),
            layers.Dense(1, activation="linear"),
        ]
    )
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    return modelo


def entrenar_red_neuronal(
    X_train: np.ndarray, y_train: np.ndarray
) -> tuple[keras.Sequential, keras.callbacks.History, float]:
    """Entrena el MLP monitoreando ``validation_split`` con EarlyStopping."""
    keras.backend.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    tf.config.experimental.enable_op_determinism()

    modelo = construir_red_neuronal(X_train.shape[1])
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )

    inicio = time.perf_counter()
    history = modelo.fit(
        X_train,
        y_train,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=0,
    )
    tiempo_entrenamiento = time.perf_counter() - inicio
    return modelo, history, tiempo_entrenamiento


def _calcular_metricas(y_real: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula MSE, RMSE, MAE y R² para regresión."""
    mse = float(mean_squared_error(y_real, y_pred))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_real, y_pred)),
        "r2": float(r2_score(y_real, y_pred)),
    }


def _imprimir_reporte(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_pred_test: np.ndarray,
    modelo: keras.Sequential,
    history: keras.callbacks.History,
    metricas_train: dict[str, float],
    metricas_test: dict[str, float],
    metricas_baseline: dict[str, float],
    tiempo_entrenamiento: float,
) -> None:
    """Imprime arquitectura, convergencia de entrenamiento y métricas finales."""
    loss_hist = history.history["loss"]
    val_loss_hist = history.history["val_loss"]
    val_mae_hist = history.history["val_mae"]
    epocas_ejecutadas = len(loss_hist)
    mejor_epoca = int(np.argmin(val_loss_hist)) + 1
    brecha_r2 = metricas_train["r2"] - metricas_test["r2"]
    negativas = int((y_pred_test < 0).sum())

    print("=" * 72)
    print(f"FASE 3: {NOMBRE_MODELO.upper()}")
    print("=" * 72)
    print(f"Entrenamiento: {X_train.shape[0]} muestras x {X_train.shape[1]} variables (float32)")
    print(f"Prueba (Test): {X_test.shape[0]} muestras x {X_test.shape[1]} variables (float32)")
    print(
        "\nArquitectura Sequential:\n"
        f"  Input({X_train.shape[1]}) -> Dense(64, ReLU, L2={L2_REG}) -> Dropout({DROPOUT_RATE}) -> "
        f"Dense(32, ReLU, L2={L2_REG}) -> Dropout({DROPOUT_RATE}) -> Dense(1, linear)"
    )
    print(
        f"  Parámetros entrenables: {modelo.count_params()} | "
        f"Optimizador: Adam(lr={LEARNING_RATE}) | Pérdida: mse"
    )

    print("\nConvergencia del entrenamiento (validation_split=0.20, sin tocar X_test):")
    print(f"  Épocas ejecutadas: {epocas_ejecutadas}/{EPOCHS} | Mejor época (EarlyStopping): {mejor_epoca}")
    print(
        f"  loss (Train)     -> Época 1: {loss_hist[0]:.2f} | "
        f"Mejor época ({mejor_epoca}): {loss_hist[mejor_epoca - 1]:.2f}"
    )
    print(
        f"  val_loss (Val)   -> Época 1: {val_loss_hist[0]:.2f} | "
        f"Mejor época ({mejor_epoca}): {val_loss_hist[mejor_epoca - 1]:.2f}"
    )
    print(
        f"  val_mae (Val)    -> Época 1: {val_mae_hist[0]:.2f} USD | "
        f"Mejor época ({mejor_epoca}): {val_mae_hist[mejor_epoca - 1]:.2f} USD"
    )

    print(
        f"\nBaseline ingenuo (predecir media de Train = {float(y_train.mean()):.2f} USD):"
    )
    print(
        f"  MSE={metricas_baseline['mse']:.2f} | "
        f"RMSE={metricas_baseline['rmse']:.2f} USD | "
        f"MAE={metricas_baseline['mae']:.2f} USD | "
        f"R²={metricas_baseline['r2']:.4f}"
    )

    print("\nControl de Overfitting (Train completo vs. Test):")
    print(f"  {'Conjunto':<10} {'MSE':>10} {'RMSE (USD)':>12} {'MAE (USD)':>12} {'R²':>10}")
    print("  " + "-" * 56)
    print(
        f"  {'Train':<10} {metricas_train['mse']:>10.2f} "
        f"{metricas_train['rmse']:>12.2f} {metricas_train['mae']:>12.2f} "
        f"{metricas_train['r2']:>10.4f}"
    )
    print(
        f"  {'Test':<10} {metricas_test['mse']:>10.2f} "
        f"{metricas_test['rmse']:>12.2f} {metricas_test['mae']:>12.2f} "
        f"{metricas_test['r2']:>10.4f}"
    )
    print(f"  Brecha R² (Train - Test): {brecha_r2:+.4f}")

    print("\nRango de predicciones en Test:")
    print(
        f"  min={y_pred_test.min():.2f} | mediana={np.median(y_pred_test):.2f} | "
        f"max={y_pred_test.max():.2f} USD | predicciones < 0: {negativas}"
    )

    print("\nResumen final sobre Test:")
    print(
        f"  MSE={metricas_test['mse']:.4f} | RMSE={metricas_test['rmse']:.4f} USD | "
        f"MAE={metricas_test['mae']:.4f} USD | R²={metricas_test['r2']:.4f} | "
        f"Tiempo={tiempo_entrenamiento:.4f} s"
    )
    print("=" * 72)


def evaluar_red_neuronal(verbose: bool = True) -> dict[str, object]:
    """Entrena y evalúa la red neuronal retornando métricas estándar en Test."""
    # Fijar semilla dentro de la función garantiza reproducibilidad al importarla en 4_comparacion.py
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    X_train, X_test, y_train, y_test = cargar_datos()
    modelo, history, tiempo_entrenamiento = entrenar_red_neuronal(X_train, y_train)

    y_pred_train = modelo.predict(X_train, verbose=0).ravel().astype(float)
    y_pred_test = modelo.predict(X_test, verbose=0).ravel().astype(float)
    y_true_train = y_train.astype(float)
    y_true_test = y_test.astype(float)
    y_pred_baseline = np.full_like(y_true_test, fill_value=float(y_true_train.mean()))

    metricas_train = _calcular_metricas(y_true_train, y_pred_train)
    metricas_test = _calcular_metricas(y_true_test, y_pred_test)
    metricas_baseline = _calcular_metricas(y_true_test, y_pred_baseline)

    if verbose:
        _imprimir_reporte(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_pred_test=y_pred_test,
            modelo=modelo,
            history=history,
            metricas_train=metricas_train,
            metricas_test=metricas_test,
            metricas_baseline=metricas_baseline,
            tiempo_entrenamiento=tiempo_entrenamiento,
        )

    return {
        "modelo": NOMBRE_MODELO,
        "mse": metricas_test["mse"],
        "rmse": metricas_test["rmse"],
        "mae": metricas_test["mae"],
        "r2": metricas_test["r2"],
        "train_mse": metricas_train["mse"],
        "train_rmse": metricas_train["rmse"],
        "train_mae": metricas_train["mae"],
        "train_r2": metricas_train["r2"],
        "y_pred": y_pred_test,
        "tiempo_entrenamiento": tiempo_entrenamiento,
    }


if __name__ == "__main__":
    evaluar_red_neuronal(verbose=True)
