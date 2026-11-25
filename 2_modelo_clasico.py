"""Script 2: Modelo clásico de Machine Learning (Regresión Lineal Múltiple).

Implementa, entrena y evalúa ``LinearRegression`` de Scikit-Learn sobre las
matrices preprocesadas en ``datos/``, reportando métricas completas en Train
y Test, comparación contra el baseline ingenuo e interpretabilidad de
coeficientes en USD.
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATOS_DIR = BASE_DIR / "datos"
TARGET_COL = "price_usd"
NOMBRE_MODELO = "Regresión Lineal Múltiple (Scikit-Learn)"


def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Carga los conjuntos preprocesados desde la carpeta ``datos/``."""
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

    X_train = pd.read_csv(rutas["X_train"])
    X_test = pd.read_csv(rutas["X_test"])
    y_train = pd.read_csv(rutas["y_train"])[TARGET_COL].to_numpy(dtype=float)
    y_test = pd.read_csv(rutas["y_test"])[TARGET_COL].to_numpy(dtype=float)
    return X_train, X_test, y_train, y_test


def entrenar_modelo_clasico(
    X_train: pd.DataFrame, y_train: np.ndarray
) -> tuple[LinearRegression, float]:
    """Instancia y entrena la Regresión Lineal Múltiple midiendo su tiempo."""
    modelo = LinearRegression()
    inicio = time.perf_counter()
    modelo.fit(X_train, y_train)
    tiempo_entrenamiento = time.perf_counter() - inicio
    return modelo, tiempo_entrenamiento


def _calcular_metricas(y_real: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula MSE, RMSE, MAE y R² para regresión."""
    mse = float(mean_squared_error(y_real, y_pred))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_real, y_pred)),
        "r2": float(r2_score(y_real, y_pred)),
    }


def _calcular_curva_aprendizaje(
    X_train: pd.DataFrame, y_train: np.ndarray, val_split: float = 0.20
) -> dict[str, object]:
    """Evalúa curva de aprendizaje sobre el mismo holdout interno usado en Keras."""
    n_val = int(len(X_train) * val_split)
    X_sub = X_train.iloc[:-n_val]
    y_sub = y_train[:-n_val]
    X_val = X_train.iloc[-n_val:]
    y_val = y_train[-n_val:]

    registros: list[dict[str, float | int]] = []
    for fraccion in (0.20, 0.40, 0.60, 0.80, 1.00):
        n_muestras = max(1, int(len(X_sub) * fraccion))
        modelo_parcial = LinearRegression()
        modelo_parcial.fit(X_sub.iloc[:n_muestras], y_sub[:n_muestras])

        m_tr = _calcular_metricas(
            y_sub[:n_muestras], modelo_parcial.predict(X_sub.iloc[:n_muestras])
        )
        m_val = _calcular_metricas(y_val, modelo_parcial.predict(X_val))
        registros.append(
            {
                "fraccion": float(fraccion),
                "n_muestras": n_muestras,
                "train_mse": m_tr["mse"],
                "train_rmse": m_tr["rmse"],
                "train_mae": m_tr["mae"],
                "train_r2": m_tr["r2"],
                "val_mse": m_val["mse"],
                "val_rmse": m_val["rmse"],
                "val_mae": m_val["mae"],
                "val_r2": m_val["r2"],
            }
        )

    return {
        "tipo_entrenamiento": (
            "Solución analítica cerrada (Mínimos Cuadrados / SVD en 1 paso, sin épocas)"
        ),
        "registros": registros,
    }


def _imprimir_reporte(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_pred_test: np.ndarray,
    modelo: LinearRegression,
    metricas_train: dict[str, float],
    metricas_test: dict[str, float],
    metricas_baseline: dict[str, float],
    historial_aprendizaje: dict[str, object],
    tiempo_entrenamiento: float,
) -> None:
    """Imprime en consola el análisis completo de desempeño e interpretación."""
    brecha_r2 = metricas_train["r2"] - metricas_test["r2"]
    coefs = pd.Series(modelo.coef_, index=X_train.columns).sort_values(ascending=False)
    negativas = int((y_pred_test < 0).sum())

    print("=" * 72)
    print(f"FASE 2: {NOMBRE_MODELO.upper()}")
    print("=" * 72)
    print(f"Entrenamiento: {X_train.shape[0]} muestras x {X_train.shape[1]} variables")
    print(f"Prueba (Test): {X_test.shape[0]} muestras x {X_test.shape[1]} variables")

    print("\nCurva de aprendizaje (LinearRegression resuelve óptimo global en 1 paso analítico):")
    print(
        f"  {'Sub-train':>10} {'Train RMSE':>12} {'Val MSE':>10} "
        f"{'Val RMSE':>10} {'Val MAE':>10} {'Val R²':>9}"
    )
    print("  " + "-" * 65)
    for reg in historial_aprendizaje["registros"]:
        print(
            f"  {int(reg['fraccion']*100):>3d}% ({reg['n_muestras']:>3d}) "
            f"{reg['train_rmse']:>12.2f} {reg['val_mse']:>10.2f} "
            f"{reg['val_rmse']:>10.2f} {reg['val_mae']:>10.2f} {reg['val_r2']:>9.4f}"
        )

    print(
        f"\nBaseline ingenuo (predecir media de Train = {y_train.mean():.2f} USD):"
    )
    print(
        f"  MSE={metricas_baseline['mse']:.2f} | "
        f"RMSE={metricas_baseline['rmse']:.2f} USD | "
        f"MAE={metricas_baseline['mae']:.2f} USD | "
        f"R²={metricas_baseline['r2']:.4f}"
    )

    print("\nControl de Overfitting (Train vs. Test):")
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

    print("\nInterpretación del modelo (coeficientes en USD):")
    print(f"  Intercepto (β0): {modelo.intercept_:+.2f} USD")
    print(
        "  (Nota: las 8 numéricas miden cambio por +1 desv. estándar; "
        "las dummy miden diferencia vs. su categoría de referencia)."
    )
    print("\n  Top 8 coeficientes más positivos:")
    for feat, val in coefs.head(8).items():
        print(f"    {feat:<48s} {val:>+9.2f} USD")
    print("\n  Top 8 coeficientes más negativos:")
    for feat, val in coefs.tail(8).iloc[::-1].items():
        print(f"    {feat:<48s} {val:>+9.2f} USD")

    print("\nRango de predicciones en Test:")
    print(
        f"  min={y_pred_test.min():.2f} | mediana={np.median(y_pred_test):.2f} | "
        f"max={y_pred_test.max():.2f} USD | predicciones < 0: {negativas}"
    )

    print("\nResumen final sobre Test:")
    print(
        f"  MSE={metricas_test['mse']:.4f} | RMSE={metricas_test['rmse']:.4f} USD | "
        f"MAE={metricas_test['mae']:.4f} USD | R²={metricas_test['r2']:.4f} | "
        f"Tiempo={tiempo_entrenamiento:.6f} s"
    )
    print("=" * 72)


def evaluar_modelo_clasico(verbose: bool = True) -> dict[str, object]:
    """Entrena y evalúa ``LinearRegression`` retornando métricas estándar en Test."""
    X_train, X_test, y_train, y_test = cargar_datos()
    modelo, tiempo_entrenamiento = entrenar_modelo_clasico(X_train, y_train)

    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)
    y_pred_baseline = np.full_like(y_test, fill_value=float(y_train.mean()))

    metricas_train = _calcular_metricas(y_train, y_pred_train)
    metricas_test = _calcular_metricas(y_test, y_pred_test)
    metricas_baseline = _calcular_metricas(y_test, y_pred_baseline)
    historial_aprendizaje = _calcular_curva_aprendizaje(X_train, y_train)

    if verbose:
        _imprimir_reporte(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_pred_test=y_pred_test,
            modelo=modelo,
            metricas_train=metricas_train,
            metricas_test=metricas_test,
            metricas_baseline=metricas_baseline,
            historial_aprendizaje=historial_aprendizaje,
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
        "historial_aprendizaje": historial_aprendizaje,
    }


if __name__ == "__main__":
    evaluar_modelo_clasico(verbose=True)
