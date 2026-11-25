"""Script 4: Evaluación comparativa final (Scikit-Learn vs. TensorFlow/Keras).

Consolida la evaluación sobre el mismo conjunto de prueba no visto (``X_test``,
``y_test``), presenta la evolución de entrenamiento de ambos enfoques (época por
época en la Red Neuronal y curva de aprendizaje en Regresión Lineal), imprime un
cuadro comparativo lado a lado y formula la conclusión técnica justificada.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATOS_DIR = BASE_DIR / "datos"
TARGET_COL = "price_usd"


def _cargar_test_y_baseline() -> tuple[pd.DataFrame, np.ndarray, dict[str, object]]:
    """Carga ``X_test``, ``y_test`` e ``y_train`` para verificar dimensiones y baseline."""
    X_test = pd.read_csv(DATOS_DIR / "X_test.csv")
    y_train = pd.read_csv(DATOS_DIR / "y_train.csv")[TARGET_COL].to_numpy(dtype=float)
    y_test = pd.read_csv(DATOS_DIR / "y_test.csv")[TARGET_COL].to_numpy(dtype=float)

    y_pred_base = np.full_like(y_test, fill_value=float(y_train.mean()))
    mse_base = float(mean_squared_error(y_test, y_pred_base))
    baseline = {
        "modelo": f"Baseline ingenuo (media={y_train.mean():.2f})",
        "mse": mse_base,
        "rmse": float(np.sqrt(mse_base)),
        "mae": float(mean_absolute_error(y_test, y_pred_base)),
        "r2": float(r2_score(y_test, y_pred_base)),
        "train_r2": 0.0,
        "y_pred": y_pred_base,
        "tiempo_entrenamiento": 0.0,
    }
    return X_test, y_test, baseline


def _verificar_metricas_en_mismo_test(
    y_test: np.ndarray, resultado: dict[str, object]
) -> None:
    """Verifica independientemente que las métricas correspondan al mismo ``y_test``."""
    y_pred = np.asarray(resultado["y_pred"], dtype=float)
    mse_check = float(mean_squared_error(y_test, y_pred))
    rmse_check = float(np.sqrt(mse_check))
    mae_check = float(mean_absolute_error(y_test, y_pred))
    r2_check = float(r2_score(y_test, y_pred))

    for llave, valor_check in [
        ("mse", mse_check),
        ("rmse", rmse_check),
        ("mae", mae_check),
        ("r2", r2_check),
    ]:
        if not np.isclose(float(resultado[llave]), valor_check, rtol=1e-5, atol=1e-5):
            raise AssertionError(
                f"Discrepancia en '{llave}' para {resultado['modelo']}: "
                f"{resultado[llave]} vs {valor_check}"
            )


def _imprimir_evolucion_modelos(
    res_clasico: dict[str, object], res_nn: dict[str, object]
) -> None:
    """Imprime la evolución de aprendizaje época por época (NN) y curva (Lineal)."""
    hist_nn = res_nn["historial_aprendizaje"]
    print("\n" + "-" * 94)
    print(
        "1. EVOLUCIÓN ÉPOCA POR ÉPOCA — RED NEURONAL MLP (validation_split=0.20, sin tocar X_test)"
    )
    print(
        f"   Método: {hist_nn['tipo_entrenamiento']} | "
        f"Épocas: {hist_nn['epocas_ejecutadas']} | Mejor época: {hist_nn['mejor_epoca']}"
    )
    print("-" * 94)
    print(
        f"{'Época':>6} | {'Train MSE':>10} | {'Train RMSE':>10} | {'Train MAE':>10} | "
        f"{'Val MSE':>10} | {'Val RMSE':>10} | {'Val MAE':>10} | {'Marca':<8}"
    )
    print("-" * 94)
    for reg in hist_nn["registros"]:
        marca = "<- MEJOR" if reg["es_mejor"] else ""
        print(
            f"{reg['epoca']:>6d} | {reg['train_mse']:>10.2f} | {reg['train_rmse']:>10.2f} | "
            f"{reg['train_mae']:>10.2f} | {reg['val_mse']:>10.2f} | {reg['val_rmse']:>10.2f} | "
            f"{reg['val_mae']:>10.2f} | {marca:<8}"
        )

    hist_cl = res_clasico["historial_aprendizaje"]
    print("\n" + "-" * 94)
    print(
        "2. EVOLUCIÓN DEL APRENDIZAJE — REGRESIÓN LINEAL MÚLTIPLE (mismo holdout interno 20%)"
    )
    print(f"   Método: {hist_cl['tipo_entrenamiento']}")
    print(
        "   Nota: LinearRegression calcula el óptimo global en un solo paso matricial (sin épocas);"
    )
    print(
        "   su mejora al disponer de más datos se evalúa mediante esta curva de aprendizaje:"
    )
    print("-" * 94)
    print(
        f"{'Sub-train':>12} | {'Train MSE':>10} | {'Train RMSE':>10} | {'Train R²':>9} | "
        f"{'Val MSE':>10} | {'Val RMSE':>10} | {'Val MAE':>9} | {'Val R²':>8}"
    )
    print("-" * 94)
    for reg in hist_cl["registros"]:
        etiqueta = f"{int(reg['fraccion'] * 100):>3d}% ({reg['n_muestras']:>3d})"
        print(
            f"{etiqueta:>12} | {reg['train_mse']:>10.2f} | {reg['train_rmse']:>10.2f} | "
            f"{reg['train_r2']:>9.4f} | {reg['val_mse']:>10.2f} | {reg['val_rmse']:>10.2f} | "
            f"{reg['val_mae']:>9.2f} | {reg['val_r2']:>8.4f}"
        )


def _imprimir_tabla_comparativa(
    X_test: pd.DataFrame,
    baseline: dict[str, object],
    res_clasico: dict[str, object],
    res_nn: dict[str, object],
    ganador: str,
) -> None:
    """Imprime la tabla lado a lado sobre el mismo Test Set y la conclusión técnica."""
    print("\n" + "=" * 98)
    print(
        f"3. CUADRO COMPARATIVO FINAL EN TEST NO VISTO "
        f"(X_test: {X_test.shape[0]} muestras x {X_test.shape[1]} variables)"
    )
    print("=" * 98)
    print(
        f"{'Modelo':<40} | {'MSE':>8} | {'RMSE':>7} | {'MAE':>7} | "
        f"{'R²':>7} | {'Train R²':>8} | {'Brecha':>7} | {'<0':>3} | {'Tiempo':>8}"
    )
    print("-" * 98)

    filas = [baseline, res_clasico, res_nn]
    for fila in filas:
        y_pred = np.asarray(fila["y_pred"], dtype=float)
        neg = int((y_pred < 0).sum())
        if fila is baseline:
            tr_r2_txt = "   ---  "
            brecha_txt = "  ---  "
            t_txt = "   ---  "
        else:
            tr_r2 = float(fila["train_r2"])
            brecha = tr_r2 - float(fila["r2"])
            tr_r2_txt = f"{tr_r2:>8.4f}"
            brecha_txt = f"{brecha:>+7.4f}"
            t_txt = f"{float(fila['tiempo_entrenamiento']):>6.4f} s"
        print(
            f"{str(fila['modelo']):<40} | {float(fila['mse']):>8.2f} | "
            f"{float(fila['rmse']):>7.2f} | {float(fila['mae']):>7.2f} | "
            f"{float(fila['r2']):>7.4f} | {tr_r2_txt} | {brecha_txt} | {neg:>3d} | {t_txt}"
        )
    print("=" * 98)

    d_mse = float(res_nn["mse"]) - float(res_clasico["mse"])
    d_rmse = float(res_nn["rmse"]) - float(res_clasico["rmse"])
    d_mae = float(res_nn["mae"]) - float(res_clasico["mae"])
    d_r2 = float(res_nn["r2"]) - float(res_clasico["r2"])

    mejora_mse_pct = (1.0 - float(res_nn["mse"]) / float(res_clasico["mse"])) * 100.0
    mejora_rmse_pct = (1.0 - float(res_nn["rmse"]) / float(res_clasico["rmse"])) * 100.0
    mejora_mae_pct = (1.0 - float(res_nn["mae"]) / float(res_clasico["mae"])) * 100.0

    neg_cl = int((np.asarray(res_clasico["y_pred"]) < 0).sum())
    neg_nn = int((np.asarray(res_nn["y_pred"]) < 0).sum())

    print("\nDiferencias directas (Red Neuronal vs. Regresión Lineal en Test):")
    print(
        f"  Δ MSE  = {d_mse:+.2f} ({mejora_mse_pct:+.2f}% de reducción de error cuadrático)"
    )
    print(
        f"  Δ RMSE = {d_rmse:+.2f} USD ({mejora_rmse_pct:+.2f}% de mejora)"
    )
    print(
        f"  Δ MAE  = {d_mae:+.2f} USD ({mejora_mae_pct:+.2f}% de mejora)"
    )
    print(f"  Δ R²   = {d_r2:+.4f} puntos de varianza explicada")

    print("\n" + "=" * 98)
    print("4. CONCLUSIÓN TÉCNICA JUSTIFICADA")
    print("=" * 98)
    print(f"Modelo ganador en Test: {ganador}")
    if ganador == res_nn["modelo"]:
        print(
            f"  1) Desempeño predictivo superior: La Red Neuronal MLP alcanza R² = {res_nn['r2']:.4f} "
            f"(RMSE = {res_nn['rmse']:.2f} USD, MAE = {res_nn['mae']:.2f} USD), superando a la "
            f"Regresión Lineal Múltiple (R² = {res_clasico['r2']:.4f}, RMSE = {res_clasico['rmse']:.2f} USD, "
            f"MAE = {res_clasico['mae']:.2f} USD) con una reducción de MAE de {abs(d_mae):.2f} USD "
            f"({mejora_mae_pct:.1f}%) y de MSE del {mejora_mse_pct:.1f}%."
        )
        print(
            "  2) Causa estructural (no linealidad): El precio en perfumería no es puramente aditivo; "
            "existen interacciones multiplicativas entre marca y formato/subcategoría (ej. el salto de "
            "tamaño viaje a frasco estándar difiere mucho entre marcas de lujo y comerciales). Las "
            "capas ocultas ReLU (64 y 32 neuronas) modelan esas interacciones automáticamente."
        )
        print(
            f"  3) Coherencia física y control de sobreajuste: Con regularización L2 (0.01), Dropout (0.10) "
            f"y EarlyStopping, la red mantuvo una brecha controlada (Train R²={res_nn['train_r2']:.4f} vs. "
            f"Test R²={res_nn['r2']:.4f}) y produjo {neg_nn} predicciones negativas (frente a {neg_cl} "
            "predicciones negativas en la ecuación lineal irrestricta)."
        )
        print(
            f"  4) Compromiso (trade-off): La Regresión Lineal entrenó ~"
            f"{float(res_nn['tiempo_entrenamiento']) / max(float(res_clasico['tiempo_entrenamiento']), 1e-6):.0f}x "
            f"más rápido ({res_clasico['tiempo_entrenamiento']:.4f} s vs. {res_nn['tiempo_entrenamiento']:.2f} s) "
            "y ofrece interpretabilidad directa por coeficiente en USD, por lo que sigue siendo un "
            "excelente modelo base explicativo, mientras que la Red Neuronal es la mejor opción para "
            "precisión predictiva."
        )
    else:
        print(
            f"  La Regresión Lineal Múltiple obtuvo mejor desempeño en Test (R²={res_clasico['r2']:.4f} "
            f"vs. {res_nn['r2']:.4f}) con entrenamiento más rápido e interpretabilidad directa en USD."
        )
    print("=" * 98)


def comparar_modelos(verbose: bool = True) -> dict[str, object]:
    """Ejecuta ambos modelos sobre el mismo Test Set y consolida la comparación."""
    X_test, y_test, baseline = _cargar_test_y_baseline()

    mod_clasico = importlib.import_module("2_modelo_clasico")
    mod_nn = importlib.import_module("3_red_neuronal")

    res_clasico = mod_clasico.evaluar_modelo_clasico(verbose=False)
    res_nn = mod_nn.evaluar_red_neuronal(verbose=False)

    _verificar_metricas_en_mismo_test(y_test, res_clasico)
    _verificar_metricas_en_mismo_test(y_test, res_nn)

    if float(res_nn["r2"]) >= float(res_clasico["r2"]) and float(res_nn["rmse"]) <= float(res_clasico["rmse"]):
        ganador = str(res_nn["modelo"])
    else:
        ganador = str(res_clasico["modelo"])

    if verbose:
        print("=" * 98)
        print("TALLER FINAL INTEGRADOR DE MACHINE LEARNING — COMPARACIÓN DE MODELOS")
        print("=" * 98)
        print(
            "Pregunta de negocio: ¿Con qué precisión se puede predecir el precio en USD (price_usd) "
            "de una fragancia en Sephora según su marca, subcategoría, tipo de variación, popularidad "
            "y atributos comerciales?"
        )
        print(
            f"Evaluación unificada sobre Test Set no visto: {X_test.shape[0]} muestras x {X_test.shape[1]} variables."
        )
        print(
            "Nota métrica: Al ser un problema de REGRESIÓN continua (price_usd en USD), se evalúa con "
            "MSE, RMSE, MAE y R² (Accuracy, Precision, Recall y F1 aplican exclusivamente a clasificación)."
        )
        _imprimir_evolucion_modelos(res_clasico, res_nn)
        _imprimir_tabla_comparativa(X_test, baseline, res_clasico, res_nn, ganador)

    return {
        "baseline": baseline,
        "clasico": res_clasico,
        "red_neuronal": res_nn,
        "ganador": ganador,
    }


if __name__ == "__main__":
    comparar_modelos(verbose=True)
