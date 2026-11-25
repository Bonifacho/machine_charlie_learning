"""Script 1: Preprocesamiento modular del catálogo de fragancias Sephora.

Transforma el conjunto crudo ``datos/sephora_fragrance.csv`` en matrices
limpias, imputadas, codificadas y escaladas listas para Scikit-Learn y
TensorFlow/Keras, evitando fuga de datos (*data leakage*) al ajustar todos
los transformadores exclusivamente sobre el subconjunto de entrenamiento.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATOS_DIR = BASE_DIR / "datos"
DATASET_ORIGINAL = DATOS_DIR / "sephora_fragrance.csv"

RANDOM_STATE = 412
TEST_SIZE = 0.20
TARGET_COL = "price_usd"

# Protección defensiva frente a identificadores o precios derivados (data leakage)
COLUMNAS_EXCLUIR = [
    "product_id",
    "brand_id",
    "product_name",
    "value_price_usd",
    "sale_price_usd",
    "child_max_price",
    "child_min_price",
]

NUMERIC_COLS = [
    "loves_count",
    "rating",
    "reviews",
    "child_count",
    "limited_edition",
    "new",
    "online_only",
    "sephora_exclusive",
]

CATEGORICAL_COLS = [
    "brand_name",
    "secondary_category",
    "tertiary_category",
    "variation_type",
]


def _cargar_y_validar_datos(ruta_csv: Path) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Carga el CSV original, descarta columnas prohibidas y valida el esquema."""
    if not ruta_csv.exists():
        raise FileNotFoundError(f"No se encontró el dataset original en: {ruta_csv}")

    df = pd.read_csv(ruta_csv)
    excluidas = [col for col in COLUMNAS_EXCLUIR if col in df.columns]
    if excluidas:
        df = df.drop(columns=excluidas)

    requeridas = NUMERIC_COLS + CATEGORICAL_COLS + [TARGET_COL]
    faltantes = [col for col in requeridas if col not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas en el dataset: {faltantes}")

    if df[TARGET_COL].isna().any():
        raise ValueError(f"La variable objetivo '{TARGET_COL}' contiene valores nulos.")

    X = df[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    y = df[TARGET_COL].astype(float).copy()
    return X, y, excluidas


def _normalizar_categoricas(df_cat: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas de texto a dtype object con np.nan para compatibilidad."""
    # En Pandas 3.x las cadenas usan StringDtype/pd.NA; SimpleImputer opera de forma
    # uniforme sobre object + np.nan sin generar advertencias de tipo.
    normalizado = df_cat.astype(object)
    return normalizado.where(pd.notna(normalizado), np.nan)


def _imputar_sin_leakage(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    """Ajusta SimpleImputer (mediana y moda) solo sobre Train y transforma ambos."""
    imp_num = SimpleImputer(strategy="median")
    X_train_num = pd.DataFrame(
        imp_num.fit_transform(X_train[NUMERIC_COLS]),
        columns=NUMERIC_COLS,
        index=X_train.index,
    )
    X_test_num = pd.DataFrame(
        imp_num.transform(X_test[NUMERIC_COLS]),
        columns=NUMERIC_COLS,
        index=X_test.index,
    )
    medianas = {
        col: float(val) for col, val in zip(NUMERIC_COLS, imp_num.statistics_)
    }

    imp_cat = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    train_cat_raw = _normalizar_categoricas(X_train[CATEGORICAL_COLS])
    test_cat_raw = _normalizar_categoricas(X_test[CATEGORICAL_COLS])

    X_train_cat = pd.DataFrame(
        imp_cat.fit_transform(train_cat_raw),
        columns=CATEGORICAL_COLS,
        index=X_train.index,
    )
    X_test_cat = pd.DataFrame(
        imp_cat.transform(test_cat_raw),
        columns=CATEGORICAL_COLS,
        index=X_test.index,
    )
    modas = {col: str(val) for col, val in zip(CATEGORICAL_COLS, imp_cat.statistics_)}

    return X_train_num, X_test_num, X_train_cat, X_test_cat, medianas, modas


def _codificar_categoricas(
    X_train_cat: pd.DataFrame, X_test_cat: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """Aplica One-Hot Encoding consistente usando solo categorías vistas en Train."""
    train_cat = X_train_cat.copy()
    test_cat = X_test_cat.copy()
    no_vistas: dict[str, list[str]] = {}

    for col in CATEGORICAL_COLS:
        categorias_train = sorted(train_cat[col].astype(str).unique().tolist())
        train_vals = train_cat[col].astype(str)
        test_vals = test_cat[col].astype(str)
        desconocidas = sorted(set(test_vals.unique()) - set(categorias_train))
        if desconocidas:
            no_vistas[col] = desconocidas

        # Mapear categorías no vistas en Train a NaN antes de construir Categorical evita
        # Pandas4Warning y produce fila de ceros en pd.get_dummies(dummy_na=False).
        test_vals_limpio = test_vals.where(test_vals.isin(categorias_train), np.nan)
        train_cat[col] = pd.Categorical(
            train_vals.tolist(), categories=categorias_train
        )
        test_cat[col] = pd.Categorical(
            test_vals_limpio.tolist(), categories=categorias_train
        )

    X_train_ohe = pd.get_dummies(train_cat, drop_first=True, dtype=int)
    X_test_ohe = pd.get_dummies(test_cat, drop_first=True, dtype=int).reindex(
        columns=X_train_ohe.columns, fill_value=0
    )
    return X_train_ohe, X_test_ohe, no_vistas


def _escalar_numericas(
    X_train_num: pd.DataFrame, X_test_num: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ajusta StandardScaler solo sobre Train y transforma Train y Test."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_num),
        columns=NUMERIC_COLS,
        index=X_train_num.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_num),
        columns=NUMERIC_COLS,
        index=X_test_num.index,
    )
    return X_train_scaled, X_test_scaled


def ejecutar_preprocesamiento(verbose: bool = True) -> dict[str, object]:
    """Ejecuta el pipeline completo de preprocesamiento y exporta los 4 CSVs."""
    X, y, excluidas = _cargar_y_validar_datos(DATASET_ORIGINAL)

    if verbose:
        print("=" * 70)
        print("FASE 1: PREPROCESAMIENTO DE DATOS (SEPHORA FRAGRANCE)")
        print("=" * 70)
        print(f"Dataset original: {DATASET_ORIGINAL}")
        print(f"Registros totales: {len(X)} | Variables de entrada X: {X.shape[1]}")
        print(f"Columnas ID/leakage excluidas en carga: {len(excluidas)} {excluidas}")
        print(
            f"Nulos totales en X antes de imputar: {int(X.isna().sum().sum())} "
            f"({X.isna().sum().sum() / X.size * 100:.2f}%)"
        )
        print(
            f"Variable objetivo '{TARGET_COL}' (regresión continua, sin mapeo de clases): "
            f"min={y.min():.2f}, mediana={y.median():.2f}, "
            f"media={y.mean():.2f}, max={y.max():.2f} USD"
        )

    # Dividir antes de imputar, codificar o escalar evita fuga de información de Test a Train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    X_train_num, X_test_num, X_train_cat, X_test_cat, medianas, modas = (
        _imputar_sin_leakage(X_train, X_test)
    )
    X_train_ohe, X_test_ohe, no_vistas = _codificar_categoricas(X_train_cat, X_test_cat)
    X_train_num_scaled, X_test_num_scaled = _escalar_numericas(X_train_num, X_test_num)

    X_train_final = pd.concat([X_train_num_scaled, X_train_ohe], axis=1)
    X_test_final = pd.concat([X_test_num_scaled, X_test_ohe], axis=1)

    DATOS_DIR.mkdir(parents=True, exist_ok=True)
    rutas = {
        "X_train": DATOS_DIR / "X_train.csv",
        "X_test": DATOS_DIR / "X_test.csv",
        "y_train": DATOS_DIR / "y_train.csv",
        "y_test": DATOS_DIR / "y_test.csv",
    }
    X_train_final.to_csv(rutas["X_train"], index=False, encoding="utf-8")
    X_test_final.to_csv(rutas["X_test"], index=False, encoding="utf-8")
    y_train.to_frame(name=TARGET_COL).to_csv(
        rutas["y_train"], index=False, encoding="utf-8"
    )
    y_test.to_frame(name=TARGET_COL).to_csv(
        rutas["y_test"], index=False, encoding="utf-8"
    )

    if verbose:
        print(
            f"\nSplit 80/20 (random_state={RANDOM_STATE}): "
            f"Train={len(X_train_final)} filas ({len(X_train_final)/len(X)*100:.2f}%), "
            f"Test={len(X_test_final)} filas ({len(X_test_final)/len(X)*100:.2f}%)"
        )
        print("\nMedianas aprendidas en Train (SimpleImputer numéricas):")
        for col, val in medianas.items():
            print(f"  - {col}: {val:.4f}")
        print("\nModas aprendidas en Train (SimpleImputer categóricas):")
        for col, val in modas.items():
            print(f"  - {col}: {val}")
        print("\nOne-Hot Encoding consistente (drop_first=True):")
        print(f"  - Columnas numéricas escaladas: {len(NUMERIC_COLS)}")
        print(f"  - Columnas dummy categóricas:  {X_train_ohe.shape[1]}")
        print(f"  - Total columnas finales:      {X_train_final.shape[1]}")
        if no_vistas:
            print("  - Categorías presentes en Test no vistas en Train (codificadas en 0):")
            for col, cats in no_vistas.items():
                print(f"      * {col} ({len(cats)}): {cats}")
        else:
            print("  - Categorías no vistas en Test: 0")
        print("\nArchivos exportados:")
        for nombre, ruta in rutas.items():
            print(f"  - {nombre}: {ruta}")
        print("=" * 70)

    return {
        "X_train_shape": X_train_final.shape,
        "X_test_shape": X_test_final.shape,
        "y_train_shape": (len(y_train), 1),
        "y_test_shape": (len(y_test), 1),
        "n_ohe_cols": X_train_ohe.shape[1],
        "no_vistas": no_vistas,
        "rutas": {k: str(v) for k, v in rutas.items()},
    }


if __name__ == "__main__":
    ejecutar_preprocesamiento(verbose=True)
