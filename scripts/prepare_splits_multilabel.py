"""
scripts/prepare_splits_multilabel.py

Genera splits train/val/test para clasificación multi-label (14 clases).

Diferencias respecto a prepare_splits_s4.py (4 clases):
  - NO hay balance de clases (todas las imágenes se usan).
  - El label de cada imagen es un vector binario de 14 posiciones.
  - Split estratificado por presencia/ausencia de la clase más rara.
  - Train: TODAS las imágenes del paciente seleccionado.
  - Val/Test: 1 imagen/paciente.

Con 102,120 imágenes disponibles, train será ~70,000+ imágenes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from src.training.utils import set_seed, get_logger, load_config

CLASSES_14 = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion",    "Emphysema",    "Fibrosis",      "Hernia",
    "Infiltration","Mass",         "Nodule",         "Pleural_Thickening",
    "Pneumonia",   "Pneumothorax",
]


def parse_multilabel(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas binarias para cada una de las 14 clases."""
    df = df.copy()
    for cls in CLASSES_14:
        df[cls] = df["Finding Labels"].str.contains(cls).astype(int)
    df["is_no_finding"] = (df["Finding Labels"] == "No Finding").astype(int)
    return df


def filter_views(df: pd.DataFrame, views: list) -> pd.DataFrame:
    return df[df["View Position"].isin(views)].copy()


def get_patient_stratification_label(df: pd.DataFrame) -> pd.Series:
    """
    Para estratificar el split, usamos si el paciente tiene
    al menos una patología (1) o no (0).
    Esto mantiene la proporción de casos positivos/negativos.
    """
    def patient_label(group):
        # 1 si tiene alguna patología, 0 si es puro "No Finding"
        return int((group["is_no_finding"] == 0).any())

    return df.groupby("Patient ID").apply(patient_label)


def stratified_split_patients(patient_labels: pd.Series,
                               train_ratio: float,
                               val_ratio: float,
                               test_ratio: float,
                               seed: int,
                               logger) -> tuple[set, set, set]:
    X = patient_labels.index.to_numpy()
    y = patient_labels.values

    # Separar test
    n_splits_test = max(2, round(1.0 / test_ratio))
    skf1 = StratifiedKFold(n_splits=n_splits_test, shuffle=True, random_state=seed)
    tv_idx, test_idx = next(skf1.split(X, y))

    test_pats = set(X[test_idx])
    X_tv, y_tv = X[tv_idx], y[tv_idx]

    # Separar val
    val_rel = val_ratio / (train_ratio + val_ratio)
    n_splits_val = max(2, round(1.0 / val_rel))
    skf2 = StratifiedKFold(n_splits=n_splits_val, shuffle=True, random_state=seed)
    tr_idx, val_idx = next(skf2.split(X_tv, y_tv))

    train_pats = set(X_tv[tr_idx])
    val_pats   = set(X_tv[val_idx])

    logger.info("Split: train=%d, val=%d, test=%d pacientes",
                len(train_pats), len(val_pats), len(test_pats))
    return train_pats, val_pats, test_pats


def sample_one_per_patient(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng  = np.random.default_rng(seed)
    rows = []
    for _, group in df.groupby("Patient ID"):
        idx = rng.integers(0, len(group))
        rows.append(group.iloc[idx])
    return pd.DataFrame(rows).reset_index(drop=True)


def report_split(name: str, df: pd.DataFrame, logger) -> None:
    n = len(df)
    n_pats = df["Patient ID"].nunique()
    n_nf   = (df["is_no_finding"] == 1).sum()
    logger.info("%s: %d imgs | %d pacientes | No Finding=%d (%.1f%%)",
                name, n, n_pats, n_nf, n_nf/n*100)
    for cls in CLASSES_14:
        count = df[cls].sum()
        logger.info("  %s: %d (%.1f%%)", cls, count, count/n*100)


def main(config_path=None):
    if config_path is None:
        config_path = CODE_DIR / "configs" / "sprint4_multilabel.yaml"
    cfg = load_config(config_path)
    set_seed(cfg.project["seed"])

    log_file = Path(cfg.paths["logs"]) / "prepare_splits_ml.log"
    Path(cfg.paths["logs"]).mkdir(parents=True, exist_ok=True)
    logger = get_logger("prepare_splits_ml", log_file=log_file)

    logger.info("=" * 65)
    logger.info("PREPARE SPLITS Multi-label (14 clases NIH)")
    logger.info("=" * 65)

    # 1. Cargar CSV
    subset_csv = Path(cfg.paths["data_raw"]) / "Data_Entry_subset_local.csv"
    df = pd.read_csv(subset_csv)
    logger.info("CSV cargado: %d filas", len(df))

    # 2. Filtrar vistas
    df = filter_views(df, cfg.dataset["views"])
    logger.info("Tras filtrar vistas PA/AP: %d filas", len(df))

    # 3. Parsear labels multi-label
    df = parse_multilabel(df)

    # Estadísticas
    logger.info("Distribución de las 14 clases:")
    for cls in CLASSES_14:
        n   = df[cls].sum()
        n_p = df[df[cls] == 1]["Patient ID"].nunique()
        logger.info("  %-22s: %6d imgs | %5d pacientes", cls, n, n_p)
    logger.info("  No Finding: %d imgs", df["is_no_finding"].sum())

    # 4. Split estratificado por paciente
    patient_labels = get_patient_stratification_label(df)
    logger.info("Total pacientes únicos: %d", len(patient_labels))

    train_pats, val_pats, test_pats = stratified_split_patients(
        patient_labels,
        cfg.splits["train_ratio"],
        cfg.splits["val_ratio"],
        cfg.splits["test_ratio"],
        seed=cfg.project["seed"],
        logger=logger,
    )

    # Verificar no leakage
    assert not (train_pats & val_pats),  "Leakage train-val"
    assert not (train_pats & test_pats), "Leakage train-test"
    assert not (val_pats   & test_pats), "Leakage val-test"
    logger.info("Sin leakage entre splits.")

    # 5. Construir DataFrames
    # Train: TODAS las imágenes del paciente
    df_train = df[df["Patient ID"].isin(train_pats)].reset_index(drop=True)

    # Val/Test: 1 imagen/paciente
    df_val  = sample_one_per_patient(
        df[df["Patient ID"].isin(val_pats)],
        seed=cfg.project["seed"] + 1
    )
    df_test = sample_one_per_patient(
        df[df["Patient ID"].isin(test_pats)],
        seed=cfg.project["seed"] + 2
    )

    # 6. Reportar
    logger.info("-" * 65)
    report_split("TRAIN", df_train, logger)
    report_split("VAL",   df_val,   logger)
    report_split("TEST",  df_test,  logger)

    # 7. Guardar
    processed_dir = Path(cfg.paths["data_processed"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Columnas a guardar
    base_cols = ["Image Index", "Finding Labels", "Patient ID",
                 "View Position", "is_no_finding"]
    all_cols  = base_cols + CLASSES_14

    df_train[all_cols].to_csv(processed_dir / "train.csv", index=False)
    df_val[all_cols].to_csv(processed_dir   / "val.csv",   index=False)
    df_test[all_cols].to_csv(processed_dir  / "test.csv",  index=False)

    logger.info("-" * 65)
    logger.info("Splits guardados en: %s", processed_dir)
    logger.info("  train.csv: %d filas", len(df_train))
    logger.info("  val.csv:   %d filas", len(df_val))
    logger.info("  test.csv:  %d filas", len(df_test))
    logger.info("  Ratio train/S2: %.1fx", len(df_train) / 739)
    logger.info("=" * 65)
    logger.info("PREPARE SPLITS ML OK")


if __name__ == "__main__":
    main()
