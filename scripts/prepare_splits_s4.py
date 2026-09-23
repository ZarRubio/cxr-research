"""
scripts/prepare_splits_s4.py

Genera los splits para Sprint 4 con una estrategia de datos mejorada:

DIFERENCIA CLAVE respecto a Sprint 2:
  - Train: TODAS las imágenes disponibles de cada paciente seleccionado.
           No se limita a 1 imagen/paciente.
  - Val/Test: 1 imagen/paciente (igual que Sprint 2, para evaluación limpia).

Justificación:
  La regla de 1 imagen/paciente en train fue conservadora para el baseline.
  Para el CNN-ViT necesitamos más datos en train para que el ViT aprenda
  relaciones globales. Val y test mantienen 1 imagen/paciente para garantizar
  que la evaluación sea comparable con Sprint 2.

Resultado esperado:
  Sprint 2: ~739  imágenes train (1/paciente)
  Sprint 4: ~3,900 imágenes train (todas/paciente, estimado)
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


CLASS_NAMES = {
    0: "No Finding",
    1: "Cardiomegaly",
    2: "Effusion",
    3: "Infiltration",
}

LABEL_HIERARCHY = {
    "Infiltration": 3,
    "Effusion":     2,
    "Cardiomegaly": 1,
    "No Finding":   0,
}


def build_multiclass_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Asigna label por jerarquía clínica (igual que Sprint 2)."""
    df = df.copy()

    def assign(finding: str) -> float:
        for keyword, label in sorted(LABEL_HIERARCHY.items(),
                                     key=lambda x: -x[1]):
            if keyword == "No Finding":
                if finding == "No Finding":
                    return 0
            elif keyword in finding:
                return label
        return float("nan")

    df["label"] = df["Finding Labels"].apply(assign)
    return df


def filter_views(df: pd.DataFrame, views: list) -> pd.DataFrame:
    return df[df["View Position"].isin(views)].copy()


def build_patient_label(df: pd.DataFrame) -> pd.Series:
    """Label representativo por paciente (máximo = prioridad más alta)."""
    return df.groupby("Patient ID")["label"].max().astype(int)


def balance_patients(df: pd.DataFrame, ratio: float, seed: int,
                     logger) -> set:
    """
    Selecciona pacientes balanceados.
    Devuelve el set de Patient IDs seleccionados.
    """
    rng = np.random.default_rng(seed)
    patient_labels = build_patient_label(df)

    patients_by_class = {
        label: list(patient_labels[patient_labels == label].index)
        for label in [0, 1, 2, 3]
    }

    # Techo = mínimo entre clases positivas
    ceiling = min(len(patients_by_class[l]) for l in [1, 2, 3])
    logger.info("Techo de balance: %d pacientes por clase positiva", ceiling)

    selected = set()
    for label in [1, 2, 3]:
        pats = patients_by_class[label]
        rng.shuffle(pats)
        chosen = pats[:ceiling]
        selected.update(chosen)
        logger.info("  Clase %d (%s): %d/%d pacientes",
                    label, CLASS_NAMES[label], len(chosen), len(pats))

    # No Finding
    n_nf = min(len(patients_by_class[0]), int(ceiling * ratio))
    nf_pats = patients_by_class[0]
    rng.shuffle(nf_pats)
    selected.update(nf_pats[:n_nf])
    logger.info("  Clase 0 (No Finding): %d pacientes", n_nf)

    logger.info("Total pacientes seleccionados: %d", len(selected))
    return selected


def stratified_split_patients(df: pd.DataFrame, selected_patients: set,
                               train_ratio: float, val_ratio: float,
                               test_ratio: float, seed: int, logger
                               ) -> tuple[set, set, set]:
    """Split estratificado a nivel paciente."""
    patient_labels = build_patient_label(df)
    patient_labels = patient_labels[patient_labels.index.isin(selected_patients)]

    X = patient_labels.index.to_numpy()
    y = patient_labels.values

    # Separar test
    n_test = max(2, round(1.0 / test_ratio))
    skf1 = StratifiedKFold(n_splits=n_test, shuffle=True, random_state=seed)
    tv_idx, test_idx = next(skf1.split(X, y))

    test_pats = set(X[test_idx])
    X_tv, y_tv = X[tv_idx], y[tv_idx]

    # Separar val de train
    val_rel = val_ratio / (train_ratio + val_ratio)
    n_val = max(2, round(1.0 / val_rel))
    skf2 = StratifiedKFold(n_splits=n_val, shuffle=True, random_state=seed)
    tr_idx, val_idx = next(skf2.split(X_tv, y_tv))

    train_pats = set(X_tv[tr_idx])
    val_pats   = set(X_tv[val_idx])

    logger.info("Split pacientes: train=%d, val=%d, test=%d",
                len(train_pats), len(val_pats), len(test_pats))
    return train_pats, val_pats, test_pats


def sample_one_per_patient(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """1 imagen aleatoria por paciente (para val y test)."""
    rng = np.random.default_rng(seed)
    rows = []
    for _, group in df.groupby("Patient ID"):
        idx = rng.integers(0, len(group))
        rows.append(group.iloc[idx])
    return pd.DataFrame(rows).reset_index(drop=True)


def report_split(name: str, df: pd.DataFrame, logger) -> None:
    n = len(df)
    n_pat = df["Patient ID"].nunique()
    logger.info("%s: %d imgs | %d pacientes", name.upper(), n, n_pat)
    for label, cname in CLASS_NAMES.items():
        count = (df["label"] == label).sum()
        logger.info("  clase %d (%s): %d (%.1f%%)",
                    label, cname, count, count / n * 100 if n else 0)


def main(config_path=None):
    if config_path is None:
        config_path = CODE_DIR / "configs" / "sprint4.yaml"
    cfg = load_config(config_path)

    set_seed(cfg.project["seed"])
    log_file = Path(cfg.paths["logs"]) / "prepare_splits_s4.log"
    Path(cfg.paths["logs"]).mkdir(parents=True, exist_ok=True)
    logger = get_logger("prepare_splits_s4", log_file=log_file)

    logger.info("=" * 65)
    logger.info("PREPARE SPLITS Sprint 4 — Todas las imágenes en train")
    logger.info("=" * 65)

    # 1. Cargar CSV
    subset_csv = Path(cfg.paths["data_raw"]) / "Data_Entry_subset_local.csv"
    df = pd.read_csv(subset_csv)
    logger.info("CSV cargado: %d filas", len(df))

    # 2. Filtrar vistas y construir labels
    df = filter_views(df, cfg.dataset["views"])
    df = build_multiclass_labels(df)
    n_before = len(df)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    logger.info("Tras filtrar y etiquetar: %d filas (%d descartadas)",
                len(df), n_before - len(df))

    for label, cname in CLASS_NAMES.items():
        logger.info("  %s: %d imágenes, %d pacientes",
                    cname,
                    (df["label"] == label).sum(),
                    df[df["label"] == label]["Patient ID"].nunique())

    # 3. Seleccionar pacientes balanceados
    ratio = cfg.get("class_ratio", 2.0)
    selected_patients = balance_patients(
        df, ratio=ratio, seed=cfg.project["seed"], logger=logger
    )
    df_balanced = df[df["Patient ID"].isin(selected_patients)].copy()

    # 4. Split estratificado por paciente
    processed_dir = Path(cfg.paths["data_processed"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_pats, val_pats, test_pats = stratified_split_patients(
        df_balanced, selected_patients,
        cfg.splits["train_ratio"],
        cfg.splits["val_ratio"],
        cfg.splits["test_ratio"],
        seed=cfg.project["seed"],
        logger=logger,
    )

    # Verificar no leakage
    assert not (train_pats & val_pats), "Leakage train-val"
    assert not (train_pats & test_pats), "Leakage train-test"
    assert not (val_pats & test_pats), "Leakage val-test"
    logger.info("Sin leakage entre splits.")

    # 5. Construir DataFrames
    # TRAIN: TODAS las imágenes de los pacientes seleccionados
    df_train = df_balanced[df_balanced["Patient ID"].isin(train_pats)].reset_index(drop=True)

    # VAL/TEST: 1 imagen por paciente
    df_val_full  = df_balanced[df_balanced["Patient ID"].isin(val_pats)]
    df_test_full = df_balanced[df_balanced["Patient ID"].isin(test_pats)]
    df_val  = sample_one_per_patient(df_val_full,  seed=cfg.project["seed"] + 1)
    df_test = sample_one_per_patient(df_test_full, seed=cfg.project["seed"] + 2)

    # 6. Reportar
    logger.info("-" * 65)
    report_split("TRAIN (todas las imágenes/paciente)", df_train, logger)
    report_split("VAL   (1 imagen/paciente)", df_val, logger)
    report_split("TEST  (1 imagen/paciente)", df_test, logger)

    # 7. Guardar
    cols = ["Image Index", "Finding Labels", "Patient ID", "View Position", "label"]
    df_train[cols].to_csv(processed_dir / "train.csv", index=False)
    df_val[cols].to_csv(processed_dir / "val.csv", index=False)
    df_test[cols].to_csv(processed_dir / "test.csv", index=False)

    logger.info("-" * 65)
    logger.info("Splits guardados en: %s", processed_dir)
    logger.info("  train.csv: %d filas", len(df_train))
    logger.info("  val.csv:   %d filas", len(df_val))
    logger.info("  test.csv:  %d filas", len(df_test))
    logger.info("  Ratio train imgs/train S2: %.1fx",
                len(df_train) / 739)
    logger.info("=" * 65)
    logger.info("PREPARE SPLITS S4 OK")


if __name__ == "__main__":
    main()
