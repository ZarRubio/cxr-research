"""
scripts/prepare_splits_s2.py

Genera los splits train/val/test para el Sprint 2: 4 clases.

Clases:
  0 - No Finding
  1 - Cardiomegaly   (insuficiencia cardíaca)
  2 - Effusion       (derrame pleural)
  3 - Infiltration   (TB pulmonar / neumonía)

Diferencias respecto a prepare_splits.py (Sprint 1):
  - 4 clases en vez de 2.
  - Asignación de labels por jerarquía de prioridad clínica.
    Una imagen con multiples patologias recibe el label de mayor prioridad:
    Infiltration (3) > Effusion (2) > Cardiomegaly (1) > No Finding (0).
  - Balance a nivel paciente: la clase mas pequeña define el techo.
    Todas las clases positivas se submuestrean a ese techo.
    No Finding se submuestrea a ratio*techo.
  - Misma estrategia de split v4: StratifiedKFold a nivel paciente
    + 1 imagen aleatoria por paciente.

Uso (desde Colab):
    !python /content/drive/MyDrive/Tesis_CXR/code/scripts/prepare_splits_s2.py
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

from src.training.utils import set_seed, get_logger, load_config  # noqa: E402


# ---------------------------------------------------------------------------
# Clases y jerarquía
# ---------------------------------------------------------------------------
# Jerarquía de prioridad (mayor número = mayor prioridad).
# Si una imagen tiene múltiples patologías, se asigna la de mayor prioridad.
LABEL_HIERARCHY = {
    "Infiltration": 3,  # prioridad máxima: TB / neumonía
    "Effusion":     2,
    "Cardiomegaly": 1,
    "No Finding":   0,  # solo si no tiene ninguna de las anteriores
}

# Nombres de clase para logging
CLASS_NAMES = {0: "No Finding", 1: "Cardiomegaly",
               2: "Effusion",   3: "Infiltration"}


# ---------------------------------------------------------------------------
# Construcción de labels multi-clase con jerarquía
# ---------------------------------------------------------------------------
def build_multiclass_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asigna un label entero 0-3 a cada imagen basándose en jerarquía clínica.

    Política:
      - Si 'Finding Labels' contiene 'Infiltration' → label=3
      - Si contiene 'Effusion'                      → label=2
      - Si contiene 'Cardiomegaly'                  → label=1
      - Si es exactamente 'No Finding'              → label=0
      - Cualquier otra combinación                  → NaN (se descarta)

    Nota: si una imagen tiene 'Effusion|Cardiomegaly', recibe label=2 (Effusion
    tiene prioridad). Esto refleja que en un contexto de triage el hallazgo
    de mayor urgencia define la clase.
    """
    df = df.copy()

    def assign_label(finding: str) -> float:
        for keyword, label in sorted(LABEL_HIERARCHY.items(),
                                     key=lambda x: -x[1]):
            if keyword == "No Finding":
                if finding == "No Finding":
                    return 0
            else:
                if keyword in finding:
                    return label
        return float('nan')  # ni "No Finding" ni ninguna de las 3 positivas

    df['label'] = df['Finding Labels'].apply(assign_label)
    return df


def filter_views(df: pd.DataFrame, views: list) -> pd.DataFrame:
    return df[df['View Position'].isin(views)].copy()


# ---------------------------------------------------------------------------
# Balance multi-clase a nivel paciente
# ---------------------------------------------------------------------------
def build_patient_level_label(df: pd.DataFrame) -> pd.Series:
    """
    Determina el label representativo de cada paciente usando la misma
    jerarquía: si un paciente tiene imágenes con distintos labels,
    se le asigna el de mayor prioridad.
    """
    return df.groupby('Patient ID')['label'].max().astype(int)


def balance_patients_multiclass(df: pd.DataFrame, ratio_negative: float,
                                seed: int, logger) -> pd.DataFrame:
    """
    Balancea el dataset a nivel paciente:
      1. Calcula el label representativo por paciente.
      2. Submuestrea cada clase positiva a la más pequeña (techo).
      3. Submuestrea No Finding a ratio_negative * techo.

    Args:
        df: DataFrame con columnas 'Patient ID', 'label'.
        ratio_negative: n_no_finding_pacientes = ratio_negative * n_min_positivo.
        seed: semilla para reproducibilidad.
    """
    rng = np.random.default_rng(seed)

    patient_labels = build_patient_level_label(df)

    # Separar pacientes por clase
    patients_by_class = {}
    for label in [0, 1, 2, 3]:
        pats = patient_labels[patient_labels == label].index.tolist()
        patients_by_class[label] = pats
        logger.info("  Clase %d (%s): %d pacientes",
                    label, CLASS_NAMES[label], len(pats))

    # Techo = mínimo entre las 3 clases positivas
    positive_counts = [len(patients_by_class[l]) for l in [1, 2, 3]]
    ceiling = min(positive_counts)
    logger.info("Techo de balance (clase más pequeña): %d pacientes", ceiling)

    # Submuestrear cada clase positiva al techo
    selected_patients = []
    for label in [1, 2, 3]:
        pats = patients_by_class[label]
        rng.shuffle(pats := list(pats))  # in-place shuffle
        selected = pats[:ceiling]
        selected_patients.extend(selected)
        logger.info("  Clase %d (%s): %d/%d pacientes seleccionados",
                    label, CLASS_NAMES[label], len(selected), len(pats))

    # Submuestrear No Finding
    n_neg_target = int(ceiling * ratio_negative)
    pats_neg = patients_by_class[0]
    rng.shuffle(pats_neg := list(pats_neg))
    selected_neg = pats_neg[:min(n_neg_target, len(pats_neg))]
    selected_patients.extend(selected_neg)
    logger.info("  Clase 0 (No Finding): %d pacientes seleccionados",
                len(selected_neg))

    # Filtrar df a los pacientes seleccionados
    selected_set = set(selected_patients)
    df_balanced = df[df['Patient ID'].isin(selected_set)].copy()
    df_balanced = df_balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    logger.info("Total tras balanceo: %d imágenes, %d pacientes",
                len(df_balanced), df_balanced['Patient ID'].nunique())
    return df_balanced


# ---------------------------------------------------------------------------
# Split estratificado a nivel paciente (misma lógica que Sprint 1 v4)
# ---------------------------------------------------------------------------
def stratified_patient_split(df: pd.DataFrame, train_ratio: float,
                             val_ratio: float, test_ratio: float,
                             seed: int, logger
                             ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    patient_labels = build_patient_level_label(df)
    logger.info("Pacientes únicos para split: %d", len(patient_labels))

    X_pat = patient_labels.index.to_numpy()
    y_pat = patient_labels.values

    # Paso 1: separar test
    n_splits_test = max(2, round(1.0 / test_ratio))
    skf_test = StratifiedKFold(n_splits=n_splits_test, shuffle=True,
                               random_state=seed)
    train_val_idx, test_idx = next(skf_test.split(X_pat, y_pat))

    test_patients = set(X_pat[test_idx])
    tv_patient_df = pd.DataFrame({
        'Patient ID': X_pat[train_val_idx],
        'patient_label': y_pat[train_val_idx],
    })

    # Paso 2: separar val de train+val
    val_ratio_rel = val_ratio / (train_ratio + val_ratio)
    n_splits_val = max(2, round(1.0 / val_ratio_rel))
    skf_val = StratifiedKFold(n_splits=n_splits_val, shuffle=True,
                              random_state=seed)
    X2 = tv_patient_df['Patient ID'].to_numpy()
    y2 = tv_patient_df['patient_label'].to_numpy()
    train_idx, val_idx = next(skf_val.split(X2, y2))

    train_patients = set(X2[train_idx])
    val_patients = set(X2[val_idx])

    logger.info("Split pacientes: train=%d, val=%d, test=%d",
                len(train_patients), len(val_patients), len(test_patients))

    df_train = df[df['Patient ID'].isin(train_patients)].reset_index(drop=True)
    df_val   = df[df['Patient ID'].isin(val_patients)].reset_index(drop=True)
    df_test  = df[df['Patient ID'].isin(test_patients)].reset_index(drop=True)

    return df_train, df_val, df_test


def sample_one_image_per_patient(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Selecciona 1 imagen aleatoria por paciente (estrategia v4)."""
    rng = np.random.default_rng(seed)
    sampled = []
    for _, group in df.groupby('Patient ID'):
        idx = rng.integers(0, len(group))
        sampled.append(group.iloc[idx])
    return pd.DataFrame(sampled).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Reporte y validaciones
# ---------------------------------------------------------------------------
def report_split(name: str, df: pd.DataFrame, logger) -> None:
    n = len(df)
    n_patients = df['Patient ID'].nunique()
    logger.info("%-5s: %4d imgs | %3d pacientes", name.upper(), n, n_patients)
    for label, cname in CLASS_NAMES.items():
        count = (df['label'] == label).sum()
        pct = count / n * 100 if n else 0
        logger.info("  clase %d (%s): %d (%.1f%%)", label, cname, count, pct)


def check_no_leakage(train, val, test, logger) -> None:
    p_tr = set(train['Patient ID'])
    p_va = set(val['Patient ID'])
    p_te = set(test['Patient ID'])
    for name, overlap in [('train-val', p_tr & p_va),
                           ('train-test', p_tr & p_te),
                           ('val-test', p_va & p_te)]:
        if overlap:
            logger.error("LEAKAGE en %s: %d pacientes", name, len(overlap))
            raise AssertionError(f"Data leakage en {name}")
    logger.info("Sin leakage: pacientes disjuntos entre splits.")


def check_stratification(train, val, test, logger,
                         max_deviation: float = 10.0) -> None:
    combined = pd.concat([train, val, test], ignore_index=True)
    logger.info("Distribución global de clases:")
    for label, cname in CLASS_NAMES.items():
        global_pct = (combined['label'] == label).mean() * 100
        logger.info("  clase %d (%s): %.1f%% global", label, cname, global_pct)
        for sname, sdf in [("train", train), ("val", val), ("test", test)]:
            pct = (sdf['label'] == label).mean() * 100
            dev = abs(pct - global_pct)
            if dev > max_deviation:
                logger.warning("  %s tiene %.1f%% de clase %d "
                               "(desviación %.1fpp)", sname, pct, label, dev)
    logger.info("Verificación de estratificación completada.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(config_path: str | None = None) -> None:
    if config_path is None:
        config_path = CODE_DIR / "configs" / "sprint2.yaml"
    cfg = load_config(config_path)

    set_seed(cfg.project['seed'])
    log_file = Path(cfg.paths['logs']) / "prepare_splits_s2.log"
    logger = get_logger("prepare_splits_s2", log_file=log_file)

    logger.info("=" * 60)
    logger.info("PREPARE SPLITS Sprint 2 — 4 clases")
    logger.info("Jerarquía: Infiltration(3) > Effusion(2) > "
                "Cardiomegaly(1) > No Finding(0)")
    logger.info("=" * 60)

    nih_dir = Path(cfg.paths['data_raw'])
    processed_dir = Path(cfg.paths['data_processed'])
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cargar CSV combinado de los 3 tarballs
    subset_csv = nih_dir / "Data_Entry_subset_local.csv"
    if not subset_csv.exists():
        raise FileNotFoundError(
            f"No se encontró {subset_csv}. "
            "Corre primero S2_01_download_more_data.ipynb."
        )

    df = pd.read_csv(subset_csv)
    logger.info("CSV cargado: %d filas", len(df))

    # 2. Filtrar vistas
    df = filter_views(df, cfg.dataset['views'])
    logger.info("Tras filtrar vistas %s: %d filas",
                cfg.dataset['views'], len(df))

    # 3. Construir labels con jerarquía
    df = build_multiclass_labels(df)
    n_before = len(df)
    df = df.dropna(subset=['label']).copy()
    df['label'] = df['label'].astype(int)
    logger.info("Tras construir labels: %d filas (descartadas %d sin clase)",
                len(df), n_before - len(df))

    for label, cname in CLASS_NAMES.items():
        n = (df['label'] == label).sum()
        logger.info("  %s: %d imágenes", cname, n)

    # 4. Balancear a nivel paciente
    ratio = cfg.get('class_ratio', 2.0)
    logger.info("Balanceando con ratio NF=%.1f * min_positivo...", ratio)
    df_balanced = balance_patients_multiclass(
        df, ratio_negative=ratio, seed=cfg.project['seed'], logger=logger
    )

    # 5. Split estratificado por paciente
    train_df, val_df, test_df = stratified_patient_split(
        df_balanced,
        train_ratio=cfg.splits['train_ratio'],
        val_ratio=cfg.splits['val_ratio'],
        test_ratio=cfg.splits['test_ratio'],
        seed=cfg.project['seed'],
        logger=logger,
    )

    # 6. Una imagen por paciente
    train_df = sample_one_image_per_patient(train_df, seed=cfg.project['seed'])
    val_df   = sample_one_image_per_patient(val_df,   seed=cfg.project['seed'] + 1)
    test_df  = sample_one_image_per_patient(test_df,  seed=cfg.project['seed'] + 2)

    # 7. Reportar y validar
    logger.info("-" * 60)
    report_split("train", train_df, logger)
    report_split("val",   val_df,   logger)
    report_split("test",  test_df,  logger)
    logger.info("-" * 60)
    check_no_leakage(train_df, val_df, test_df, logger)
    check_stratification(train_df, val_df, test_df, logger)

    # 8. Guardar
    cols = ['Image Index', 'Finding Labels', 'Patient ID', 'View Position', 'label']
    train_df[cols].to_csv(processed_dir / "train.csv", index=False)
    val_df[cols].to_csv(processed_dir   / "val.csv",   index=False)
    test_df[cols].to_csv(processed_dir  / "test.csv",  index=False)

    logger.info("Splits guardados en: %s", processed_dir)
    logger.info("  train.csv: %d filas", len(train_df))
    logger.info("  val.csv:   %d filas", len(val_df))
    logger.info("  test.csv:  %d filas", len(test_df))
    logger.info("=" * 60)
    logger.info("PREPARE SPLITS S2 OK")


if __name__ == "__main__":
    main()
