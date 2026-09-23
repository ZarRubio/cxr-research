"""
scripts/prepare_splits.py  (v4)

Genera los splits train/val/test para el baseline binario del Sprint 1.

CAMBIO EN v4 frente a v3:
  Despues del split estratificado por paciente, se selecciona UNA SOLA imagen
  aleatoria por paciente. Esto elimina el sesgo introducido por la
  heterogeneidad en el numero de imagenes por paciente, garantizando
  estratificacion exacta tanto a nivel paciente como a nivel imagen.

Trade-off:
  - PRO: balance perfecto entre splits (cada paciente cuenta como 1).
  - CONTRA: dataset mas chico (~280 imgs en vez de 870).

Justificacion academica:
  Es el enfoque usado en CheXNet (Rajpurkar et al., 2017) y otros papers
  de baseline en CXR. Si en el futuro se quiere usar mas imagenes, se puede
  modificar el sampleo para tomar k>1 imagenes por paciente, o ponderar
  por inverso de imagenes/paciente para reducir el sesgo.

Pipeline completo:
  1. Cargar Data_Entry_subset_local.csv.
  2. Filtrar vistas (PA/AP).
  3. Construir labels binarios.
  4. Balancear a NIVEL PACIENTE (ratio 2:1).
  5. Split estratificado a nivel paciente.
  6. Sampling: 1 imagen aleatoria por paciente en cada split.
  7. Validar y guardar.
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
# Construccion de labels
# ---------------------------------------------------------------------------
def build_binary_labels(df: pd.DataFrame, positive_class: str,
                        negative_class: str) -> pd.DataFrame:
    df = df.copy()
    is_positive = df['Finding Labels'].str.contains(positive_class, regex=False)
    is_negative = df['Finding Labels'] == negative_class
    df['label'] = np.where(is_positive, 1,
                   np.where(is_negative, 0, np.nan))
    return df


def filter_views(df: pd.DataFrame, views: list[str]) -> pd.DataFrame:
    return df[df['View Position'].isin(views)].copy()


def undersample_negatives(df: pd.DataFrame, ratio: float,
                          seed: int) -> pd.DataFrame:
    """Submuestrea pacientes negativos para alcanzar ratio = n_neg / n_pos."""
    positives = df[df['label'] == 1]
    negatives = df[df['label'] == 0]

    pos_patients = positives['Patient ID'].unique()
    neg_patients = negatives['Patient ID'].unique()
    # Excluir pacientes que aparecen como positivos
    neg_patients_clean = np.setdiff1d(neg_patients, pos_patients)

    n_pos_pat = len(pos_patients)
    n_neg_pat_target = min(len(neg_patients_clean), int(n_pos_pat * ratio))

    rng = np.random.default_rng(seed)
    rng.shuffle(neg_patients_clean)
    neg_patients_sampled = neg_patients_clean[:n_neg_pat_target]

    keep_patients = set(pos_patients) | set(neg_patients_sampled)
    balanced = df[df['Patient ID'].isin(keep_patients)].copy()
    balanced = balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return balanced


# ---------------------------------------------------------------------------
# Split estratificado a nivel paciente
# ---------------------------------------------------------------------------
def collapse_to_patient_level(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby('Patient ID')['label']
              .max()
              .astype(int)
              .reset_index()
              .rename(columns={'label': 'patient_label'}))


def stratified_patient_split(df: pd.DataFrame, train_ratio: float,
                             val_ratio: float, test_ratio: float,
                             seed: int, logger
                             ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    patient_df = collapse_to_patient_level(df)
    logger.info("Pacientes unicos en pool balanceado: %d (%d positivos, %d negativos)",
                len(patient_df),
                int((patient_df['patient_label'] == 1).sum()),
                int((patient_df['patient_label'] == 0).sum()))

    # Paso 1: separar test
    n_splits_test = max(2, round(1.0 / test_ratio))
    skf_test = StratifiedKFold(n_splits=n_splits_test, shuffle=True,
                               random_state=seed)
    X_pat = patient_df['Patient ID'].to_numpy()
    y_pat = patient_df['patient_label'].to_numpy()
    train_val_idx, test_idx = next(skf_test.split(X_pat, y_pat))

    test_patients = set(X_pat[test_idx])
    train_val_patient_df = patient_df.iloc[train_val_idx].reset_index(drop=True)

    # Paso 2: separar val de train+val
    val_ratio_rel = val_ratio / (train_ratio + val_ratio)
    n_splits_val = max(2, round(1.0 / val_ratio_rel))
    skf_val = StratifiedKFold(n_splits=n_splits_val, shuffle=True,
                              random_state=seed)
    X_tv = train_val_patient_df['Patient ID'].to_numpy()
    y_tv = train_val_patient_df['patient_label'].to_numpy()
    train_idx, val_idx = next(skf_val.split(X_tv, y_tv))

    train_patients = set(X_tv[train_idx])
    val_patients = set(X_tv[val_idx])

    logger.info("Split a nivel paciente: train=%d, val=%d, test=%d pacientes",
                len(train_patients), len(val_patients), len(test_patients))

    df_train = df[df['Patient ID'].isin(train_patients)].reset_index(drop=True)
    df_val = df[df['Patient ID'].isin(val_patients)].reset_index(drop=True)
    df_test = df[df['Patient ID'].isin(test_patients)].reset_index(drop=True)

    return df_train, df_val, df_test


# ---------------------------------------------------------------------------
# NUEVO en v4: 1 imagen aleatoria por paciente
# ---------------------------------------------------------------------------
def sample_one_image_per_patient(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Selecciona una imagen aleatoria por paciente.

    Para pacientes con multiples imagenes con labels distintos (raro pero
    posible si una visita es 'Cardiomegaly' y otra es 'No Finding'),
    se prioriza una imagen positiva si existe (consistencia con el
    patient_label asignado en el split).
    """
    rng = np.random.default_rng(seed)

    sampled_rows = []
    for patient_id, group in df.groupby('Patient ID'):
        # Si el paciente tiene imagenes con distintos labels, priorizar positivas
        if (group['label'] == 1).any() and (group['label'] == 0).any():
            group = group[group['label'] == 1]
        # Tomar una al azar
        idx = rng.integers(0, len(group))
        sampled_rows.append(group.iloc[idx])

    sampled_df = pd.DataFrame(sampled_rows).reset_index(drop=True)
    return sampled_df


# ---------------------------------------------------------------------------
# Reportes
# ---------------------------------------------------------------------------
def report_split(name: str, df: pd.DataFrame, logger) -> None:
    n = len(df)
    n_pos = int((df['label'] == 1).sum())
    n_neg = int((df['label'] == 0).sum())
    n_patients = df['Patient ID'].nunique()
    pct_pos = (n_pos / n * 100) if n else 0.0
    logger.info(
        "%-5s: %4d imgs | %3d pacientes | pos=%3d (%.1f%%) | neg=%3d",
        name.upper(), n, n_patients, n_pos, pct_pos, n_neg,
    )


def check_no_leakage(train, val, test, logger) -> None:
    p_train = set(train['Patient ID'])
    p_val = set(val['Patient ID'])
    p_test = set(test['Patient ID'])
    overlaps = {
        'train-val': p_train & p_val,
        'train-test': p_train & p_test,
        'val-test': p_val & p_test,
    }
    for name, overlap in overlaps.items():
        if overlap:
            logger.error("LEAKAGE en %s: %d pacientes", name, len(overlap))
            raise AssertionError(f"Data leakage en {name}")
    logger.info("Sin leakage: pacientes disjuntos entre splits.")


def check_one_image_per_patient(train, val, test, logger) -> None:
    """Validacion v4: cada paciente debe aparecer exactamente 1 vez."""
    for name, df in [("train", train), ("val", val), ("test", test)]:
        if len(df) != df['Patient ID'].nunique():
            logger.error(
                "%s tiene %d imagenes pero %d pacientes (deberian coincidir).",
                name, len(df), df['Patient ID'].nunique(),
            )
            raise AssertionError(f"Mas de 1 imagen por paciente en {name}")
    logger.info("Validado: 1 imagen por paciente en cada split.")


def check_stratification(train, val, test, logger,
                         max_deviation: float = 5.0) -> None:
    combined = pd.concat([train, val, test], ignore_index=True)
    global_pct = (combined['label'] == 1).mean() * 100
    logger.info("Proporcion global de positivos: %.1f%%", global_pct)

    all_ok = True
    for name, df in [("train", train), ("val", val), ("test", test)]:
        pct = (df['label'] == 1).mean() * 100
        deviation = abs(pct - global_pct)
        if deviation > max_deviation:
            logger.warning(
                "%s tiene %.1f%% positivos (desviacion %.1fpp).",
                name, pct, deviation,
            )
            all_ok = False
    if all_ok:
        logger.info("Estratificacion OK: las 3 proporciones estan dentro de "
                    "%.0fpp del global.", max_deviation)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(config_path: str | None = None) -> None:
    if config_path is None:
        config_path = CODE_DIR / "configs" / "baseline.yaml"
    cfg = load_config(config_path)

    set_seed(cfg.project['seed'])
    log_file = Path(cfg.paths['logs']) / "prepare_splits.log"
    logger = get_logger("prepare_splits", log_file=log_file)

    logger.info("=" * 60)
    logger.info("PREPARE SPLITS v4 - 1 imagen por paciente")
    logger.info("=" * 60)

    nih_dir = Path(cfg.paths['data_raw'])
    processed_dir = Path(cfg.paths['data_processed'])
    processed_dir.mkdir(parents=True, exist_ok=True)

    subset_csv = nih_dir / "Data_Entry_subset_local.csv"
    if not subset_csv.exists():
        raise FileNotFoundError(f"No se encontro {subset_csv}.")

    # 1. Cargar
    df = pd.read_csv(subset_csv)
    logger.info("CSV cargado: %d filas", len(df))

    # 2. Filtrar vistas
    df = filter_views(df, cfg.dataset['views'])
    logger.info("Tras filtrar vistas %s: %d filas", cfg.dataset['views'], len(df))

    # 3. Labels
    df = build_binary_labels(
        df,
        positive_class=cfg.dataset['positive_class'],
        negative_class=cfg.dataset['negative_class'],
    )
    n_before = len(df)
    df = df.dropna(subset=['label']).copy()
    df['label'] = df['label'].astype(int)
    logger.info("Tras construir labels: %d filas (descartadas %d ambiguas)",
                len(df), n_before - len(df))

    # 4. Balancear a nivel paciente
    ratio = cfg.get('class_ratio', 2.0)
    df_balanced = undersample_negatives(df, ratio=ratio, seed=cfg.project['seed'])
    logger.info("Tras balancear (ratio %.1f:1 paciente): %d imagenes, %d pacientes",
                ratio, len(df_balanced),
                df_balanced['Patient ID'].nunique())

    # 5. Split estratificado por paciente
    train_df, val_df, test_df = stratified_patient_split(
        df_balanced,
        train_ratio=cfg.splits['train_ratio'],
        val_ratio=cfg.splits['val_ratio'],
        test_ratio=cfg.splits['test_ratio'],
        seed=cfg.project['seed'],
        logger=logger,
    )

    logger.info("--- ANTES de tomar 1 imagen/paciente: ---")
    report_split("train", train_df, logger)
    report_split("val", val_df, logger)
    report_split("test", test_df, logger)

    # 6. Tomar 1 imagen aleatoria por paciente
    train_df = sample_one_image_per_patient(train_df, seed=cfg.project['seed'])
    val_df = sample_one_image_per_patient(val_df, seed=cfg.project['seed'] + 1)
    test_df = sample_one_image_per_patient(test_df, seed=cfg.project['seed'] + 2)

    logger.info("--- DESPUES de tomar 1 imagen/paciente: ---")
    report_split("train", train_df, logger)
    report_split("val", val_df, logger)
    report_split("test", test_df, logger)

    # 7. Validaciones
    logger.info("-" * 60)
    check_no_leakage(train_df, val_df, test_df, logger)
    check_one_image_per_patient(train_df, val_df, test_df, logger)
    check_stratification(train_df, val_df, test_df, logger)

    # 8. Guardar
    columns_to_save = ['Image Index', 'Finding Labels', 'Patient ID',
                       'View Position', 'label']
    train_df[columns_to_save].to_csv(processed_dir / "train.csv", index=False)
    val_df[columns_to_save].to_csv(processed_dir / "val.csv", index=False)
    test_df[columns_to_save].to_csv(processed_dir / "test.csv", index=False)

    logger.info("Splits guardados en: %s", processed_dir)
    logger.info("  train.csv: %d filas", len(train_df))
    logger.info("  val.csv:   %d filas", len(val_df))
    logger.info("  test.csv:  %d filas", len(test_df))
    logger.info("=" * 60)
    logger.info("PREPARE SPLITS v4 OK")


if __name__ == "__main__":
    main()
