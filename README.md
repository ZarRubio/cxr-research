# CXR Research

Research, training, evaluation, and explainability pipeline for a chest X-ray classification thesis using the NIH ChestX-ray14 dataset, PyTorch, DenseNet121, and hybrid CNN-ViT models.

> **Research use only.** This repository is an academic prototype. It is not a medical device and must not be used for diagnosis or clinical decision-making.

The deployed application is maintained separately in [`ZarRubio/cxr-system`](https://github.com/ZarRubio/cxr-system).

## Project status

The repository covers the complete experimental path from the initial binary baseline to the final 14-label CNN-ViT ensemble.

| Stage | Task | Main approach | Key result |
|---|---|---|---|
| Sprint 1 | Binary classification | DenseNet121, head fine-tuning | Test AUC 0.9121 |
| Sprint 2 | Four-class optimization | Full fine-tuning and focal loss | Test macro AUC 0.8653 |
| Sprint 3/4 | Hybrid modeling | DenseNet121 + Vision Transformer | CNN-ViT training pipeline |
| Sprint 4 ML | 14-label classification | Two CNN-ViT variants | v1 AUC 0.7909; v2 AUC 0.7950 |
| Final ensemble | 14-label classification | 30% v1 + 70% v2 | Test macro AUC 0.8045 |
| Calibration/validation | Calibrated thresholds | Temperature scaling and per-class thresholds | 12 of 14 classes above Wang et al. (2017) |

The final ensemble was evaluated on 4,023 test images and improved macro AUC by approximately 0.059 over the reported Wang et al. (2017) reference (0.7452).

## Repository structure

```text
cxr-research/
├── configs/                 # Experiment configurations
├── src/                     # Datasets, models, training, evaluation, Grad-CAM
├── scripts/                 # Reproducible preparation, training, and evaluation entrypoints
├── notebooks/               # Sprint notebooks and archived iterations
├── documentation/           # Methodology and sprint reports
├── experiments/             # Sprint 1 metrics and figures
├── experiments_s2/          # Sprint 2 metrics and figures
├── experiments_s4/          # CNN-ViT experiment history and calibration
├── experiments_s4ml/        # Multi-label v1 history
├── experiments_s4ml_v2/     # Final validation, ensemble metrics, and figures
├── requirements.txt
└── .gitignore
```

Datasets, generated splits, training logs, and model checkpoints are intentionally excluded from Git.

## Setup

Python 3.12 was used for the experiments.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Data

The experiments use the public NIH ChestX-ray14 dataset. Download instructions and data-preparation workflows are available in `notebooks/00b_download_nih.ipynb` and the scripts under `scripts/`.

Expected local artifacts include:

```text
data/raw/nih/
data/processed*/
```

The large `nih_images.h5` file, original radiographs, processed splits, and any patient-level data are not included in this repository.

Dataset reference: Wang et al., *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases*, CVPR 2017.

## Running the pipeline

The project contains separate entrypoints for each experimental stage. Examples:

```bash
# Prepare the original baseline splits
python scripts/prepare_splits.py

# Train and evaluate the Sprint 1 baseline
python scripts/run_train.py
python scripts/run_eval.py

# Prepare and train the Sprint 2 experiments
python scripts/prepare_splits_s2.py
python scripts/run_train_s2.py

# Prepare the 14-label split and train the CNN-ViT model
python scripts/prepare_splits_multilabel.py
H5_PATH=/path/to/nih_images.h5 \
PROCESSED_DIR=/path/to/processed_s4ml \
python scripts/run_train_multilabel.py
```

Configuration files in `configs/` define paths, class selection, preprocessing, model architecture, optimization, scheduling, early stopping, and reproducibility settings.

## Final results

The final ensemble combines two CNN-ViT variants with four and six Transformer blocks.

| Metric | Value |
|---|---:|
| Test samples | 4,023 |
| Macro AUC | 0.8045 |
| Mean average precision | 0.1521 |
| Wang et al. reference macro AUC | 0.7452 |
| Macro AUC improvement | +0.0593 |
| Classes above reference | 12 / 14 |

Detailed per-class AUC, calibrated thresholds, sensitivity, specificity, precision, F1, and class counts are available in:

- `experiments_s4ml_v2/ensemble_results.json`
- `experiments_s4ml_v2/validation_report.json`
- `experiments_s4ml_v2/thresholds_calibrated.json`

![Final ensemble comparison](experiments_s4ml_v2/figures/ensemble_final.png)

## Reproducibility

- Random seed: `42`
- Framework: PyTorch 2.x
- Backbone: DenseNet121 through TorchXRayVision
- Explainability: Grad-CAM
- Original execution environment: Google Colab with an NVIDIA Tesla T4

Paths stored in historical notebooks and result files may still refer to the original Google Drive workspace. Set the environment variables and configuration paths for your local environment before running an experiment.

## Related repository

- Production application: [`ZarRubio/cxr-system`](https://github.com/ZarRubio/cxr-system)

This repository documents how the models were trained and evaluated; `cxr-system` contains the FastAPI/Next.js application used to serve the resulting models.

