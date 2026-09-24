# CXR Research

Research, training, evaluation, and explainability pipeline for a chest X-ray classification thesis using the NIH ChestX-ray14 dataset, PyTorch, DenseNet121, and hybrid CNN-ViT models.

> **Research use only.** This repository is an academic prototype. It is not a medical device and must not be used for diagnosis or clinical decision-making.

The deployed application is maintained separately in [`ZarRubio/cxr-system`](https://github.com/ZarRubio/cxr-system).

## Project status

The repository contains research code and notebooks for binary, four-class, and 14-label chest X-ray experiments. Experiment metrics and figures are omitted from this repository while the evaluation protocol is being revised.

The earlier DenseNet backbone used TorchXRayVision weights trained on NIH ChestX-ray14, so an NIH holdout does not provide an independent generalization estimate. The earlier comparison with Wang et al. (2017) is not retained because the split and evaluation protocols are not matched.

## Repository structure

```text
cxr-research/
├── configs/                 # Experiment configurations
├── src/                     # Datasets, models, training, evaluation, Grad-CAM
├── scripts/                 # Reproducible preparation, training, and evaluation entrypoints
├── notebooks/               # Sprint notebooks and archived iterations
├── documentation/           # Methodology and sprint reports
├── requirements.txt
└── .gitignore
```

Datasets, generated splits, per-image outputs, experiment figures, training logs, and model checkpoints are intentionally excluded from Git.

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

# Prepare the revised patient-level, multilabel-stratified split
CONFIG_PATH=configs/sprint4_multilabel_v3.yaml python scripts/prepare_splits_multilabel.py

# Train with a CheXpert-initialized backbone (no NIH pretraining)
CONFIG_PATH=configs/sprint4_multilabel_v3.yaml python scripts/run_train_multilabel.py
CONFIG_PATH=configs/sprint4_multilabel_v3.yaml python scripts/run_eval_multilabel.py

# Train the paired NIH-pretrained control on the same split
CONFIG_PATH=configs/sprint4_multilabel_v3_nih_control.yaml python scripts/run_train_multilabel.py
CONFIG_PATH=configs/sprint4_multilabel_v3_nih_control.yaml python scripts/run_eval_multilabel.py
```

Configuration files in `configs/` define paths, class selection, preprocessing, model architecture, optimization, scheduling, early stopping, and reproducibility settings.

## Evaluation status

No experimental metrics or figures are included in this repository. The revised patient-level split and paired backbone-initialization configs provide a new evaluation protocol; results should be reported only after those experiments have been rerun and reviewed.

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
