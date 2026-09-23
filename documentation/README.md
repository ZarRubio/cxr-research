# Tesis CXR — Clasificación de Radiografías de Tórax con Deep Learning

Proyecto de tesis sobre clasificación automatizada de radiografías de tórax (CXR) mediante técnicas de aprendizaje profundo, como apoyo al diagnóstico preliminar en contextos hospitalarios. El proyecto es de alcance académico y no está destinado a despliegue clínico real.

## Estado actual

**Sprint 1 completado.** Baseline CNN binario funcional sobre el dataset NIH ChestX-ray14.

| Métrica (sobre test, 40 imágenes nunca vistas) | Valor |
|---|---|
| AUC | 0.9121 |
| Accuracy | 0.8250 |
| Precisión | 0.8182 |
| Sensibilidad | 0.6429 |
| Especificidad | 0.9231 |

## Stack

- **Lenguaje:** Python 3.12
- **Framework:** PyTorch 2.x
- **Modelos y datasets CXR:** TorchXRayVision 1.3.4
- **Explicabilidad:** pytorch-grad-cam 1.5.4
- **Entorno de ejecución:** Google Colab (GPU Tesla T4)
- **Almacenamiento:** Google Drive

## Estructura del proyecto

```
Tesis_CXR/
├── code/
│   ├── configs/
│   │   └── baseline.yaml              # hiperparámetros del Sprint 1
│   ├── src/
│   │   ├── datasets/
│   │   │   ├── transforms.py          # preprocesamiento de imagen
│   │   │   └── nih.py                 # Dataset de PyTorch para NIH
│   │   ├── models/
│   │   │   └── baseline.py            # DenseNet121 preentrenado + cabeza binaria
│   │   ├── training/
│   │   │   ├── utils.py               # seed, logger, device, checkpoints
│   │   │   └── train.py               # loop de entrenamiento
│   │   ├── evaluation/
│   │   │   ├── metrics.py             # AUC, accuracy, precisión, sens, espec
│   │   │   └── evaluate.py            # evaluación sobre test
│   │   └── explainability/
│   │       └── gradcam.py             # Grad-CAM
│   ├── scripts/
│   │   ├── prepare_splits.py          # train/val/test estratificado por paciente
│   │   ├── run_train.py               # entrypoint de entrenamiento
│   │   └── run_eval.py                # entrypoint de evaluación
│   └── notebooks/
│       ├── 00b_download_nih.ipynb
│       ├── 00c_prepare_splits_v4.ipynb
│       ├── 01_data_pipeline.ipynb
│       ├── 03_train_baseline.ipynb
│       ├── 04_sprint1_closing.ipynb
│       └── _archive/                  # versiones intermedias (referencia)
├── data/
│   ├── raw/nih/                       # dataset original (no versionado)
│   │   ├── Data_Entry_2017.csv
│   │   ├── Data_Entry_subset_local.csv
│   │   └── images/                    # 4,999 imágenes .png
│   └── processed/                     # splits generados
│       ├── train.csv                  # 199 imágenes
│       ├── val.csv                    # 40 imágenes
│       └── test.csv                   # 40 imágenes
└── experiments/
    ├── checkpoints/
    │   └── baseline_best.pt           # mejor modelo (epoch 10, val AUC 0.8434)
    ├── logs/
    │   ├── prepare_splits.log
    │   ├── train.log
    │   └── evaluate.log
    ├── history.json                   # curvas de entrenamiento por epoch
    ├── test_metrics.json              # métricas finales sobre test
    ├── test_roc.png                   # curva ROC
    ├── test_confusion_matrix.png      # matriz de confusión
    └── gradcam_test.png               # mapas de calor sobre 6 imágenes
```

## Reproducibilidad

El proyecto es reproducible con `seed=42` fijado en `configs/baseline.yaml`. Con el mismo seed y la misma versión de dependencias, los splits, el entrenamiento y las métricas son idénticos.

### Pipeline de ejecución completo

Los 5 notebooks se ejecutan en orden. Cada uno es idempotente (se puede ejecutar múltiples veces sin efectos secundarios).

```
1. 00b_download_nih.ipynb       # descarga NIH a Drive (~20-40 min)
2. 00c_prepare_splits_v4.ipynb  # genera train/val/test (~30 seg)
3. 01_data_pipeline.ipynb       # valida transforms y dataset (~1 min)
4. 03_train_baseline.ipynb      # entrena el modelo (~3-8 min en T4)
5. 04_sprint1_closing.ipynb     # evaluación sobre test + Grad-CAM (~2-5 min)
```

## Configuración clave (`baseline.yaml`)

```yaml
dataset:
  positive_class: "Cardiomegaly"  # incluye combinaciones con otras patologías
  negative_class: "No Finding"
  views: ["PA", "AP"]

class_ratio: 2.0                  # negativos:positivos tras undersampling

splits:
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  split_by: "patient_id"          # evita data leakage

model:
  backbone: "densenet121-res224-nih"  # preentrenado NIH
  num_classes: 2
  finetune_mode: "head"               # entrena solo capa final (2,050 params)

training:
  batch_size: 32
  epochs: 10
  learning_rate: 0.001
  optimizer: "adam"
  patience: 3                        # early stopping
  use_amp: true                      # mixed precision para T4
```

## Decisiones metodológicas principales

### Dataset: NIH ChestX-ray14 (en lugar de CheXpert)

Se seleccionó NIH por acceso inmediato (CheXpert requiere aprobación de Stanford AIMI), labels binarios limpios (CheXpert tiene "uncertain labels" que requieren decisiones metodológicas adicionales), y disponibilidad de DenseNet121 preentrenado específicamente en este dataset vía TorchXRayVision.

### Tarea: Clasificación binaria "No Finding vs Cardiomegaly"

Se priorizó cardiomegalia como caso de estudio inicial por tener señal visual clara y localizada (silueta cardíaca aumentada), lo que permite validar cualitativamente los mapas de Grad-CAM. La definición de positivo incluye cardiomegalia combinada con otras patologías (196 imágenes) para ampliar el volumen de datos, siguiendo el enfoque de CheXNet (Rajpurkar et al., 2017).

### Split train/val/test estratificado a nivel paciente

Se evolucionó el algoritmo de split en 4 iteraciones (v1→v4) hasta lograr proporciones balanceadas en los tres splits (~33% positivos en cada uno). La versión final colapsa a un row por paciente, aplica `StratifiedKFold` sobre pacientes, y toma una imagen aleatoria por paciente. Esto elimina el sesgo por heterogeneidad en número de imágenes por paciente y previene data leakage.

### Modelo: DenseNet121 preentrenado con fine-tuning de cabeza

Se parte del modelo `densenet121-res224-nih` de TorchXRayVision (6,949,634 parámetros, preentrenado en las 14 patologías de NIH). Se reemplaza la capa final (14 salidas multi-label) por una capa binaria de 2 salidas (2,050 parámetros). Solo esta última capa se entrena (`finetune_mode: "head"`); el backbone permanece congelado. Esta configuración es apropiada para el volumen de datos disponible (199 imágenes de entrenamiento) y minimiza el riesgo de sobreajuste.

## Evolución de los splits (iteraciones metodológicas)

| Versión | Estrategia | Train % pos | Val % pos | Test % pos | Total imgs |
|---|---|---|---|---|---|
| v1 | Random por paciente | 34.3% | 41.7% | **19.8%** | 588 |
| v2 | StratifiedGroupKFold a nivel imagen | 35.8% | 34.7% | **12.1%** | 588 |
| v3 | StratifiedKFold a nivel paciente | 24.0% | **14.3%** | 21.0% | 870 |
| **v4** | **v3 + 1 imagen/paciente** | **32.7%** | **35.0%** | **35.0%** | **279** |

La v4 sacrifica volumen (870→279 imágenes) a cambio de estratificación exacta entre splits, siguiendo el enfoque de CheXNet. Esta decisión es consistente con el objetivo del Sprint 1 de validar el pipeline; se proyecta aumentar el volumen en Sprint 2 con tarballs adicionales del dataset NIH.

## Limitaciones reconocidas

- **Dataset chico.** 279 imágenes post-balanceo implican alta varianza en las métricas. Un solo error en test cambia accuracy en 2.5 puntos porcentuales.
- **Sensibilidad moderada (64%).** Con threshold=0.5, el modelo no detecta 36% de las cardiomegalias. Ajustable con threshold o class weights (Sprint 2).
- **Fine-tuning limitado.** Solo 2,050 de 6,949,634 parámetros son entrenables. Modo `"full"` con más datos es la vía natural para mejorar (Sprint 2).
- **Una sola partición train/val/test.** Ausencia de cross-validation. Las métricas de test están sujetas a la variabilidad de la partición específica.

## Próximos sprints

- **Sprint 2 — Optimización del baseline:** agregar `images_002.tar.gz` (+3.7 GB, ~5k imágenes), cambiar a `finetune_mode: "full"` con learning rate más bajo, agregar LR scheduler, probar class weights. Reportar contra Sprint 1 como baseline.
- **Sprint 3 — Extensión multi-clase:** pasar a 3 clases (agregar Effusion o Pneumothorax) o multi-label sobre las 14 patologías NIH.
- **Sprint 4 — Modelo híbrido CNN-ViT:** implementación propia (no preentrenada) del modelo objetivo de la tesis. Comparar contra el baseline usando los mismos splits.
- **Sprint 5 — Interfaz funcional:** conforme al Project Charter.

## Licencia y atribución de datos

El dataset NIH ChestX-ray14 es de acceso público. Atribución: Wang et al., "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases", IEEE CVPR, pp. 3462-3471, 2017. Link al dataset original: https://nihcc.app.box.com/v/ChestXray-NIHCC.

## Contacto

Proyecto de tesis (autor), ubicación: Lima, Perú. Año: 2026.
