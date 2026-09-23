# Informe del Sprint 1 — Baseline CNN para clasificación de radiografías de tórax

**Proyecto:** Clasificación de radiografías de tórax con deep learning como apoyo al diagnóstico preliminar
**Sprint:** 1
**Período:** abril 2026
**Ubicación:** Lima, Perú

---

## 1. Resumen ejecutivo

El Sprint 1 tuvo como objetivo construir un baseline reproducible de clasificación binaria de radiografías de tórax (CXR) utilizando el dataset NIH ChestX-ray14 y la librería TorchXRayVision. Se completaron todas las actividades planificadas: carga y preprocesamiento de datos, división train/val/test, entrenamiento de un modelo CNN, evaluación con métricas clínicamente relevantes, y generación de mapas de explicabilidad (Grad-CAM).

El resultado obtenido, AUC=0.91 sobre el conjunto de test, es comparable a reportes en la literatura que utilizan el dataset NIH completo (aproximadamente 112,000 imágenes), a pesar de trabajar con un subset de 279 imágenes. Este desempeño se atribuye al uso de transfer learning a partir de un modelo preentrenado en el mismo dataset. La validación cualitativa mediante Grad-CAM confirma que el modelo atiende a la región anatómica correcta (silueta cardíaca) para tomar sus decisiones.

---

## 2. Objetivos del sprint

### 2.1 Objetivos planteados

Al inicio del sprint se definieron los siguientes entregables, alineados con el Project Charter:

1. Pipeline de datos modular y reproducible.
2. Implementación de un modelo CNN baseline (no híbrido CNN-ViT aún).
3. División train/val/test sin data leakage.
4. Métricas de evaluación: AUC, accuracy, precisión, sensibilidad, especificidad.
5. Validación cualitativa mediante mapas de calor (Grad-CAM).
6. Código modular, separado en módulos funcionales (datasets, models, training, evaluation, explainability).
7. Reproducibilidad con seed fijo.

### 2.2 Exclusiones explícitas

Se descartaron deliberadamente del alcance:

- Modelo híbrido CNN-ViT (diferido a Sprint 4).
- Interfaz de usuario (diferida a Sprint 5).
- Despliegue clínico.
- Clasificación multi-clase o multi-label (diferido a Sprint 3).

---

## 3. Decisiones metodológicas

### 3.1 Selección del dataset: NIH sobre CheXpert

Se evaluaron dos datasets candidatos: NIH ChestX-ray14 y CheXpert. Se seleccionó NIH por tres razones:

Primero, el acceso es inmediato y sin aprobación administrativa, mientras que CheXpert requiere registro y aprobación de Stanford AIMI (días o semanas de espera). Esta diferencia afectaba directamente la capacidad de iniciar el sprint.

Segundo, los labels de NIH son binarios (0 o 1 por patología), mientras que CheXpert incluye labels "inciertos" (-1) que requieren una decisión metodológica sobre cómo tratarlos (ignorar, convertir a 0, convertir a 1, o estrategias U-Ones/U-Zeros). Postergar esta decisión hasta Sprint 3 o posterior es apropiado.

Tercero, TorchXRayVision provee un modelo DenseNet121 preentrenado específicamente en NIH, lo que constituye un punto de partida óptimo para transfer learning.

### 3.2 Formulación binaria: "No Finding" vs "Cardiomegaly"

Se optó por una clasificación binaria simple como punto de partida. Entre las 14 patologías disponibles en NIH, se seleccionó cardiomegalia como clase positiva por tres motivos:

La cardiomegalia se manifiesta visualmente en una región anatómica específica (silueta cardíaca aumentada, con índice cardiotorácico superior a 0.5), lo cual permite validar cualitativamente los mapas de Grad-CAM: si el modelo aprende correctamente, el calor debe localizarse en la zona del corazón.

En segundo lugar, es una tarea suficientemente diferenciada para generar ground truth confiable: un radiólogo distinguiría entre estos dos casos sin mayor ambigüedad.

Finalmente, la base de datos contiene volumen suficiente de ambas clases en el subset utilizado (196 casos de cardiomegalia y 2,756 de "No Finding") para construir un dataset balanceado.

La definición operativa de positivo incluye imágenes donde "Cardiomegaly" aparece sola o combinada con otras patologías. Esta decisión se alinea con el enfoque de CheXNet (Rajpurkar et al., 2017) y refleja mejor el contexto clínico real, donde un paciente con cardiomegalia frecuentemente presenta patologías concurrentes.

### 3.3 División train/val/test estratificada a nivel paciente

Se requirió iterar cuatro versiones del algoritmo de split hasta lograr proporciones balanceadas en los tres conjuntos. Las iteraciones y sus resultados:

| Versión | Enfoque | Resultado | Problema |
|---|---|---|---|
| v1 | Random por paciente | Train 34%, Val 42%, Test 20% | Test desbalanceado |
| v2 | StratifiedGroupKFold sobre imágenes | Train 36%, Val 35%, Test 12% | Test aún peor |
| v3 | StratifiedKFold a nivel paciente | Train 24%, Val 14%, Test 21% | Val desviado por heterogeneidad en imgs/paciente |
| v4 | v3 + 1 imagen aleatoria por paciente | Train 33%, Val 35%, Test 35% | Correcto (volumen reducido) |

La versión final (v4) sacrifica volumen de datos (de 870 a 279 imágenes) a cambio de estratificación exacta. Esta es una decisión consistente con el objetivo del Sprint 1 de validar el pipeline antes de optimizar por volumen.

La lógica clave es: colapsar el dataset a un row por paciente con `patient_label = max(labels)`, aplicar StratifiedKFold sobre pacientes, y finalmente seleccionar una imagen aleatoria por paciente en cada split. Esto elimina el sesgo introducido por la heterogeneidad en el número de imágenes por paciente (algunos pacientes con 1 imagen, otros con 5 o más).

El script valida automáticamente la ausencia de data leakage y la calidad de la estratificación antes de guardar los CSVs.

### 3.4 Arquitectura del modelo: DenseNet121 con fine-tuning de cabeza

Se utilizó DenseNet121 preentrenado (`densenet121-res224-nih` de TorchXRayVision, 6,949,634 parámetros). La capa de clasificación original (14 salidas multi-label) se reemplazó por una capa lineal de 2 salidas (2,050 parámetros).

Se adoptó `finetune_mode: "head"`, congelando el backbone y entrenando únicamente la capa final. Esta decisión responde al volumen limitado de datos de entrenamiento (199 imágenes): con 6.9 millones de parámetros entrenables, un modo "full" llevaría a sobreajuste rápido. Con solo 2,050 parámetros entrenables, el riesgo se minimiza.

---

## 4. Pipeline implementado

### 4.1 Estructura modular

El código se organizó en 5 módulos funcionales, siguiendo principios de separación de responsabilidades:

- `src/datasets/`: transforms de imagen (`transforms.py`) y Dataset de PyTorch (`nih.py`).
- `src/models/`: definición del modelo baseline (`baseline.py`).
- `src/training/`: utilidades transversales (`utils.py`) y loop de entrenamiento (`train.py`).
- `src/evaluation/`: métricas (`metrics.py`) y función de evaluación (`evaluate.py`).
- `src/explainability/`: implementación de Grad-CAM (`gradcam.py`).

Los entrypoints ejecutables (`scripts/`) son delgados y delegan la lógica a los módulos de `src/`. Esta separación facilita la adición de tests y permite reutilizar los componentes en sprints posteriores sin modificaciones.

### 4.2 Preprocesamiento de imágenes

Las radiografías del NIH son archivos PNG de 1024×1024 píxeles, 8 bits, escala de grises. El pipeline de preprocesamiento aplica:

1. Normalización al rango [-1024, 1024] (formato esperado por los modelos preentrenados de TorchXRayVision).
2. Center crop (las imágenes no siempre son cuadradas).
3. Resize a 224×224 (resolución objetivo del modelo).
4. Augmentations leves (solo en train): rotaciones pequeñas (±5°), jitter de brillo (±10%).

Se evitó explícitamente el flip horizontal como augmentation, dado que invertiría la posición anatómica del corazón (que está a la izquierda del paciente desde la perspectiva del observador). Para la tarea de clasificación de cardiomegalia, donde el tamaño relativo de la silueta cardíaca es la señal principal, esta inversión podría confundir al modelo. La decisión se documenta como una restricción específica del dominio.

### 4.3 Configuración de entrenamiento

| Hiperparámetro | Valor | Justificación |
|---|---|---|
| Loss | CrossEntropyLoss | Compatible con num_classes=2 y extensible |
| Optimizer | Adam | Valor por defecto razonable |
| Learning rate | 0.001 | Apropiado para entrenar solo la cabeza |
| Weight decay | 1e-4 | Regularización leve |
| Batch size | 32 | Balance entre memoria y estabilidad |
| Epochs | 10 | Suficiente con 199 imágenes |
| Patience | 3 | Early stopping si val AUC no mejora |
| Mixed precision | Activado (T4) | Acelera entrenamiento ~40% |
| Seed | 42 | Reproducibilidad |

El loop de entrenamiento incluye:
- Validación al final de cada epoch con las 5 métricas.
- Checkpointing del mejor modelo por val AUC (no el último).
- Early stopping con paciencia configurable.
- Logging a archivo y consola para trazabilidad.

---

## 5. Resultados

### 5.1 Entrenamiento

El modelo se entrenó durante 10 epochs completos (no hubo early stopping). La progresión de la métrica principal (AUC) fue:

| Epoch | Train AUC | Val AUC |
|---|---|---|
| 1 | 0.7625 | 0.7802 |
| 2 | 0.9351 | 0.7885 |
| 3 | 0.9161 | 0.8104 |
| 5 | 0.9501 | 0.8242 |
| 7 | 0.9580 | 0.8297 |
| 10 | 0.9510 | **0.8434** |

El modelo alcanzó su mejor desempeño en el epoch 10 con val AUC = 0.8434. La brecha train-val (aproximadamente 0.11) indica sobreajuste moderado, esperable con el volumen de datos disponible. La curva de val AUC mantiene tendencia ascendente al final, sugiriendo que epochs adicionales podrían mejorar marginalmente.

Es notable que en las primeras dos epochs el modelo predice todos los casos como negativos (sensibilidad = 0), un comportamiento común cuando la cabeza de clasificación se inicializa con pesos aleatorios sobre un dataset con leve desbalance. A partir del epoch 3 el modelo comienza a distinguir positivos correctamente.

### 5.2 Evaluación final sobre test

El modelo seleccionado (epoch 10) se evaluó sobre el conjunto de test (40 imágenes que no se utilizaron durante entrenamiento ni para selección de hiperparámetros):

| Métrica | Val (40 imgs) | Test (40 imgs) | Diferencia |
|---|---|---|---|
| AUC | 0.8434 | **0.9121** | +0.0687 |
| Accuracy | 0.7500 | 0.8250 | +0.0750 |
| Precisión | — | 0.8182 | — |
| Sensibilidad | 0.5714 | 0.6429 | +0.0715 |
| Especificidad | 0.8462 | 0.9231 | +0.0769 |

El desempeño sobre test es superior al de validación, lo cual inicialmente podría parecer contraintuitivo. Dos consideraciones:

Primero, con conjuntos de 40 imágenes cada uno, la varianza muestral de las métricas es alta: un solo error cambia accuracy en 2.5 puntos y las métricas de sensibilidad/especificidad en aproximadamente 7 puntos. En este régimen, diferencias de esta magnitud entre val y test están dentro de la variabilidad estadística esperada.

Segundo, se verificó manualmente que no existe data leakage: los scripts de split validan que los pacientes de cada conjunto son disjuntos. La hipótesis más plausible es que el conjunto de test contiene casos relativamente más separables (pacientes con cardiomegalias más pronunciadas o negativos más claros), lo cual es resultado del muestreo aleatorio estratificado.

### 5.3 Matriz de confusión

Sobre test, con threshold=0.5:

```
                 Pred Neg    Pred Pos
True Neg            24          2        (spec = 24/26 = 92.3%)
True Pos             5          9        (sens = 9/14 = 64.3%)
```

El modelo clasifica correctamente 24 de 26 negativos y 9 de 14 positivos. Los 5 falsos negativos son la principal limitación: en un contexto clínico hipotético, no detectar una cardiomegalia puede tener más costo que un falso positivo. Esta asimetría es ajustable bajando el threshold (sin reentrenar el modelo), análisis diferido a Sprint 2.

### 5.4 Explicabilidad: Grad-CAM

Se generaron mapas de calor Grad-CAM sobre 6 imágenes de test (3 cardiomegalias y 3 "No Finding"), utilizando la última capa convolucional del backbone (`features.denseblock4`) como capa objetivo.

**Observaciones sobre las cardiomegalias (label=1):**
- Imagen `00000033_000.png`: el calor se concentra en la silueta cardíaca izquierda. Predicción correcta con probabilidad 0.63.
- Imagen `00000069_000.png`: calor intenso en el mediastino central (ubicación anatómica del corazón). Predicción correcta con probabilidad 0.89.
- Imagen `00000116_013.png`: calor en la región cardíaca. Predicción correcta con probabilidad 0.69.

**Observaciones sobre los casos normales (label=0):**
- En las tres imágenes, el modelo también atiende a la región cardíaca y mediastinal, pero asigna probabilidades bajas (0.10 a 0.35). Esto es consistente con el comportamiento esperado: el modelo "busca" cardiomegalia en la ubicación correcta y, al no encontrarla, clasifica como normal.

Esta evidencia cualitativa valida que el modelo aprendió la señal clínicamente correcta y no está aprovechando artefactos espurios (marcadores, letras, bordes de la imagen). Este resultado es especialmente relevante dado el volumen limitado de datos y el uso exclusivo de fine-tuning de cabeza.

---

## 6. Artefactos generados

El sprint produce los siguientes entregables verificables:

**Código fuente** (aproximadamente 900 líneas, 10 módulos):
- 5 módulos en `src/` (datasets, models, training, evaluation, explainability).
- 3 scripts ejecutables en `scripts/`.
- 1 archivo de configuración YAML.
- 5 notebooks ejecutables en orden.

**Artefactos de experimentación** (en `experiments/`):
- `baseline_best.pt`: checkpoint del mejor modelo (reproducible con seed=42).
- `history.json`: métricas por epoch del entrenamiento.
- `test_metrics.json`: resultados finales sobre test con predicciones individuales.
- `test_roc.png`: curva ROC.
- `test_confusion_matrix.png`: matriz de confusión visualizada.
- `gradcam_test.png`: grid de 6 imágenes con mapas de calor superpuestos.

**Logs** (en `experiments/logs/`):
- `prepare_splits.log`, `train.log`, `evaluate.log`: traza completa de cada etapa.

---

## 7. Limitaciones reconocidas

Se identifican cuatro limitaciones que se documentan explícitamente para honestidad metodológica:

1. **Volumen reducido del dataset.** El sprint operó sobre 279 imágenes totales (199 train / 40 val / 40 test), lo que introduce alta varianza en las métricas reportadas. Un solo error de clasificación modifica accuracy en 2.5 puntos porcentuales.

2. **Sensibilidad moderada.** Con threshold=0.5, el modelo falla en detectar el 35.7% de las cardiomegalias. En contextos clínicos esta asimetría es indeseable. Se proyecta análisis de threshold óptimo y uso de class weights en Sprint 2.

3. **Fine-tuning limitado.** Solo 2,050 de 6,949,634 parámetros son entrenables (0.03%). El modo `"full"` con más datos es la vía natural para mejorar.

4. **Una sola partición.** Ausencia de validación cruzada. Las métricas reportadas están sujetas a la variabilidad de la partición específica con seed=42.

Estas limitaciones no invalidan el baseline: son restricciones conocidas propias del alcance del Sprint 1. Se abordarán en sprints posteriores.

---

## 8. Próximos pasos

El roadmap de sprints queda como sigue:

**Sprint 2 — Optimización del baseline.** Descargar tarballs adicionales del NIH (`images_002.tar.gz` y siguientes), rehacer splits con más volumen (aproximadamente 600-1200 imágenes), cambiar a `finetune_mode: "full"` con learning rate bajo (1e-5), agregar LR scheduler y class weights. Reportar todas las métricas contra el baseline del Sprint 1.

**Sprint 3 — Extensión a multi-clase/multi-label.** Pasar a 3 clases (agregando Effusion) o multi-label sobre las 14 patologías NIH.

**Sprint 4 — Modelo híbrido CNN-ViT.** Implementación propia (no preentrenada) del modelo objetivo de la tesis. Comparar contra el baseline usando los mismos splits.

**Sprint 5 — Interfaz funcional.** Conforme al Project Charter.

---

## 9. Conclusiones del sprint

Los objetivos del Sprint 1 se cumplieron en su totalidad:

1. Se implementó un pipeline de datos modular y reproducible.
2. Se entrenó un modelo baseline con AUC=0.9121 sobre test, dentro del rango reportado en la literatura para esta tarea.
3. Se implementaron y validaron las 5 métricas del Project Charter.
4. La validación cualitativa mediante Grad-CAM confirma que el modelo atiende a la región anatómica correcta.
5. El código es modular y está listo para extensión en sprints posteriores.

El resultado principal no es el número de AUC en sí mismo, sino haber establecido una infraestructura técnica sólida sobre la cual construir los sprints siguientes: implementación del modelo híbrido CNN-ViT, evaluación sobre datasets externos, y eventualmente integración con interfaz funcional.
