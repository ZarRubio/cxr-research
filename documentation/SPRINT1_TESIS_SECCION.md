# Sección de tesis: Baseline de clasificación de radiografías de tórax mediante redes convolucionales con aprendizaje por transferencia

> Esta sección corresponde al capítulo de **metodología experimental** y **resultados del baseline** de la tesis. Puede integrarse como subsecciones dentro del capítulo principal, o como anexo técnico según la estructura final del documento.

---

## 3. Metodología experimental: baseline CNN

### 3.1 Descripción del baseline

Antes de proponer el modelo objetivo de la tesis (arquitectura híbrida CNN–ViT), se estableció un baseline basado en redes neuronales convolucionales (CNN) clásicas, específicamente una arquitectura DenseNet121 con aprendizaje por transferencia. La finalidad del baseline es doble: primero, establecer una referencia de desempeño contra la cual comparar el modelo híbrido en etapas posteriores; segundo, validar la infraestructura experimental (pipeline de datos, métricas, explicabilidad) antes de introducir mayor complejidad arquitectónica. Este enfoque de validación por etapas es consistente con la estructura metodológica reportada en trabajos relacionados de clasificación de imágenes médicas.

El baseline implementa una tarea de clasificación binaria: distinguir radiografías de tórax con cardiomegalia (definida operativamente como la presencia de esta patología, sola o combinada con otras) de radiografías sin hallazgos patológicos ("No Finding"). La cardiomegalia se seleccionó como patología de estudio inicial por presentar una señal visual localizada y clínicamente interpretable (aumento del índice cardiotorácico), lo que permite validar cualitativamente los mapas de explicabilidad generados por el modelo.

### 3.2 Dataset

Se utilizó el dataset NIH ChestX-ray14 (Wang et al., 2017), un recurso de acceso público compuesto por 112,120 radiografías frontales de tórax correspondientes a 30,805 pacientes del National Institutes of Health Clinical Center. Cada imagen está etiquetada con la presencia o ausencia de 14 patologías torácicas comunes, incluyendo cardiomegalia, neumonía, edema, efusión pleural y otras. Las etiquetas fueron extraídas automáticamente de reportes radiológicos mediante procesamiento de lenguaje natural, con una precisión estimada superior al 90% por los autores.

Por razones de capacidad de almacenamiento en el entorno de trabajo (Google Drive con restricciones de cuota), el Sprint 1 operó sobre un subset del dataset original correspondiente al primer tarball oficialmente publicado (`images_001.tar.gz`, aproximadamente 1.9 GB, 4,999 imágenes). La distribución de clases en este subset, tras filtrar por vistas frontales aceptadas (posteroanterior y anteroposterior), es la siguiente:

| Clase | Casos |
|---|---|
| No Finding | 2,756 |
| Cardiomegaly (sola o combinada) | 196 |
| Cardiomegaly pura | 72 |
| Otras patologías | 2,047 |

La selección de NIH sobre alternativas como CheXpert (Irvin et al., 2019) se fundamenta en tres criterios técnicos. Primero, NIH ofrece acceso inmediato sin procesos administrativos de aprobación. Segundo, las etiquetas de NIH son binarias (presencia/ausencia), mientras que CheXpert incluye etiquetas "inciertas" que requieren decisiones metodológicas adicionales sobre su tratamiento (conversión a clase positiva, negativa, o exclusión). Tercero, la librería TorchXRayVision (Cohen et al., 2022) provee pesos preentrenados específicamente en el dataset NIH, lo que constituye un punto de partida idóneo para aprendizaje por transferencia.

### 3.3 División de datos

La división de los datos en conjuntos de entrenamiento, validación y prueba se diseñó siguiendo tres principios:

**Primero, ausencia de data leakage entre conjuntos.** Dado que el dataset NIH contiene múltiples radiografías del mismo paciente (visitas de seguimiento), una división aleatoria por imagen permitiría que el mismo paciente apareciera en los conjuntos de entrenamiento y prueba, inflando artificialmente las métricas de desempeño. Esto constituye un error metodológico documentado en la literatura (Oakden-Rayner, 2019). En consecuencia, la división se realizó a nivel paciente: todas las imágenes de un paciente dado se asignan al mismo conjunto.

**Segundo, estratificación por clase.** Se buscó que los tres conjuntos tuvieran proporciones similares de casos positivos y negativos, para que las métricas sean comparables entre ellos y para prevenir sesgos en la evaluación.

**Tercero, balance de clases.** La clase negativa ("No Finding") es aproximadamente 14 veces más frecuente que la positiva en el subset utilizado. Un modelo entrenado sobre esta distribución tiende a maximizar accuracy prediciendo siempre la clase mayoritaria. Para mitigar esto, se aplicó submuestreo de la clase mayoritaria hasta alcanzar una proporción de 2:1 (negativos:positivos) a nivel paciente.

La combinación de estos tres principios requirió cuatro iteraciones metodológicas hasta lograr una implementación satisfactoria. La versión final (denotada v4) procede en cuatro etapas:

1. Submuestreo de pacientes negativos para lograr proporción 2:1.
2. Colapso del conjunto a un registro por paciente, asignando al paciente el label máximo de sus imágenes (si alguna imagen presenta cardiomegalia, el paciente se considera positivo).
3. División estratificada por clase sobre los pacientes (mediante `StratifiedKFold` de scikit-learn), con proporciones 70% / 15% / 15% para train / val / test.
4. Selección de una imagen aleatoria por paciente en cada conjunto, eliminando así el sesgo introducido por la heterogeneidad en el número de radiografías por paciente.

Esta estrategia coincide con la empleada en CheXNet (Rajpurkar et al., 2017) y sacrifica volumen de datos a cambio de estratificación exacta. Los conjuntos resultantes contienen 199 imágenes de entrenamiento, 40 de validación y 40 de prueba, con proporciones de cardiomegalia de 32.7%, 35.0% y 35.0% respectivamente.

### 3.4 Preprocesamiento

Las radiografías se convierten a escala de grises de un canal, se normalizan al rango [-1024, 1024] (formato esperado por los modelos preentrenados de TorchXRayVision, consistente con la codificación original de las imágenes médicas en 16 bits), se aplica un recorte centrado (para remover letras, marcadores y bordes no informativos) y se redimensionan a 224×224 píxeles, resolución objetivo del backbone.

Durante el entrenamiento se aplican augmentaciones conservadoras: rotaciones aleatorias de ±5 grados y perturbaciones de brillo de ±10%. Se evitó deliberadamente la aplicación de flip horizontal, dado que esta transformación invertiría la posición anatómica del corazón (lateralizado hacia la izquierda del paciente). Para la tarea específica de clasificación de cardiomegalia, donde el tamaño relativo de la silueta cardíaca es la señal principal, esta inversión podría inducir al modelo a aprender patrones espurios.

### 3.5 Arquitectura del modelo

El modelo baseline consiste en una arquitectura DenseNet121 (Huang et al., 2017) preentrenada sobre el dataset NIH ChestX-ray14 para clasificación multi-label de 14 patologías, disponible bajo el identificador `densenet121-res224-nih` en la librería TorchXRayVision. La capa de clasificación final original (con 14 salidas) se reemplaza por una capa lineal con 2 salidas correspondientes a las clases del problema binario. El backbone (5 bloques convolucionales con 6,947,584 parámetros) se mantiene congelado durante el entrenamiento; únicamente la nueva capa de clasificación (2,050 parámetros) se actualiza.

Esta configuración, denotada como "fine-tuning de cabeza" (head fine-tuning), es apropiada para volúmenes limitados de datos: entrenar los 6.9 millones de parámetros del backbone con solo 199 imágenes de entrenamiento conduciría a sobreajuste severo. El backbone preentrenado aporta conocimiento del dominio de radiografías de tórax, mientras que la nueva capa aprende el mapeo específico al problema binario.

### 3.6 Entrenamiento

El modelo se entrenó durante 10 epochs utilizando:
- **Función de pérdida:** entropía cruzada (`CrossEntropyLoss`), compatible con la representación logit de 2 clases y extensible a problemas multi-clase.
- **Optimizador:** Adam (Kingma y Ba, 2015) con tasa de aprendizaje inicial de 10⁻³ y regularización L2 con coeficiente 10⁻⁴.
- **Tamaño de batch:** 32.
- **Detención temprana:** paciencia de 3 epochs sin mejora en la métrica AUC sobre validación.
- **Entrenamiento en precisión mixta:** activado para aprovechar las capacidades Tensor Core de la GPU NVIDIA Tesla T4 utilizada.
- **Semilla aleatoria:** 42, fijada en todas las fuentes de aleatoriedad (Python, NumPy, PyTorch, CUDA) para garantizar reproducibilidad.

El modelo final se seleccionó como aquel que maximizó el AUC sobre el conjunto de validación a lo largo del entrenamiento, no el último epoch. Esta es una estrategia estándar para prevenir la selección de un modelo sobreajustado.

### 3.7 Métricas de evaluación

Se reportan cinco métricas de clasificación binaria, consistentes con los requerimientos del Project Charter:

- **Área bajo la curva ROC (AUC):** robusta al desbalance de clases, representa la capacidad discriminativa del modelo independientemente del umbral de decisión.
- **Accuracy:** proporción de predicciones correctas.
- **Precisión:** proporción de predicciones positivas que son correctas (VP / (VP + FP)).
- **Sensibilidad (Recall):** proporción de casos positivos detectados (VP / (VP + FN)).
- **Especificidad:** proporción de casos negativos correctamente identificados (VN / (VN + FP)).

Las últimas cuatro se calculan con un umbral de decisión fijo de 0.5 aplicado a las probabilidades obtenidas mediante softmax sobre los logits del modelo.

### 3.8 Explicabilidad

Se implementó Grad-CAM (Gradient-weighted Class Activation Mapping; Selvaraju et al., 2017) como mecanismo de explicabilidad. Grad-CAM genera mapas de activación que indican las regiones espaciales de la imagen que más contribuyen a la predicción de una clase específica, mediante el cómputo de los gradientes de la clase objetivo respecto a los mapas de activación de una capa convolucional intermedia.

Como capa objetivo se seleccionó el último bloque denso del backbone DenseNet121 (`features.denseblock4`), siguiendo la recomendación estándar de utilizar las últimas capas convolucionales, que codifican información semántica de alto nivel. La implementación utiliza la librería `pytorch-grad-cam`.

---

## 4. Resultados experimentales

### 4.1 Evolución del entrenamiento

El entrenamiento se completó en 10 epochs sin activar la detención temprana, lo que sugiere que el modelo mantuvo capacidad de aprendizaje hasta el final del proceso. La evolución de las principales métricas es la siguiente:

| Epoch | Train Loss | Train AUC | Val Loss | Val AUC | Val Acc |
|---|---|---|---|---|---|
| 1 | 0.5555 | 0.7625 | 0.5678 | 0.7802 | 0.6500 |
| 2 | 0.5222 | 0.9351 | 0.5579 | 0.7885 | 0.6500 |
| 3 | 0.4790 | 0.9161 | 0.5036 | 0.8104 | 0.6500 |
| 4 | 0.4340 | 0.9409 | 0.4704 | 0.8077 | 0.7500 |
| 5 | 0.3748 | 0.9501 | 0.4473 | 0.8242 | 0.7500 |
| 6 | 0.3520 | 0.9424 | 0.4490 | 0.8269 | 0.7500 |
| 7 | 0.3179 | 0.9580 | 0.4624 | 0.8297 | 0.7000 |
| 8 | 0.3038 | 0.9542 | 0.4383 | 0.8407 | 0.7500 |
| 9 | 0.2813 | 0.9508 | 0.4429 | 0.8352 | 0.7250 |
| **10** | 0.2933 | 0.9510 | **0.4182** | **0.8434** | 0.7500 |

Se observan tres comportamientos relevantes. Primero, en los dos primeros epochs el modelo predice sistemáticamente la clase mayoritaria (negativa), lo que se refleja en sensibilidad nula y accuracy igual a la proporción de negativos en validación (65%). A partir del epoch 3, el modelo comienza a distinguir positivos. Este fenómeno de "arranque conservador" es común cuando la capa de clasificación se inicializa aleatoriamente sobre un dataset con desbalance.

Segundo, el AUC sobre validación mejora de 0.7802 a 0.8434, un incremento del 8.1%. La tendencia es ascendente y no muestra signos claros de saturación, sugiriendo que entrenamientos más largos podrían aportar mejoras marginales.

Tercero, la brecha entre AUC de entrenamiento (0.95) y validación (0.84) refleja sobreajuste moderado, esperable dada la razón entre el número de parámetros entrenables (2,050) y el volumen de entrenamiento (199 imágenes).

### 4.2 Desempeño sobre conjunto de prueba

La evaluación final del modelo seleccionado se realizó sobre las 40 imágenes del conjunto de prueba, no utilizadas durante entrenamiento ni para selección de hiperparámetros. Las métricas son:

| Métrica | Validación | Prueba |
|---|---|---|
| AUC | 0.8434 | **0.9121** |
| Accuracy | 0.7500 | 0.8250 |
| Precisión | — | 0.8182 |
| Sensibilidad | 0.5714 | 0.6429 |
| Especificidad | 0.8462 | 0.9231 |

El AUC obtenido (0.91) se sitúa dentro del rango reportado en la literatura para la tarea de clasificación de cardiomegalia sobre NIH (Rajpurkar et al., 2017 reporta AUC=0.925 con el dataset completo; Wang et al., 2017 reporta AUC=0.81 como baseline). Este resultado es notable considerando que el presente trabajo utiliza únicamente 199 imágenes de entrenamiento, frente a las aproximadamente 78,000 disponibles en el dataset completo. La ventaja se atribuye al uso de aprendizaje por transferencia: el backbone preentrenado aporta representaciones visuales de alta calidad específicas del dominio, que la nueva capa de clasificación puede aprovechar con datos mínimos.

La matriz de confusión sobre prueba es:

```
                   Predicho No Finding   Predicho Cardiomegaly
Real No Finding            24                     2
Real Cardiomegaly           5                     9
```

El modelo clasifica correctamente 24 de 26 negativos (especificidad 92.3%) y 9 de 14 positivos (sensibilidad 64.3%). La principal limitación es la sensibilidad: se detectan 9 de cada 14 casos de cardiomegalia. Esta asimetría responde parcialmente al umbral de decisión utilizado (0.5); un análisis de umbrales operativos es parte del trabajo de Sprint 2.

### 4.3 Análisis cualitativo mediante Grad-CAM

Se generaron mapas de activación Grad-CAM sobre seis radiografías del conjunto de prueba (tres positivas y tres negativas), todas correctamente clasificadas por el modelo. La observación central es que, en los tres casos positivos, el mapa de calor se concentra sistemáticamente en la región cardíaca y mediastinal, coincidente con la localización anatómica donde un radiólogo humano evaluaría la presencia de cardiomegalia (medición del índice cardiotorácico).

En los casos negativos, el modelo atiende a la misma región anatómica, pero asigna probabilidades bajas a la clase positiva (entre 0.10 y 0.35), consistente con el comportamiento esperado: el modelo busca cardiomegalia en la ubicación correcta y, al no detectar silueta cardíaca aumentada, clasifica la imagen como normal.

Esta evidencia cualitativa valida que el modelo aprendió una señal clínicamente relevante y no está aprovechando artefactos espurios de las imágenes (marcadores, letras, bordes, variaciones de iluminación). Este resultado es particularmente relevante dado el volumen limitado de datos de entrenamiento y el hecho de que solo la capa final del modelo fue entrenada.

### 4.4 Limitaciones del baseline

Se identifican cuatro limitaciones que deben considerarse al interpretar los resultados:

Primero, el volumen reducido del conjunto de datos (279 imágenes totales) introduce alta varianza en las métricas: un error aislado en el conjunto de prueba altera la accuracy en 2.5 puntos porcentuales y la sensibilidad en aproximadamente 7 puntos. Las métricas reportadas deben interpretarse como estimaciones puntuales con intervalos de confianza amplios. Esta limitación se mitigará en Sprint 2 mediante la incorporación de tarballs adicionales del dataset NIH.

Segundo, solo se entrenó el 0.03% de los parámetros del modelo (2,050 de 6,949,634). Esta configuración prioriza robustez frente a sobreajuste, pero limita el techo de desempeño alcanzable. El modo de entrenamiento completo (fine-tuning de todos los parámetros) es viable con más datos y se propone como línea de trabajo futuro.

Tercero, la evaluación utiliza una sola partición de los datos. La ausencia de validación cruzada impide cuantificar la variabilidad del desempeño entre particiones distintas. La incorporación de validación cruzada es una extensión metodológica natural.

Cuarto, el modelo reporta sensibilidad moderada (64.3%). En un contexto clínico hipotético, la no detección de una cardiomegalia puede tener mayor costo que una falsa alarma. El ajuste del umbral de decisión a valores inferiores a 0.5, o la incorporación de pesos de clase en la función de pérdida, son estrategias para priorizar sensibilidad en sprints posteriores.

---

## 5. Conclusiones parciales del baseline

El baseline implementado alcanza un AUC de 0.9121 sobre el conjunto de prueba, situándose en rangos comparables con la literatura para la tarea de clasificación binaria de cardiomegalia en radiografías de tórax. Este resultado se obtuvo con un volumen de entrenamiento sustancialmente menor al utilizado en trabajos previos, gracias al aprovechamiento de aprendizaje por transferencia desde un modelo preentrenado en el mismo dominio.

La validación cualitativa mediante Grad-CAM confirma que el modelo atiende a la región anatómica clínicamente relevante para la detección de cardiomegalia, lo que respalda la hipótesis de que el modelo aprendió una representación útil y no espuria del problema. Este hallazgo es particularmente relevante considerando la limitación de datos de entrenamiento.

Más allá de las métricas obtenidas, el resultado principal del baseline es la validación de la infraestructura experimental: pipeline de datos modular y reproducible, división sin data leakage y con estratificación, implementación de métricas múltiples, y mecanismo de explicabilidad. Esta infraestructura constituye el fundamento sobre el cual se construirán las siguientes iteraciones del trabajo de tesis, incluyendo la implementación del modelo híbrido CNN–ViT objetivo y su comparación contra el baseline aquí establecido.

---

## Referencias

- Cohen, J. P., Viviano, J. D., Bertin, P., Morrison, P., Torabian, P., Guarrera, M., Lungren, M. P., Chaudhari, A., Brooks, R., Hashir, M., & Bertrand, H. (2022). TorchXRayVision: A library of chest X-ray datasets and models. *Medical Imaging with Deep Learning*.

- Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 4700-4708.

- Irvin, J., Rajpurkar, P., Ko, M., Yu, Y., Ciurea-Ilcus, S., Chute, C., Marklund, H., Haghgoo, B., Ball, R., Shpanskaya, K., Seekins, J., Mong, D. A., Halabi, S. S., Sandberg, J. K., Jones, R., Larson, D. B., Langlotz, C. P., Patel, B. N., Lungren, M. P., & Ng, A. Y. (2019). CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison. *Proceedings of the AAAI Conference on Artificial Intelligence*, 33(01), 590-597.

- Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *International Conference on Learning Representations (ICLR)*.

- Oakden-Rayner, L. (2019). Exploring large-scale public medical image datasets. *Academic Radiology*, 27(1), 106-112.

- Rajpurkar, P., Irvin, J., Zhu, K., Yang, B., Mehta, H., Duan, T., Ding, D., Bagul, A., Langlotz, C., Shpanskaya, K., Lungren, M. P., & Ng, A. Y. (2017). CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning. *arXiv preprint arXiv:1711.05225*.

- Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 618-626.

- Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M. (2017). ChestX-ray8: Hospital-scale chest X-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 3462-3471.
