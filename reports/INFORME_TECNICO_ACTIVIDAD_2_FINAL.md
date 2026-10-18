# Clasificación de residuos electrónicos con EfficientNet-B0

**Actividad 2 — Ejecución, optimización y validación comparativa**  
**Técnicas Avanzadas de Modelado en Inteligencia Artificial · Unidad 4**  
**Integrantes:** Sebastian Mejia Ortiz · Laura Mariana Jimenez Mienez · Luis Gonzalo Duran  
**Docente:** Oscar Alberto Rodriguez Melendez  
**Fecha:** 17 de octubre de 2026

## Resumen

Este trabajo evalúa una red EfficientNet-B0 para clasificar imágenes de residuos electrónicos en siete categorías: baterías, cables, teclados, computadores portátiles, teléfonos móviles, ratones y tarjetas de circuito impreso. La pregunta experimental fue si una red convolucional con transferencia de aprendizaje podía mejorar una línea base HOG + LinearSVC al comparar ambos modelos sobre grupos de procedencia independientes. Se conservaron las particiones de la Actividad 1, se modificó un solo factor por experimento y se usó F1 macro como criterio principal debido al desbalance entre clases. El modelo seleccionado obtuvo un F1 macro de **0,8777** en validación y **0,7461** en prueba, frente a **0,2849** de la línea base. La mejora absoluta fue de **0,4612**, con un intervalo *bootstrap* pareado del 95 % entre **0,4237 y 0,5022**. Aunque el avance es claro dentro de ETID, la caída entre validación y prueba y los errores concentrados en algunas clases aconsejan validar el sistema con nuevas fuentes antes de plantear un uso operativo.

**Palabras clave:** residuos electrónicos, visión por computador, EfficientNet, transferencia de aprendizaje, clasificación de imágenes.

<!-- SALTO DE PÁGINA 1 -->

## 1. Problema y diseño experimental

La separación inicial de residuos electrónicos suele depender de una inspección humana y puede perder consistencia cuando cambian el fondo, el estado del objeto o la fuente de captura. Por ello se estudió un sistema de apoyo que recibe una imagen RGB y propone una de siete categorías. Su alcance es deliberadamente acotado: no determina peligrosidad, composición química ni reciclabilidad, y tampoco sustituye la decisión de una persona responsable del proceso.

Se utilizó **E-Tadweer Image Dataset (ETID), versión 1**, publicado por Sayed et al. (2026) en Mendeley Data bajo licencia CC BY 4.0. La auditoría inicial encontró 6.269 archivos legibles, dos copias binarias mal ubicadas y nueve etiquetas que requerían corrección. Sin modificar los originales, el manifiesto experimental quedó compuesto por 6.267 imágenes. La documentación de procedencia, composición y limitaciones siguió el enfoque de trazabilidad propuesto por Gebru et al. (2021).

| Partición | Imágenes | Grupos de procedencia | Uso |
|---|---:|---:|---|
| Entrenamiento | 4.289 | 16 | Ajuste de parámetros |
| Validación | 983 | 6 | Selección, parada e hiperparámetros |
| Prueba | 995 | 8 | Evaluación final única |

Las imágenes relacionadas permanecieron en el mismo grupo. La verificación registró cero rutas, hashes exactos, grupos o coincidencias perceptuales estrictas compartidas entre particiones. Este esquema reduce la posibilidad de obtener métricas infladas por fotografías casi duplicadas y aproxima mejor el cambio de procedencia que enfrentaría el modelo fuera del entrenamiento. Asimismo, la selección se realizó únicamente con validación, como recomiendan las prácticas de evaluación para evitar que el conjunto de prueba influya en las decisiones de modelado (Scikit-learn Developers, s. f.).

Cada imagen se orientó según EXIF, se convirtió a RGB y se llevó a un lienzo de 224 × 224 mediante *letterbox*, conservando la relación de aspecto. Después se aplicó la normalización de ImageNet. Solo entrenamiento recibió volteo horizontal, rotación de ±8° y variaciones moderadas de brillo y contraste; validación y prueba conservaron transformaciones deterministas. Como `Cable` representa menos del 3 % del conjunto depurado y `Battery` supera el 23 %, se empleó entropía cruzada ponderada con pesos calculados exclusivamente en entrenamiento. La métrica principal fue F1 macro, acompañada por exactitud balanceada, exactitud, precisión y *recall* macro.

Se fijó la semilla `20261017` y se conservaron configuraciones, versiones, historiales, predicciones y hashes SHA-256. El conjunto de prueba permaneció cerrado durante la búsqueda. El checkpoint se congeló antes de evaluarlo y el notebook bloquea una evaluación posterior con un archivo diferente.

<!-- SALTO DE PÁGINA 2 -->

## 2. Arquitectura y optimización

EfficientNet-B0 fue elegida por su escalamiento conjunto de profundidad, anchura y resolución, diseñado para equilibrar capacidad predictiva y costo computacional (Tan & Le, 2019). Se partió de pesos preentrenados en ImageNet y se reemplazó el clasificador original por *dropout* y una capa lineal de siete salidas.

| Componente | Función | Tratamiento experimental |
|---|---|---|
| *Stem* y primeros MBConv | Rasgos visuales generales | Conservados y congelados inicialmente |
| MBConv con *squeeze-and-excitation* | Representaciones jerárquicas eficientes | Congelados al entrenar la cabeza |
| `features[7]` y `features[8]` | Rasgos de alto nivel | Liberados únicamente en el ajuste fino |
| *Global average pooling* | Resumen espacial de características | Conservado |
| *Dropout* + capa lineal | Regularización y siete puntuaciones | Componente entrenable principal |

La primera etapa entrenó solo la cabeza clasificadora. Este punto de partida aprovecha representaciones generales sin alterar de inmediato todo el extractor. Posteriormente se ensayó la liberación de los dos bloques superiores con tasas menores. El procedimiento responde al principio de controlar la complejidad y vigilar por separado el comportamiento de entrenamiento y validación (Goodfellow et al., 2016). También se utilizó precisión mixta y recorte del gradiente a norma 1,0.

Los experimentos fueron definidos antes de consultar el test. E01 actuó como referencia; E02 redujo la tasa de aprendizaje y E03 aumentó el *dropout*. Se mantuvieron constantes AdamW, `weight_decay = 1 × 10⁻⁴`, un máximo de seis épocas, paciencia de dos épocas y mejora mínima de 0,002 en F1 macro. AdamW se escogió antes de la ejecución por separar el decaimiento de pesos de la actualización adaptativa (Loshchilov & Hutter, 2019). No se ensayaron optimizadores nuevos después de observar la prueba.

| ID | Tasa de aprendizaje | *Dropout* | Cambio controlado |
|---|---:|---:|---|
| E01 | 1 × 10⁻³ | 0,2 | Configuración de referencia |
| E02 | 3 × 10⁻⁴ | 0,2 | Menor tasa de aprendizaje |
| E03 | 1 × 10⁻³ | 0,4 | Mayor regularización |

Modificar un factor por corrida hizo posible interpretar el efecto de cada cambio, en consonancia con el enfoque experimental recomendado por Google Developers (s. f.).

Los tiempos registrados de entrenamiento y validación sumaron **37,6 minutos** para E01, E02, E03 y el intento de ajuste fino. Esta cifra permite dimensionar el costo de la búsqueda, aunque no incluye carga de datos, preprocesamiento ni evaluación final y, por tanto, no representa el tiempo total del notebook.

<!-- SALTO DE PÁGINA 3 -->

## 3. Entrenamiento y selección

| Experimento | Mejor época | F1 macro | Exactitud balanceada | Exactitud | Lectura |
|---|---:|---:|---:|---:|---|
| E01 | 1 | 0,8770 | 0,8925 | 0,9054 | Referencia competitiva |
| E02 | 1 | 0,8639 | 0,8846 | 0,8942 | La tasa menor no mejoró F1 |
| E03 | 1 | **0,8777** | **0,8949** | 0,9044 | Seleccionado por validación |

E03 superó a E01 por solo **0,0008** de F1 macro. Por tanto, no se interpreta como evidencia de una superioridad sustantiva del *dropout* 0,4, sino como un empate práctico resuelto mediante la regla de selección previamente fijada. E02 tampoco ganó estabilidad suficiente para compensar su menor rendimiento inicial.

![Curvas de pérdida y F1 macro de los tres experimentos](figuras/figura_1_curvas_aprendizaje.png)

**Figura 1.** Evolución de pérdida y F1 macro en entrenamiento y validación. Las líneas discontinuas representan entrenamiento y las continuas, validación. *Fuente: elaboración propia a partir de los historiales ejecutados.*

Las curvas revelan el mismo patrón en las tres corridas. Después de la primera época, el F1 de entrenamiento sigue aumentando, mientras el de validación se estanca o disminuye. En E01 la brecha pasa de 0,0288 a 0,1299; en E02 alcanza 0,1064 y en E03 llega a 0,1202 en la tercera época. La reducción de la pérdida de entrenamiento sin una mejora equivalente fuera de muestra es evidencia de sobreajuste, no de convergencia generalizable. La parada temprana recuperó en cada caso el estado de la primera época.

El ajuste fino comenzó desde E03 y no desde la última época. Al liberar `features[7]`, `features[8]` y la cabeza, el F1 de validación cayó a 0,8481 en la primera época y se recuperó hasta 0,8671 en la segunda, todavía **0,0106 por debajo** del checkpoint de origen. En consecuencia, el ajuste fino fue rechazado y el modelo final conservó E03 antes de liberar bloques. Su hash SHA-256 es `9d5361fed4282fdbe5744f3e559cea4ecb7a252598699978b5a596a53146b349`.

<!-- SALTO DE PÁGINA 4 -->

## 4. Validación comparativa

| Modelo | F1 macro | Exactitud balanceada | Exactitud | Precisión macro | *Recall* macro |
|---|---:|---:|---:|---:|---:|
| HOG + LinearSVC | 0,2849 | 0,3601 | 0,3347 | 0,3389 | 0,3601 |
| EfficientNet-B0 seleccionada | **0,7461** | **0,7840** | **0,7769** | **0,7846** | **0,7840** |
| Mejora absoluta | +0,4612 | +0,4238 | +0,4422 | +0,4457 | +0,4238 |

En las 995 imágenes de prueba, EfficientNet-B0 mejoró todas las métricas globales frente a la línea base. Para F1 macro, los 2.000 remuestreos *bootstrap* pareados produjeron un intervalo percentil del 95 % de **[0,4237; 0,5022]**; la diferencia fue positiva en el 100 % de las réplicas. Esto respalda una mejora consistente en esta muestra, aunque no reemplaza una validación externa.

![Comparación de modelos y matriz de confusión normalizada](figuras/figura_2_comparacion_confusion.png)

**Figura 2.** Comparación en prueba y matriz de confusión normalizada de EfficientNet-B0. *Fuente: elaboración propia a partir de las predicciones congeladas.*

| Clase | Precisión | *Recall* | F1 | Soporte |
|---|---:|---:|---:|---:|
| Battery | 0,884 | 0,840 | 0,861 | 244 |
| Cable | 0,345 | 0,909 | 0,500 | 22 |
| Keyboard | 0,862 | 0,704 | 0,775 | 71 |
| Laptop | 0,981 | 0,622 | 0,761 | 82 |
| Mobile | 0,541 | 0,958 | 0,691 | 167 |
| Mouse | 0,890 | 0,779 | 0,831 | 104 |
| PCB | 0,990 | 0,675 | 0,803 | 305 |

<!-- SALTO DE PÁGINA 5 -->

## 5. Análisis de errores y conclusiones

El comportamiento por clase muestra una tensión clara entre sensibilidad y precisión. `Cable` recupera 20 de sus 22 casos, pero su precisión de 0,345 indica numerosos falsos positivos. Es una clase escasa y visualmente heterogénea, de modo que el resultado debe leerse con cautela. `Mobile` presenta el *recall* más alto (0,958), aunque recibe objetos de otras categorías: 65 PCB, 30 portátiles y 22 baterías fueron clasificados como teléfonos. Esta concentración sugiere que algunos fondos, encuadres o formas rectangulares compartidas pesan más que los rasgos específicos del objeto.

En contraste, `Laptop` y `PCB` logran precisiones de 0,981 y 0,990, pero dejan sin recuperar una proporción relevante de casos. No es un modelo que confunda indiscriminadamente esas clases; cuando las predice suele acertar, pero adopta una decisión demasiado restrictiva. `Battery` y `Mouse` presentan el equilibrio más estable. Estas diferencias justifican mantener F1 macro como criterio, porque una exactitud global aislada ocultaría el bajo desempeño de la clase minoritaria.

El F1 desciende de 0,8777 en validación a 0,7461 en prueba, una brecha de **0,1316**. Dado que las particiones se separaron por procedencia, esta caída es compatible con un cambio visual entre grupos y constituye una advertencia sobre la generalización. No se utilizó para reajustar hiperparámetros: hacerlo convertiría el test en un conjunto adicional de validación.

En respuesta a la pregunta experimental, **EfficientNet-B0 sí superó de manera amplia la línea base HOG + LinearSVC bajo las mismas particiones**. La mejora no provino de prolongar el entrenamiento: las curvas mostraron sobreajuste temprano y el ajuste fino empeoró la validación. La decisión técnicamente más sólida fue conservar el checkpoint de E03 en la primera época. El resultado demuestra el valor de la transferencia de aprendizaje para ETID, pero todavía no demuestra confiabilidad operativa en imágenes de otras instituciones o condiciones de captura.

Como trabajo posterior se propone ampliar `Cable`, revisar ejemplos ambiguos, estudiar calibración y una opción de abstención, y realizar una evaluación externa con imágenes recolectadas en un punto de recepción distinto. Cualquier nueva optimización debe abrir un ciclo experimental separado; el conjunto de prueba de esta entrega queda congelado.

### Reproducibilidad

El notebook entregado ejecuta el flujo completo, conserva la semilla, comprueba los manifiestos, registra historiales y predicciones, y verifica el checkpoint por SHA-256. Los resultados finales proceden del mismo conjunto de 995 observaciones usado por la línea base. El repositorio incluye configuración, dependencias, ficha del dataset, ficha del modelo, métricas por clase y artefactos necesarios para auditar la comparación. No se redistribuyen las imágenes originales de ETID.

### Referencias

Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & Crawford, K. (2021). Datasheets for datasets. *Communications of the ACM, 64*(12), 86–92. https://doi.org/10.1145/3458723

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press. https://www.deeplearningbook.org/

Google Developers. (s. f.). *A scientific approach to improving model performance*. https://developers.google.com/machine-learning/guides/deep-learning-tuning-playbook/scientific

Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. En *International Conference on Learning Representations*. https://openreview.net/forum?id=Bkg6RiCqY7

Sayed, A., Hossen, M. M., Jano, T., Hamila, R., Bensaali, F., Touati, F., & Varlamis, I. (2026). *E-Tadweer Image Dataset (ETID)* (Version 1) [Conjunto de datos]. Mendeley Data. https://doi.org/10.17632/bndsgcspbx.1

Scikit-learn Developers. (s. f.). *Cross-validation: Evaluating estimator performance*. https://scikit-learn.org/stable/modules/cross_validation.html

Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. En *Proceedings of the 36th International Conference on Machine Learning* (Vol. 97, pp. 6105–6114). PMLR. https://proceedings.mlr.press/v97/tan19a.html
