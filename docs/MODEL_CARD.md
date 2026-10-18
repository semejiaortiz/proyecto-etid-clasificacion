# Ficha del modelo

## Identificación

- **Arquitectura:** EfficientNet-B0
- **Inicialización:** pesos preentrenados en ImageNet
- **Entrada:** imagen RGB de 224 × 224
- **Salida:** `Battery`, `Cable`, `Keyboard`, `Laptop`, `Mobile`, `Mouse` o `PCB`
- **Métrica principal:** F1 macro
- **Línea base:** HOG + LinearSVC

## Modelo seleccionado

El experimento E03 entrenó la cabeza clasificadora con AdamW, tasa de aprendizaje `1 × 10⁻³`, `dropout = 0,4`, `weight_decay = 1 × 10⁻⁴` y entropía cruzada ponderada. Su época 1 obtuvo el mejor F1 macro de validación (`0,8777`). La ventaja frente a E01 fue de apenas `0,0008`, por lo que se interpreta como un resultado prácticamente empatado.

El ajuste fino de `features[7]`, `features[8]` y la cabeza redujo el F1 de validación a `0,8481` y `0,8671`. Ninguna época superó el checkpoint de origen; por tanto, el modelo final conserva E03 antes del ajuste fino.

## Evaluación final

| Métrica | Línea base | EfficientNet-B0 seleccionada | Diferencia |
|---|---:|---:|---:|
| F1 macro | 0,2849 | 0,7461 | +0,4612 |
| Exactitud balanceada | 0,3601 | 0,7840 | +0,4238 |
| Exactitud | 0,3347 | 0,7769 | +0,4422 |
| Precisión macro | 0,3389 | 0,7846 | +0,4457 |
| *Recall* macro | 0,3601 | 0,7840 | +0,4238 |

El intervalo *bootstrap* pareado del 95 % para la diferencia de F1 macro fue `[0,4237; 0,5022]`. El conjunto de prueba contiene 995 imágenes y no intervino en el entrenamiento ni en la selección.

## Desempeño por clase

| Clase | Precisión | *Recall* | F1 | Soporte |
|---|---:|---:|---:|---:|
| Battery | 0,8836 | 0,8402 | 0,8613 | 244 |
| Cable | 0,3448 | 0,9091 | 0,5000 | 22 |
| Keyboard | 0,8621 | 0,7042 | 0,7752 | 71 |
| Laptop | 0,9808 | 0,6220 | 0,7612 | 82 |
| Mobile | 0,5405 | 0,9581 | 0,6911 | 167 |
| Mouse | 0,8901 | 0,7788 | 0,8308 | 104 |
| PCB | 0,9904 | 0,6754 | 0,8031 | 305 |

## Integridad

- **Semilla:** `20261017`
- **SHA-256 del checkpoint:** `9d5361fed4282fdbe5744f3e559cea4ecb7a252598699978b5a596a53146b349`
- **SHA-256 de train:** `b0759318003bab059154a48de9e1bff47fbc02cb4e70e72b672f956bfa9309fa`
- **SHA-256 de validation:** `bb17b6d7c7d398d02f847b2114d5d574e028dd8756a17a803d98604313ef7eed`
- **SHA-256 de test:** `e63e149a2417fb4773121710248fef286f9ab908d7133b7ccfa8c7997d19a935`

## Uso previsto

Referencia académica y apoyo experimental para una clasificación inicial de residuos electrónicos dentro de las siete categorías de ETID.

## Limitaciones y riesgos

El modelo puede fallar ante fondos, dispositivos o condiciones de captura distintos. `Cable` tiene alta sensibilidad pero baja precisión, y la brecha entre validación y test es de `−0,1316` puntos de F1 macro. La salida no determina peligrosidad, composición ni reciclabilidad y debe conservar supervisión humana. Se requiere validación externa antes de cualquier despliegue.

