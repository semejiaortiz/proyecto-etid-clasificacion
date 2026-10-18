# Código reproducible del pipeline de datos

Los scripts de esta carpeta permiten obtener ETID v1, normalizar su ubicación y comprobar que las imágenes corresponden exactamente a las particiones utilizadas en el experimento.

## 1. Descargar ETID

```powershell
python src/download_etid.py
```

La descarga se guarda en `data/raw/ETID_v1_original.zip`, admite reanudación mediante `curl` y solo se acepta si coincide con el SHA-256 publicado en este proyecto.

Si el ZIP ya fue descargado por otro medio:

```powershell
python src/download_etid.py --local-file "RUTA/ETID_v1_original.zip"
```

## 2. Extraer y preparar

```powershell
python src/prepare_etid.py --zip data/raw/ETID_v1_original.zip
```

El resultado queda normalizado bajo `data/raw/ETID/`, con las siete carpetas de clase en la raíz. La validación completa abre las imágenes y contrasta su SHA-256 con los manifiestos versionados.

Para validar una copia ya extraída:

```powershell
python src/prepare_etid.py --data-root "RUTA/A/ETID"
```

Puede definirse la misma ubicación con `ETID_DATA_ROOT`. La opción `--quick` conserva los controles estructurales, pero omite la lectura y el hash individual de cada imagen.

## 3. Verificar el repositorio

```powershell
python src/verify_repository.py
```

Este control no reentrena ni vuelve a seleccionar el modelo. Comprueba manifiestos, checkpoint, métricas congeladas, estado de ejecución del notebook y ausencia de rutas personales.

## 4. Probar una inferencia

```powershell
python src/smoke_test.py --data-root "RUTA/A/ETID"
```

La prueba carga el checkpoint versionado, reconstruye EfficientNet-B0 con su configuración seleccionada, valida una imagen contra el manifiesto de prueba y ejecuta una inferencia en CPU. Sirve para detectar rápidamente incompatibilidades del entorno sin repetir el entrenamiento.

## Decisión reproducible

Las particiones CSV son artefactos canónicos: no se recalculan aleatoriamente en cada equipo. `prepare_etid.py` comprueba su integridad, que no exista solapamiento y que las dos exclusiones y nueve correcciones de etiqueta coincidan con `data/decisiones_calidad.csv`. Esto permite comparar exactamente los mismos ejemplos y evita que una nueva optimización de particiones cambie el experimento ya reportado.
