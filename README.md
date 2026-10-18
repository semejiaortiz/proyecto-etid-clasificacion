# Clasificación visual de residuos electrónicos con ETID

Proyecto integrador de la Unidad 4 de **Técnicas Avanzadas de Modelado en Inteligencia Artificial**. Se compara EfficientNet-B0 con una línea base HOG + LinearSVC para clasificar siete categorías de residuos electrónicos, manteniendo una separación estricta entre entrenamiento, validación y prueba.

## Equipo

- Sebastian Mejia Ortiz
- Laura Mariana Jimenez Mienez
- Luis Gonzalo Duran

**Docente:** Oscar Alberto Rodriguez Melendez  
**Curso:** MIA 2026-5

## Estado

El repositorio contiene el notebook portátil y ejecutado, el checkpoint seleccionado, las particiones verificadas, las predicciones, las métricas y los informes técnicos de las actividades 2 y 3. Los scripts de descarga, preparación y verificación permiten comprobar el flujo de datos sin modificar las particiones reportadas. Consulte [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md).

## Problema

La entrada es una imagen RGB y la salida es una de siete clases: `Battery`, `Cable`, `Keyboard`, `Laptop`, `Mobile`, `Mouse` o `PCB`. El sistema se estudia como apoyo a una clasificación inicial; no determina peligrosidad, composición química ni reciclabilidad.

> ¿EfficientNet-B0 con transferencia de aprendizaje mejora el F1 macro de HOG + LinearSVC cuando ambos modelos se evalúan sobre grupos de procedencia independientes?

## Estructura

```text
Repositorio_ETID/
├── configs/       Configuraciones experimentales
├── data/          Manifiestos, decisiones e instrucciones de descarga
├── docs/          Fichas del dataset, modelo y reproducibilidad
├── models/        Checkpoint final verificado
├── notebooks/     Notebook ejecutado de principio a fin
├── presentation/  Presentación HTML final y material de sustentación
├── reports/       Informes y figuras
├── results/       Métricas, predicciones e historiales
└── src/           Descarga, preparación y controles de integridad
```

## Instalación

Se recomienda Python 3.10 y una GPU compatible con CUDA:

```powershell
conda create -n etid-vision python=3.10.18 -y
conda activate etid-vision
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements.txt
python -m ipykernel install --user --name etid-vision --display-name "Python 3 (etid-vision)"
```

## Preparación y verificación

```powershell
python src/download_etid.py
python src/prepare_etid.py --zip data/raw/ETID_v1_original.zip
python src/verify_repository.py
```

Si ETID ya está extraído, se puede indicar su ubicación:

```powershell
python src/prepare_etid.py --data-root "RUTA/A/ETID"
```

La guía completa está en [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). Las imágenes no se versionan en Git; permanecen bajo la licencia CC BY 4.0 de sus autores.

## Principios experimentales

- Semilla principal: `20261017`.
- Partición por grupos de procedencia, no por imágenes aisladas.
- Selección exclusivamente con validación.
- Test abierto después de congelar el checkpoint.
- F1 macro como métrica principal por el desbalance entre clases.
- Registro de configuraciones, historiales, predicciones y hashes SHA-256.

## Licencias y atribución

ETID conserva su propia autoría y no se redistribuye en este repositorio. Consulte [LICENSES.md](LICENSES.md) para la atribución y el alcance de los materiales de terceros.
