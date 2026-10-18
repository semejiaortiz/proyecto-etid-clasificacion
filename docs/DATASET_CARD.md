# Ficha del dataset: ETID

## Identificación

- **Nombre:** E-Tadweer Image Dataset (ETID)
- **Versión:** 1
- **Repositorio:** Mendeley Data
- **DOI:** https://doi.org/10.17632/bndsgcspbx.1
- **Licencia:** CC BY 4.0
- **Tarea utilizada:** clasificación multiclase de imágenes

## Variable objetivo

Una de siete etiquetas: `Battery`, `Cable`, `Keyboard`, `Laptop`, `Mobile`, `Mouse` o `PCB`.

## Auditoría aplicada

- 6.269 archivos abiertos correctamente.
- 61 resoluciones distintas.
- Dos copias binarias excluidas.
- Nueve etiquetas corregidas sin modificar los archivos originales.
- Partición por grupos de procedencia.
- Cero rutas, hashes exactos o grupos compartidos entre particiones.

## Usos previstos

Investigación académica sobre clasificación inicial de residuos electrónicos y comparación de modelos de visión por computador.

## Usos fuera de alcance

- Determinar peligrosidad o composición química.
- Decidir reciclabilidad o tratamiento final.
- Detectar varios objetos en una escena.
- Inferir marca, modelo o estado físico.
- Desplegar el modelo sin validación local adicional.

## Limitaciones

Existe desbalance entre clases y mezcla de fuentes. El conjunto no garantiza representar todos los países, fondos, iluminaciones o dispositivos de un punto de recepción real. La clase `Cable` es especialmente pequeña.

## Trazabilidad

El repositorio incorpora los manifiestos depurados de entrenamiento, validación y prueba en `data/splits/`, sus hashes SHA-256 y el registro de decisiones de calidad en `data/decisiones_calidad.csv`. El script `src/prepare_etid.py` contrasta estos artefactos con una copia local del dataset sin modificar las imágenes originales.
