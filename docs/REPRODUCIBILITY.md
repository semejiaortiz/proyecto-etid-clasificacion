# Protocolo de reproducibilidad

## Entorno de referencia

- Windows 10/11
- Python 3.10.18
- PyTorch 2.6.0 + CUDA 12.6
- Torchvision 0.21.0 + CUDA 12.6
- NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB

## Instalación

```powershell
conda create -n etid-vision python=3.10.18 -y
conda activate etid-vision
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements.txt
python -m ipykernel install --user --name etid-vision --display-name "Python 3 (etid-vision)"
```

## Secuencia verificada

1. Descargar ETID versión 1 y comprobar el ZIP:

   ```powershell
   python src/download_etid.py
   ```

2. Extraer y validar las 6.269 imágenes frente a las particiones canónicas:

   ```powershell
   python src/prepare_etid.py --zip data/raw/ETID_v1_original.zip
   ```

3. Comprobar los artefactos versionados sin reentrenar:

   ```powershell
   python src/verify_repository.py
   ```

4. Comprobar la carga del checkpoint y una inferencia real:

   ```powershell
   python src/smoke_test.py --data-root "RUTA/A/ETID"
   ```

5. Abrir `notebooks/ETID_Actividad_2.ipynb` con el kernel `etid-vision` y ejecutar de principio a fin. Si el dataset está en otra ubicación, definir `ETID_DATA_ROOT` antes de iniciar Jupyter.

## Controles aplicados

- Semilla fija `20261017`.
- Rutas relativas o configurables mediante `ETID_DATA_ROOT`.
- Dos exclusiones y nueve reetiquetados versionados.
- Particiones canónicas selladas mediante SHA-256.
- Cero rutas, hashes o grupos de procedencia compartidos entre particiones.
- Test no utilizado para selección.
- Historial por época y checkpoint del mejor estado.
- Hash del checkpoint final y métricas congeladas.
- Notebook ejecutado sin errores y sin rutas personales.
- Prueba ligera de carga estricta del checkpoint e inferencia con salida finita de siete clases.

## Modos de validación

`prepare_etid.py` realiza por defecto la comprobación completa: abre 6.269 imágenes y calcula los SHA-256 de las 6.267 observaciones incluidas. `--quick` omite esa lectura intensiva y sirve como diagnóstico inicial, pero no sustituye la validación completa antes de publicar resultados.

## Criterio de éxito

La preparación debe terminar con `resultado: APROBADO` y `verify_repository.py` debe devolver código cero. Una reproducción limpia debe poder abrir el notebook, localizar los datos sin editar celdas y verificar los resultados y el checkpoint antes de decidir si se reentrena.
