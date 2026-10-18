# Datos

Las imágenes de ETID no se almacenan en Git.

- **Nombre:** E-Tadweer Image Dataset (ETID)
- **Versión:** 1
- **Autores:** Sayed et al. (2026)
- **DOI:** https://doi.org/10.17632/bndsgcspbx.1
- **Licencia:** CC BY 4.0
- **Archivo:** `ETID_v1_original.zip`
- **SHA-256:** `751ae9ffdaab724c364ddb9df4fd5f82a022ef1e2bce40fb9802444146e66951`

## Composición auditada

El archivo original contiene 6.269 imágenes legibles. Dos copias binarias se excluyeron y nueve observaciones se reetiquetaron mediante decisiones documentadas, sin modificar las imágenes. El conjunto experimental contiene 6.267 observaciones.

| Partición | Imágenes | Propósito |
|---|---:|---|
| Entrenamiento | 4.289 | Ajuste de parámetros |
| Validación | 983 | Selección y parada temprana |
| Prueba | 995 | Evaluación final única |

Las particiones se construyeron por grupos de procedencia y están selladas mediante SHA-256 en `data/splits/`. Las decisiones manuales están versionadas en `data/decisiones_calidad.csv`.

## Preparación automática

Desde la raíz del repositorio:

```powershell
python src/download_etid.py
python src/prepare_etid.py --zip data/raw/ETID_v1_original.zip
```

También se puede validar un dataset existente:

```powershell
python src/prepare_etid.py --data-root "RUTA/A/ETID"
```

La ubicación normalizada es `data/raw/ETID/`. Tanto `data/raw/` como `data/processed/` están excluidas de Git.

## Alcance

Los scripts verifican integridad, legibilidad, decisiones de calidad, clases, conteos, ausencia de solapamiento y coincidencia con los manifiestos del experimento. No redistribuyen imágenes ni cambian las particiones después de conocer los resultados.
