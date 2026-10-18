#!/usr/bin/env python3
"""Comprueba la integridad de los artefactos versionados del proyecto ETID."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


def digest(path: Path, chunk_size: int = 1024 * 1024) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Audita el repositorio ETID sin reentrenar el modelo.")
    parser.add_argument("--output", type=Path, help="JSON opcional con el resultado de la verificación.")
    args = parser.parse_args()

    split_dir = repo / "data" / "splits"
    verification = json.loads((split_dir / "verificacion_particiones.json").read_text(encoding="utf-8"))
    split_hashes = {
        name: digest(split_dir / name) for name in ("train.csv", "validation.csv", "test.csv")
    }
    checks: dict[str, object] = {}
    checks["manifiestos_sha256"] = split_hashes == verification["sha256_manifiestos"]

    model_manifest = json.loads((repo / "models" / "model_manifest.json").read_text(encoding="utf-8"))
    model_path = repo / "models" / model_manifest["file"]
    checks["checkpoint_existe"] = model_path.is_file()
    checks["checkpoint_sha256"] = model_path.is_file() and digest(model_path) == model_manifest["sha256"]

    final_result = json.loads(
        (repo / "results" / "advanced" / "evaluacion_final" / "result.json").read_text(encoding="utf-8")
    )
    checks["test_cerrado_durante_seleccion"] = final_result["test_used_for_training_or_selection"] is False
    checks["checkpoint_resultado"] = final_result["checkpoint_sha256"] == model_manifest["sha256"]
    checks["f1_test"] = math.isclose(
        final_result["advanced_test"]["f1_macro"], model_manifest["test_f1_macro"], abs_tol=1e-12
    )
    checks["f1_validacion"] = math.isclose(
        model_manifest["validation_f1_macro"], 0.877723199040199, abs_tol=1e-12
    )

    notebook_path = repo / "notebooks" / "ETID_Actividad_2.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    notebook_text = notebook_path.read_text(encoding="utf-8").lower()
    checks["notebook_ejecutado"] = bool(code_cells) and all(
        cell.get("execution_count") is not None for cell in code_cells
    )
    checks["notebook_sin_errores"] = not errors
    checks["notebook_sin_ruta_personal"] = (
        re.search(r"c:\\\\+users\\\\+semej", notebook_text) is None
        and "c:/users/semej" not in notebook_text
    )

    required = [
        repo / "README.md",
        repo / "CITATION.cff",
        repo / "LICENSES.md",
        repo / "requirements.txt",
        repo / "docs" / "DATASET_CARD.md",
        repo / "docs" / "MODEL_CARD.md",
        repo / "docs" / "REPRODUCIBILITY.md",
        repo / "data" / "decisiones_calidad.csv",
        repo / "notebooks" / "ETID_Actividad_2.ipynb",
        repo / "models" / "efficientnet_b0_etid_selected.pt",
        repo / "reports" / "INFORME_TECNICO_ACTIVIDAD_2_ENTREGABLE.pdf",
        repo / "reports" / "INFORME_TECNICO_FINAL_ACTIVIDAD_3.md",
        repo / "reports" / "INFORME_TECNICO_FINAL_ACTIVIDAD_3.docx",
        repo / "reports" / "INFORME_TECNICO_FINAL_ACTIVIDAD_3.pdf",
        repo / "src" / "download_etid.py",
        repo / "src" / "prepare_etid.py",
        repo / "src" / "smoke_test.py",
    ]
    checks["archivos_requeridos"] = all(path.is_file() for path in required)
    approved = all(value is True for value in checks.values())
    report = {"resultado": "APROBADO" if approved else "REVISAR", "controles": checks}
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if approved else 1


if __name__ == "__main__":
    raise SystemExit(main())
