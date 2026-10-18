#!/usr/bin/env python3
"""Extrae ETID y valida el dataset contra las particiones canónicas del proyecto."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from zipfile import BadZipFile, ZipFile

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit("Falta Pillow. Ejecute: python -m pip install -r requirements.txt") from exc

CLASSES = {"Battery", "Cable", "Keyboard", "Laptop", "Mobile", "Mouse", "PCB"}
SPLITS = ("train", "validation", "test")
EXPECTED_COUNTS = {"train": 4289, "validation": 983, "test": 995}
EXPECTED_RAW_IMAGES = 6269
EXPECTED_INCLUDED = 6267
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
REQUIRED_COLUMNS = {"ruta", "etiqueta_final", "grupo_fuente", "grupo_particion", "sha256"}


def digest(path: Path, chunk_size: int = 1024 * 1024) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            value.update(block)
    return value.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        columns = set(reader.fieldnames or [])
    if not REQUIRED_COLUMNS <= columns and path.name.endswith(".csv") and "decisiones" not in path.name:
        missing = sorted(REQUIRED_COLUMNS - columns)
        raise RuntimeError(f"Faltan columnas en {path.name}: {missing}")
    return rows


def safe_extract(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    try:
        with ZipFile(zip_path) as archive:
            for member in archive.infolist():
                target = (destination / member.filename).resolve()
                if os.path.commonpath([str(root), str(target)]) != str(root):
                    raise RuntimeError(f"Ruta insegura dentro del ZIP: {member.filename}")
            archive.extractall(destination)
    except BadZipFile as exc:
        raise RuntimeError(f"ZIP inválido: {zip_path}") from exc


def extract_dataset(zip_path: Path, destination: Path) -> Path:
    marker = destination / ".etid_v1_extraccion_completa"
    if not marker.exists():
        print(f"Extrayendo {zip_path.name}...", flush=True)
        with tempfile.TemporaryDirectory(prefix="etid_extract_", dir=destination.parent) as temp:
            staging = Path(temp)
            safe_extract(zip_path, staging)
            nested = sorted(staging.rglob("*.zip"))
            for archive in nested:
                safe_extract(archive, staging)
                archive.unlink()
            source_root = discover_dataset_root(staging)
            destination.mkdir(parents=True, exist_ok=True)
            for class_name in sorted(CLASSES):
                target = destination / class_name
                if target.exists():
                    raise RuntimeError(f"El destino ya contiene {target.name}; revise antes de sobrescribir.")
                shutil.move(str(source_root / class_name), target)
        marker.write_text("10.17632/bndsgcspbx.1\n", encoding="ascii")
    return discover_dataset_root(destination)


def discover_dataset_root(start: Path) -> Path:
    start = start.expanduser().resolve()
    candidates = []
    for directory in [start, *(path for path in start.rglob("*") if path.is_dir())]:
        children = {path.name for path in directory.iterdir() if path.is_dir()}
        if CLASSES <= children:
            candidates.append(directory)
    if len(candidates) != 1:
        raise RuntimeError(
            f"Se esperaba una única raíz con las siete carpetas de clase; se encontraron {len(candidates)}."
        )
    return candidates[0]


def load_splits(split_dir: Path) -> dict[str, list[dict[str, str]]]:
    return {name: read_csv(split_dir / f"{name}.csv") for name in SPLITS}


def validate_manifests(split_dir: Path, verification_path: Path) -> tuple[dict, dict]:
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    rows = load_splits(split_dir)
    failures = []
    actual_hashes = {}
    for name in SPLITS:
        path = split_dir / f"{name}.csv"
        actual_hashes[path.name] = digest(path)
        if len(rows[name]) != EXPECTED_COUNTS[name]:
            failures.append(f"{name}: {len(rows[name])} filas; se esperaban {EXPECTED_COUNTS[name]}")
        expected_hash = verification["sha256_manifiestos"].get(path.name)
        if actual_hashes[path.name] != expected_hash:
            failures.append(f"SHA-256 modificado: {path.name}")

    routes = {name: {row["ruta"] for row in rows[name]} for name in SPLITS}
    hashes = {name: {row["sha256"] for row in rows[name]} for name in SPLITS}
    groups = {name: {row["grupo_particion"] for row in rows[name]} for name in SPLITS}
    labels = {name: {row["etiqueta_final"] for row in rows[name]} for name in SPLITS}
    intersections = {}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            key = f"{left}-{right}"
            intersections[key] = {
                "rutas": len(routes[left] & routes[right]),
                "sha256": len(hashes[left] & hashes[right]),
                "grupos": len(groups[left] & groups[right]),
            }
            if any(intersections[key].values()):
                failures.append(f"Solapamiento entre {left} y {right}: {intersections[key]}")
    if any(labels[name] != CLASSES for name in SPLITS):
        failures.append("No todas las clases están presentes en las tres particiones.")
    if failures:
        raise RuntimeError("; ".join(failures))
    return rows, {"hashes": actual_hashes, "intersecciones": intersections}


def validate_decisions(
    decisions_path: Path, included: dict[str, dict[str, str]], raw_routes: set[str]
) -> dict:
    with decisions_path.open(newline="", encoding="utf-8-sig") as stream:
        decisions = list(csv.DictReader(stream))
    exclusions = {row["ruta"] for row in decisions if row["accion"] == "excluir"}
    relabeled = [row for row in decisions if row["accion"] == "reetiquetar"]
    failures = []
    if len(exclusions) != 2 or len(relabeled) != 9:
        failures.append(f"Decisiones inesperadas: {len(exclusions)} exclusiones y {len(relabeled)} reetiquetados")
    for route in exclusions:
        if route not in raw_routes or route in included:
            failures.append(f"Exclusión incoherente: {route}")
    for row in relabeled:
        manifest_row = included.get(row["ruta"])
        if not manifest_row or manifest_row["etiqueta_final"] != row["etiqueta_final"]:
            failures.append(f"Reetiquetado incoherente: {row['ruta']}")
    if failures:
        raise RuntimeError("; ".join(failures))
    return {"exclusiones": len(exclusions), "reetiquetados": len(relabeled), "decisiones": len(decisions)}


def validate_images(
    data_root: Path, all_rows: list[dict[str, str]], full: bool
) -> tuple[set[str], dict]:
    raw_paths = sorted(
        path for path in data_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    raw_routes = {path.relative_to(data_root).as_posix() for path in raw_paths}
    if len(raw_routes) != EXPECTED_RAW_IMAGES:
        raise RuntimeError(f"Se encontraron {len(raw_routes)} imágenes; se esperaban {EXPECTED_RAW_IMAGES}.")
    included = {row["ruta"]: row for row in all_rows}
    missing = sorted(set(included) - raw_routes)
    if missing:
        raise RuntimeError(f"Faltan {len(missing)} imágenes incluidas; primera: {missing[0]}")

    checked = 0
    if full:
        for index, row in enumerate(all_rows, start=1):
            path = data_root / row["ruta"]
            with Image.open(path) as image:
                image.verify()
            if digest(path) != row["sha256"]:
                raise RuntimeError(f"SHA-256 de imagen no coincide: {row['ruta']}")
            checked += 1
            if index % 500 == 0:
                print(f"Verificadas {index}/{len(all_rows)} imágenes", flush=True)
        excluded_paths = [path for path in raw_paths if path.relative_to(data_root).as_posix() not in included]
        for path in excluded_paths:
            with Image.open(path) as image:
                image.verify()
    return raw_routes, {"imagenes_en_raiz": len(raw_routes), "imagenes_con_sha256_verificado": checked}


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Extrae y valida ETID contra los manifiestos canónicos del proyecto."
    )
    parser.add_argument("--zip", type=Path, help="ZIP ETID v1 que se extraerá de forma segura.")
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Raíz existente del dataset o carpeta que la contiene. También acepta ETID_DATA_ROOT.",
    )
    parser.add_argument(
        "--extract-to",
        type=Path,
        default=repo_root / "data" / "raw" / "ETID",
        help="Carpeta de extracción cuando se usa --zip.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Omite la apertura y el SHA-256 de cada imagen; conserva controles estructurales.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "data" / "processed" / "preparation_report.json",
    )
    args = parser.parse_args()

    if args.zip:
        data_root = extract_dataset(args.zip.expanduser().resolve(), args.extract_to.expanduser().resolve())
    else:
        configured = args.data_root or os.environ.get("ETID_DATA_ROOT") or repo_root / "data" / "raw" / "ETID"
        data_root = discover_dataset_root(Path(configured))

    split_dir = repo_root / "data" / "splits"
    split_rows, manifest_checks = validate_manifests(
        split_dir, split_dir / "verificacion_particiones.json"
    )
    all_rows = [row for name in SPLITS for row in split_rows[name]]
    if len(all_rows) != EXPECTED_INCLUDED:
        raise RuntimeError(f"Los manifiestos contienen {len(all_rows)} filas; se esperaban {EXPECTED_INCLUDED}.")
    included = {row["ruta"]: row for row in all_rows}
    raw_routes, image_checks = validate_images(data_root, all_rows, full=not args.quick)
    decision_checks = validate_decisions(repo_root / "data" / "decisiones_calidad.csv", included, raw_routes)
    unassigned = raw_routes - set(included)
    if len(unassigned) != 2:
        raise RuntimeError(f"Hay {len(unassigned)} imágenes sin asignar; se esperaban las dos exclusiones.")

    distribution = {
        name: dict(Counter(row["etiqueta_final"] for row in split_rows[name])) for name in SPLITS
    }
    report = {
        "resultado": "APROBADO",
        "dataset": "E-Tadweer Image Dataset (ETID)",
        "doi": "10.17632/bndsgcspbx.1",
        "modo": "rápido" if args.quick else "completo",
        "conteos": {name: len(split_rows[name]) for name in SPLITS},
        "distribucion": distribution,
        "decisiones": decision_checks,
        "imagenes": image_checks,
        "manifiestos": manifest_checks,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
