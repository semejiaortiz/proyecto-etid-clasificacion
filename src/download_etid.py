#!/usr/bin/env python3
"""Descarga reanudable y verificable de ETID v1 desde Mendeley Data."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

DATASET_URL = "https://data.mendeley.com/public-api/zip/bndsgcspbx/download/1"
EXPECTED_SHA256 = "751ae9ffdaab724c364ddb9df4fd5f82a022ef1e2bce40fb9802444146e66951"
DEFAULT_NAME = "ETID_v1_original.zip"


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def find_curl() -> str:
    executable = shutil.which("curl") or shutil.which("curl.exe")
    if not executable:
        raise SystemExit(
            "No se encontró curl. Instálelo o descargue ETID v1 manualmente desde "
            "https://doi.org/10.17632/bndsgcspbx.1 y use --local-file."
        )
    return executable


def validate(path: Path) -> bool:
    if not path.is_file():
        return False
    print(f"Calculando SHA-256 de {path.name}...", flush=True)
    value = sha256(path)
    if value == EXPECTED_SHA256:
        print(f"SHA-256 verificado: {value}")
        return True
    print(f"SHA-256 no coincide: {value}", file=sys.stderr)
    return False


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Descarga ETID v1 con reanudación y verifica su SHA-256."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "data" / "raw" / DEFAULT_NAME,
        help="Destino del ZIP (predeterminado: data/raw/ETID_v1_original.zip).",
    )
    parser.add_argument(
        "--local-file",
        type=Path,
        help="Valida y copia un ZIP ya descargado, sin acceder a internet.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Descarta una descarga parcial y comienza de nuevo.",
    )
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.local_file:
        source = args.local_file.expanduser().resolve()
        if not validate(source):
            raise SystemExit("El archivo local no corresponde a ETID v1.")
        if source != output:
            shutil.copy2(source, output)
        print(f"ZIP disponible en: {output}")
        return 0

    if output.exists() and validate(output):
        print("La descarga ya estaba completa; no se realizó ninguna transferencia.")
        return 0
    if args.restart and output.exists():
        output.unlink()

    curl = find_curl()
    command = [
        curl,
        "--location",
        "--fail",
        "--show-error",
        "--retry",
        "5",
        "--retry-all-errors",
        "--connect-timeout",
        "30",
        "--continue-at",
        "-",
        "--output",
        str(output),
        DATASET_URL,
    ]
    print("Descargando ETID v1. La operación puede reanudarse si se interrumpe...", flush=True)
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise SystemExit(
            f"curl terminó con código {result.returncode}. Conserve el archivo parcial y repita el comando."
        )
    if not validate(output):
        raise SystemExit(
            "La descarga terminó, pero su SHA-256 no coincide. Use --restart para repetirla."
        )
    print(f"Descarga completa y verificada: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
