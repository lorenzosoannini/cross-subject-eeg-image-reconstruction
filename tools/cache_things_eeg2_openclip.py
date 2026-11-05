#!/usr/bin/env python3
"""
Genera la cache OpenCLIP image-token per THINGS-EEG2, indicizzata da
visual_index e compatibile con il target visivo usato da ZEBRA.

Configurazione predefinita:
    FrozenOpenCLIPImageEmbedder
    arch="ViT-bigG-14"
    version="laion2b_s39b_b160k"
    output_tokens=True
    only_tokens=True, quando supportato dalla versione locale
    output atteso: [N, 256, 1664]

Le immagini vengono caricate come RGB in [0,1], trasformate in [-1,1] e
preprocessate tramite il metodo `preprocess` dello stesso embedder usato da
ZEBRA. Questo evita di introdurre una pipeline di resize/normalizzazione
diversa.

La cache completa di THINGS-EEG2 contiene 16.740 immagini:
    float16: circa 13.28 GiB
    float32: circa 26.56 GiB

Output
------
OUTPUT_DIR/
├── image_tokens.npy
├── completed.npy
├── visual_index_to_row.npy
├── visual_uid_to_row.json
├── cache_rows.csv
├── metadata.json
└── audit_report.json

Durante la generazione:
    image_tokens.partial.npy

Resume
------
Dopo ogni batch:
- i token vengono scritti nel memmap;
- il memmap viene flushato;
- soltanto dopo viene impostato completed[row] = True.

In caso di interruzione, rilanciare lo stesso comando. Le righe complete
vengono saltate.

Esempi
-------
Dry-run:
    python tools/cache_things_eeg2_openclip.py \
      --root data/THINGS-EEG2

Smoke test di 32 nuove immagini, lasciando la cache pronta al resume:
    python tools/cache_things_eeg2_openclip.py \
      --root data/THINGS-EEG2 \
      --execute \
      --batch-size 2 \
      --stop-after 32

Completamento:
    python tools/cache_things_eeg2_openclip.py \
      --root data/THINGS-EEG2 \
      --execute \
      --batch-size 2

Output su disco locale non sincronizzato:
    python tools/cache_things_eeg2_openclip.py \
      --root data/THINGS-EEG2 \
      --output-dir /path/locale/things_eeg2_openclip_bigG \
      --execute \
      --batch-size 2

Dipendenze
----------
Lo script usa la copia locale di:
    generative_models/sgm/modules/encoders/modules.py

e richiede almeno:
    torch
    pillow
    numpy
    open_clip_torch
    kornia
    einops
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import gc
import hashlib
import inspect
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch

# Consente l'esecuzione diretta con ``python tools/...py``.
SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PROJECT_ROOT))

sys.path.insert(0, str(SCRIPT_PROJECT_ROOT / "generative_models"))

from data_loader.things_eeg2_cache_identity import (
    visual_identity_sha256,
)


SCRIPT_VERSION = "1.2.0"
DEFAULT_ARCH = "ViT-bigG-14"
DEFAULT_VERSION = "laion2b_s39b_b160k"
DEFAULT_SEQ_LEN = 256
DEFAULT_EMBED_DIM = 1664
DEFAULT_STORAGE_DTYPE = "float16"


@dataclass(frozen=True)
class VisualRow:
    row: int
    visual_index: int
    visual_uid: str
    partition: str
    event_id: int
    image_path_raw: str
    image_path: Path
    split: str


# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    return value


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(
            json_ready(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.npy")
    if temporary.exists():
        temporary.unlink()
    np.save(temporary, array)
    temporary.replace(path)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def human_bytes(value: int | float) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TiB"


def nearest_existing_parent(path: Path) -> Path:
    current = path.expanduser().resolve()
    while not current.exists():
        if current.parent == current:
            raise RuntimeError(f"Nessun parent esistente per {path}")
        current = current.parent
    return current


def project_root_from_script() -> Path:
    # tools/cache_things_eeg2_openclip.py -> project root
    return Path(__file__).resolve().parents[1]


def storage_numpy_dtype(name: str) -> np.dtype:
    mapping = {
        "float16": np.dtype(np.float16),
        "float32": np.dtype(np.float32),
    }
    return mapping[name]


def torch_dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    return mapping[name]


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        device = torch.device(name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA richiesto ma non disponibile.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS richiesto ma non disponibile.")
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_compute_dtype(
    precision: str,
    device: torch.device,
) -> tuple[str, torch.dtype]:
    if precision != "auto":
        name = precision
    elif device.type == "cuda":
        name = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    elif device.type == "mps":
        name = "fp16"
    else:
        name = "fp32"

    dtype = torch_dtype_from_name(name)
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("Su CPU usare --precision fp32.")
    return name, dtype


def autocast_context(
    device: torch.device,
    dtype: torch.dtype,
):
    if device.type in {"cuda", "mps"} and dtype != torch.float32:
        return torch.autocast(
            device_type=device.type,
            dtype=dtype,
        )
    return contextlib.nullcontext()


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "final_tokens": output_dir / "image_tokens.npy",
        "partial_tokens": output_dir / "image_tokens.partial.npy",
        "completed": output_dir / "completed.npy",
        "visual_index_to_row": output_dir / "visual_index_to_row.npy",
        "visual_uid_to_row": output_dir / "visual_uid_to_row.json",
        "rows_csv": output_dir / "cache_rows.csv",
        "metadata": output_dir / "metadata.json",
        "audit": output_dir / "audit_report.json",
    }


# -----------------------------------------------------------------------------
# Visual index
# -----------------------------------------------------------------------------


def resolve_image_path(
    raw_path: str,
    *,
    root: Path,
    index_dir: Path,
    images_root: Path | None,
) -> Path:
    raw = Path(raw_path).expanduser()
    candidates: list[Path] = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                root / raw,
                index_dir.parent / raw,
                index_dir / raw,
            ]
        )
        if images_root is not None:
            candidates.extend(
                [
                    images_root / raw,
                    images_root / raw.name,
                ]
            )

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate

    formatted = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        f"Immagine non trovata per {raw_path!r}. Candidati:\n{formatted}"
    )


def read_visual_rows(
    visuals_path: Path,
    *,
    root: Path,
    index_dir: Path,
    images_root: Path | None,
) -> list[VisualRow]:
    if not visuals_path.is_file():
        raise FileNotFoundError(visuals_path)

    parsed: list[dict[str, str]] = []
    with visuals_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {
            "visual_index",
            "visual_uid",
            "partition",
            "event_id",
            "image_path",
            "split",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"visuals.csv senza colonne richieste: {sorted(missing)}")
        parsed.extend(reader)

    parsed.sort(key=lambda row: int(row["visual_index"]))
    indices = [int(row["visual_index"]) for row in parsed]
    if indices != list(range(len(parsed))):
        raise ValueError("visual_index deve essere contiguo da 0 a N-1.")

    visual_uids = [row["visual_uid"] for row in parsed]
    if len(set(visual_uids)) != len(visual_uids):
        raise ValueError("visual_uid duplicati.")

    rows: list[VisualRow] = []
    seen_paths: dict[Path, str] = {}
    for row_number, row in enumerate(parsed):
        image_path = resolve_image_path(
            row["image_path"],
            root=root,
            index_dir=index_dir,
            images_root=images_root,
        )
        previous_uid = seen_paths.get(image_path)
        if previous_uid is not None:
            raise ValueError(
                f"Immagine duplicata tra {previous_uid} e "
                f"{row['visual_uid']}: {image_path}"
            )
        seen_paths[image_path] = row["visual_uid"]

        rows.append(
            VisualRow(
                row=row_number,
                visual_index=int(row["visual_index"]),
                visual_uid=row["visual_uid"],
                partition=row["partition"],
                event_id=int(row["event_id"]),
                image_path_raw=row["image_path"],
                image_path=image_path,
                split=row["split"],
            )
        )

    return rows


def image_files_signature(rows: Sequence[VisualRow]) -> str:
    """Portable size-level audit, independent of split and machine root."""
    digest = hashlib.sha256()
    for row in rows:
        stat = row.image_path.stat()
        payload = (
            f"{row.visual_index}\t{row.visual_uid}\t"
            f"{row.image_path_raw}\t{stat.st_size}\n"
        )
        digest.update(payload.encode("utf-8"))
    return digest.hexdigest()


def write_cache_mappings(
    rows: Sequence[VisualRow],
    paths: dict[str, Path],
) -> None:
    visual_index_to_row = np.full(
        max(row.visual_index for row in rows) + 1,
        -1,
        dtype=np.int32,
    )
    for row in rows:
        visual_index_to_row[row.visual_index] = row.row
    if np.any(visual_index_to_row < 0):
        raise RuntimeError("Mapping visual_index->row incompleto.")

    atomic_save_npy(
        paths["visual_index_to_row"],
        visual_index_to_row,
    )
    atomic_json(
        paths["visual_uid_to_row"],
        {row.visual_uid: row.row for row in rows},
    )

    temporary = paths["rows_csv"].with_suffix(".csv.partial")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "row",
            "visual_index",
            "visual_uid",
            "partition",
            "event_id",
            "split",
            "image_path",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "row": row.row,
                    "visual_index": row.visual_index,
                    "visual_uid": row.visual_uid,
                    "partition": row.partition,
                    "event_id": row.event_id,
                    "split": row.split,
                    "image_path": str(row.image_path),
                }
            )
    temporary.replace(paths["rows_csv"])


# -----------------------------------------------------------------------------
# Immagini
# -----------------------------------------------------------------------------


def load_rgb_minus_one_to_one(path: Path) -> torch.Tensor:
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            array = np.asarray(image, dtype=np.uint8)
    except (OSError, UnidentifiedImageError) as exc:
        raise RuntimeError(f"Impossibile leggere {path}: {exc}") from exc

    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Shape immagine inattesa {array.shape}: {path}")
    if array.shape[0] < 2 or array.shape[1] < 2:
        raise ValueError(f"Immagine troppo piccola {array.shape}: {path}")

    # copy() evita tensori non scrivibili provenienti da PIL/NumPy.
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1)
    tensor = tensor.to(dtype=torch.float32).div_(127.5).sub_(1.0)
    return tensor


def verify_image_headers(
    rows: Sequence[VisualRow],
    *,
    workers: int,
) -> None:
    def verify(row: VisualRow) -> None:
        try:
            with Image.open(row.image_path) as image:
                image.verify()
        except Exception as exc:
            raise RuntimeError(
                f"Verifica immagine fallita per {row.visual_uid}: "
                f"{row.image_path}: {exc}"
            ) from exc

    if workers <= 1:
        for row in rows:
            verify(row)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(verify, rows))


def load_image_batch(
    rows: Sequence[VisualRow],
    *,
    workers: int,
) -> list[torch.Tensor]:
    paths = [row.image_path for row in rows]
    if workers <= 1:
        return [load_rgb_minus_one_to_one(path) for path in paths]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(load_rgb_minus_one_to_one, paths))


# -----------------------------------------------------------------------------
# Modello
# -----------------------------------------------------------------------------


def import_embedder(project_root: Path):
    project_root = project_root.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from sgm.modules.encoders.modules import (
            FrozenOpenCLIPImageEmbedder,
        )
    except Exception as exc:
        raise RuntimeError(
            "Impossibile importare FrozenOpenCLIPImageEmbedder dalla "
            "cartella generative_models del progetto. Esegui lo script "
            "dalla root ZEBRA e verifica le dipendenze."
        ) from exc

    return FrozenOpenCLIPImageEmbedder


def instantiate_embedder(
    embedder_class,
    *,
    arch: str,
    version: str,
    device: torch.device,
    cache_dir: Path | None,
):
    signature = inspect.signature(embedder_class.__init__)
    supported = set(signature.parameters)

    candidate_kwargs: dict[str, Any] = {
        "arch": arch,
        "version": version,
        "device": str(device),
        "freeze": True,
        "output_tokens": True,
        "only_tokens": True,
        "init_device": "cpu",
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
    }
    kwargs = {
        key: value
        for key, value in candidate_kwargs.items()
        if key in supported and value is not None
    }

    print("[OPENCLIP] Costruttore supportato: " + ", ".join(sorted(kwargs)))
    embedder = embedder_class(**kwargs)
    embedder.eval()
    return embedder, kwargs, str(signature)


def extract_token_tensor(
    output: Any,
    *,
    batch_size: int,
    expected_seq_len: int,
    expected_embed_dim: int,
) -> torch.Tensor:
    candidates: list[tuple[str, torch.Tensor]] = []

    def visit(value: Any, path: str) -> None:
        if torch.is_tensor(value):
            candidates.append((path, value))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{path}.{key}")

    visit(output, "output")

    exact = [
        (path, tensor)
        for path, tensor in candidates
        if tuple(tensor.shape) == (batch_size, expected_seq_len, expected_embed_dim)
    ]
    if len(exact) == 1:
        return exact[0][1]
    if len(exact) > 1:
        # Identical references are harmless; otherwise refuse ambiguity.
        first = exact[0][1]
        if all(tensor.data_ptr() == first.data_ptr() for _, tensor in exact):
            return first
        raise RuntimeError(
            "Più tensori candidati con shape token esatta: "
            + ", ".join(path for path, _ in exact)
        )

    descriptions = [
        f"{path}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}"
        for path, tensor in candidates
    ]
    raise RuntimeError(
        "Output OpenCLIP senza token con shape attesa "
        f"[{batch_size},{expected_seq_len},{expected_embed_dim}]. "
        "Tensori trovati:\n  " + "\n  ".join(descriptions)
    )


def preprocess_individual_images(
    embedder,
    images: Sequence[torch.Tensor],
    *,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Applica lo stesso contesto autocast usato dal visual encoder anche al
    preprocessing. Questo è importante su MPS: il forward ufficiale
    dell'embedder esegue resize e normalizzazione dentro il contesto autocast,
    mentre un preprocess forzato in FP32 può produrre differenze visibili in
    FP16 pur mantenendo cosine similarity molto elevata.
    """
    processed: list[torch.Tensor] = []
    for image in images:
        image = image.unsqueeze(0).to(
            device=device,
            dtype=torch.float32,
            non_blocking=device.type == "cuda",
        )
        with autocast_context(device, compute_dtype):
            preprocessed = embedder.preprocess(image)
        if tuple(preprocessed.shape[1:]) != (3, 224, 224):
            raise RuntimeError(f"Preprocess shape inattesa: {preprocessed.shape}")
        processed.append(preprocessed)
    return torch.cat(processed, dim=0)


def encode_preprocessed_batch(
    embedder,
    preprocessed: torch.Tensor,
    *,
    device: torch.device,
    compute_dtype: torch.dtype,
    expected_seq_len: int,
    expected_embed_dim: int,
) -> torch.Tensor:
    pixel_values = preprocessed.to(
        device=device,
        dtype=compute_dtype,
        non_blocking=device.type == "cuda",
    )

    with torch.inference_mode(), autocast_context(device, compute_dtype):
        output = embedder.model.visual(pixel_values)
        tokens = extract_token_tensor(
            output,
            batch_size=pixel_values.shape[0],
            expected_seq_len=expected_seq_len,
            expected_embed_dim=expected_embed_dim,
        )

    if not torch.isfinite(tokens).all():
        raise FloatingPointError("NaN/Inf nei token OpenCLIP.")
    return tokens


def model_self_check(
    embedder,
    image: torch.Tensor,
    *,
    device: torch.device,
    compute_dtype: torch.dtype,
    expected_seq_len: int,
    expected_embed_dim: int,
) -> dict[str, Any]:
    preprocessed = preprocess_individual_images(
        embedder,
        [image],
        device=device,
        compute_dtype=compute_dtype,
    )
    direct = encode_preprocessed_batch(
        embedder,
        preprocessed,
        device=device,
        compute_dtype=compute_dtype,
        expected_seq_len=expected_seq_len,
        expected_embed_dim=expected_embed_dim,
    )

    image_batch = image.unsqueeze(0).to(
        device=device,
        dtype=torch.float32,
    )
    with torch.inference_mode(), autocast_context(device, compute_dtype):
        if device.type == "mps":
            # Il decorator @autocast della copia Stability AI è CUDA-only.
            # Su MPS richiama torch.cuda.amp.autocast, lo disabilita e genera
            # un percorso numerico diverso. Usiamo il corpo vision ufficiale
            # sotto il nostro autocast MPS device-aware.
            reference_output = embedder.encode_with_vision_transformer(image_batch)
            reference_path = "encode_with_vision_transformer"
        else:
            reference_output = embedder(image_batch)
            reference_path = "embedder.forward"

        reference_tokens = extract_token_tensor(
            reference_output,
            batch_size=1,
            expected_seq_len=expected_seq_len,
            expected_embed_dim=expected_embed_dim,
        )

    direct32 = direct.float()
    reference32 = reference_tokens.float()
    difference = direct32 - reference32
    maximum_error = float(difference.abs().max().item())
    mean_error = float(difference.abs().mean().item())
    relative_l2_error = float(
        difference.norm().div(reference32.norm().clamp_min(1e-12)).item()
    )
    cosine = float(
        torch.nn.functional.cosine_similarity(
            direct32.flatten(1),
            reference32.flatten(1),
            dim=1,
        ).item()
    )

    is_mps_fp16 = device.type == "mps" and compute_dtype == torch.float16
    minimum_cosine = 0.9995 if is_mps_fp16 else 0.99999
    maximum_relative_l2 = 0.03 if is_mps_fp16 else 0.001
    passed = cosine >= minimum_cosine and relative_l2_error <= maximum_relative_l2

    if not passed:
        raise RuntimeError(
            "Self-check fallito: il percorso batch diretto non è "
            "numericamente compatibile con il riferimento. "
            f"device={device}, dtype={compute_dtype}, "
            f"reference={reference_path}, cosine={cosine}, "
            f"relative_l2_error={relative_l2_error}, "
            f"max_abs_error={maximum_error}, "
            f"thresholds=(cos>={minimum_cosine}, "
            f"rel_l2<={maximum_relative_l2})"
        )

    return {
        "direct_shape": list(direct.shape),
        "reference_shape": list(reference_tokens.shape),
        "reference_path": reference_path,
        "device": str(device),
        "compute_dtype": str(compute_dtype),
        "maximum_absolute_error": maximum_error,
        "mean_absolute_error": mean_error,
        "relative_l2_error": relative_l2_error,
        "flattened_cosine_similarity": cosine,
        "minimum_cosine_threshold": minimum_cosine,
        "maximum_relative_l2_threshold": maximum_relative_l2,
        "mps_fp16_tolerance_used": is_mps_fp16,
        "passed": True,
    }


# -----------------------------------------------------------------------------
# Cache
# -----------------------------------------------------------------------------


def cache_signature(
    *,
    visuals_path: Path,
    rows: Sequence[VisualRow],
    image_signature: str,
    arch: str,
    version: str,
    expected_seq_len: int,
    expected_embed_dim: int,
    storage_dtype: str,
    compute_precision: str,
    compute_device_type: str,
) -> dict[str, Any]:
    return {
        "signature_schema_version": 2,
        "visual_identity_sha256": visual_identity_sha256(visuals_path),
        "image_files_signature": image_signature,
        "n_images": len(rows),
        "visual_index_min": min(row.visual_index for row in rows),
        "visual_index_max": max(row.visual_index for row in rows),
        "arch": arch,
        "version": version,
        "expected_seq_len": expected_seq_len,
        "expected_embed_dim": expected_embed_dim,
        "storage_dtype": storage_dtype,
        "compute_precision": compute_precision,
        "compute_device_type": compute_device_type,
        "input_range": "[-1,1]",
        "preprocess": (
            "FrozenOpenCLIPImageEmbedder.preprocess applied independently "
            "per image, then batched visual forward"
        ),
    }


def signature_hash(signature: dict[str, Any]) -> str:
    payload = json.dumps(
        signature,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prepare_output_directory(
    output_dir: Path,
    *,
    overwrite: bool,
) -> None:
    if overwrite and output_dir.exists():
        backup = output_dir.with_name(output_dir.name + ".previous")
        if backup.exists():
            shutil.rmtree(backup)
        output_dir.replace(backup)
        print(f"[BACKUP] Cache precedente spostata in {backup}")

    output_dir.mkdir(parents=True, exist_ok=True)


def initialize_or_resume_cache(
    paths: dict[str, Path],
    *,
    signature: dict[str, Any],
    shape: tuple[int, int, int],
    storage_dtype: np.dtype,
    overwrite: bool,
) -> tuple[np.memmap, np.memmap, dict[str, Any]]:
    if overwrite:
        for path in paths.values():
            if path.exists() and path.is_file():
                path.unlink()

    expected_signature_hash = signature_hash(signature)

    if paths["final_tokens"].is_file():
        if not paths["metadata"].is_file():
            raise RuntimeError("image_tokens.npy esiste senza metadata.json.")
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if metadata.get("signature_hash") != expected_signature_hash:
            raise RuntimeError(
                "La cache completa esistente ha una firma diversa. "
                "Usa --overwrite o un altro --output-dir."
            )
        if metadata.get("status") != "complete":
            raise RuntimeError("image_tokens.npy esiste ma metadata non è complete.")
        array = np.load(paths["final_tokens"], mmap_mode="r")
        if tuple(array.shape) != shape or array.dtype != storage_dtype:
            raise RuntimeError("Cache finale con shape/dtype errati.")
        completed = np.load(paths["completed"], mmap_mode="r")
        if not np.all(completed):
            raise RuntimeError("Cache finale con completed incompleto.")
        return array, completed, metadata

    if paths["partial_tokens"].exists():
        if not paths["metadata"].is_file():
            raise RuntimeError("Cache parziale presente senza metadata.json.")
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if metadata.get("signature_hash") != expected_signature_hash:
            raise RuntimeError(
                "Cache parziale con firma diversa. Usa --overwrite "
                "o un altro --output-dir."
            )

        tokens = np.load(paths["partial_tokens"], mmap_mode="r+")
        completed = np.load(paths["completed"], mmap_mode="r+")
        if tuple(tokens.shape) != shape or tokens.dtype != storage_dtype:
            raise RuntimeError("Cache parziale con shape/dtype errati.")
        if completed.shape != (shape[0],) or completed.dtype != np.bool_:
            raise RuntimeError("completed.npy incompatibile.")
        return tokens, completed, metadata

    tokens = np.lib.format.open_memmap(
        paths["partial_tokens"],
        mode="w+",
        dtype=storage_dtype,
        shape=shape,
    )
    completed = np.lib.format.open_memmap(
        paths["completed"],
        mode="w+",
        dtype=np.bool_,
        shape=(shape[0],),
    )
    completed[:] = False
    completed.flush()
    tokens.flush()

    metadata = {
        "status": "in_progress",
        "generator_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "signature": signature,
        "signature_hash": expected_signature_hash,
        "shape": list(shape),
        "storage_dtype": str(storage_dtype),
        "completed_rows": 0,
    }
    atomic_json(paths["metadata"], metadata)
    return tokens, completed, metadata


def batched(values: Sequence[int], size: int) -> Iterator[list[int]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def finalize_cache(
    tokens: np.memmap,
    completed: np.memmap,
    paths: dict[str, Path],
    metadata: dict[str, Any],
) -> None:
    if not np.all(completed):
        raise RuntimeError("Tentativo di finalizzare cache incompleta.")

    tokens.flush()
    completed.flush()
    del tokens
    gc.collect()

    paths["partial_tokens"].replace(paths["final_tokens"])
    metadata.update(
        {
            "status": "complete",
            "completed_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
            "completed_rows": int(len(completed)),
            "tokens_path": str(paths["final_tokens"].resolve()),
            "tokens_file_size_bytes": paths["final_tokens"].stat().st_size,
        }
    )
    atomic_json(paths["metadata"], metadata)


def audit_final_cache(
    tokens_path: Path,
    rows: Sequence[VisualRow],
    *,
    chunk_rows: int,
    random_seed: int,
) -> dict[str, Any]:
    array = np.load(tokens_path, mmap_mode="r")
    n_rows = len(array)

    total_values = 0
    total_sum = 0.0
    total_square_sum = 0.0
    minimum = float("inf")
    maximum = float("-inf")
    nonfinite = 0
    zero_rows = 0
    row_norm_min = float("inf")
    row_norm_max = float("-inf")
    row_norm_sum = 0.0

    for start in range(0, n_rows, chunk_rows):
        stop = min(start + chunk_rows, n_rows)
        chunk = np.asarray(array[start:stop], dtype=np.float32)
        finite = np.isfinite(chunk)
        nonfinite += int(chunk.size - finite.sum())

        if not finite.all():
            # Le statistiche successive non devono nascondere NaN/Inf.
            raise FloatingPointError(
                f"NaN/Inf nella cache alle righe [{start},{stop})."
            )

        flat = chunk.reshape(len(chunk), -1)
        row_norms = np.linalg.norm(flat, axis=1)
        zero_rows += int(np.count_nonzero(row_norms == 0.0))
        row_norm_min = min(row_norm_min, float(row_norms.min()))
        row_norm_max = max(row_norm_max, float(row_norms.max()))
        row_norm_sum += float(row_norms.sum())

        total_values += chunk.size
        total_sum += float(chunk.sum(dtype=np.float64))
        total_square_sum += float(
            np.square(chunk, dtype=np.float64).sum(dtype=np.float64)
        )
        minimum = min(minimum, float(chunk.min()))
        maximum = max(maximum, float(chunk.max()))

    mean = total_sum / total_values
    variance = max(
        total_square_sum / total_values - mean * mean,
        0.0,
    )

    rng = np.random.default_rng(random_seed)
    sample_count = min(16, n_rows)
    sample_rows = np.sort(rng.choice(n_rows, size=sample_count, replace=False))
    fingerprints = []
    for row_index in sample_rows:
        row_array = np.asarray(array[int(row_index)])
        fingerprints.append(
            {
                "row": int(row_index),
                "visual_index": rows[int(row_index)].visual_index,
                "visual_uid": rows[int(row_index)].visual_uid,
                "sha256": sha256_array(row_array),
                "mean": float(row_array.astype(np.float32).mean()),
                "std": float(row_array.astype(np.float32).std()),
                "l2_norm": float(
                    np.linalg.norm(row_array.astype(np.float32).reshape(-1))
                ),
            }
        )

    return {
        "status": "complete",
        "created_at_utc": utc_now(),
        "tokens_path": str(tokens_path.resolve()),
        "tokens_file_size_bytes": tokens_path.stat().st_size,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "n_rows": n_rows,
        "total_values": total_values,
        "nonfinite_values": nonfinite,
        "all_zero_rows": zero_rows,
        "value_mean": mean,
        "value_std": math.sqrt(variance),
        "value_min": minimum,
        "value_max": maximum,
        "row_l2_norm_min": row_norm_min,
        "row_l2_norm_mean": row_norm_sum / n_rows,
        "row_l2_norm_max": row_norm_max,
        "sample_fingerprints": fingerprints,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera cache OpenCLIP ViT-bigG-14 [N,256,1664] "
            "indicizzata da visual_index."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Radice data/THINGS-EEG2.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="Default: ROOT/zebra_reve_index.",
    )
    parser.add_argument(
        "--visuals-path",
        type=Path,
        default=None,
        help="Default: INDEX_DIR/visuals.csv.",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=None,
        help="Percorso opzionale per risolvere image_path relativi.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: ROOT/clip_openclip_bigG.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help=(
            "Root ZEBRA contenente generative_models/. "
            "Default: parent della cartella tools."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache dei pesi OpenCLIP, se supportata dall'embedder locale.",
    )
    parser.add_argument("--arch", default=DEFAULT_ARCH)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument(
        "--expected-seq-len",
        type=int,
        default=DEFAULT_SEQ_LEN,
    )
    parser.add_argument(
        "--expected-embed-dim",
        type=int,
        default=DEFAULT_EMBED_DIM,
    )
    parser.add_argument(
        "--storage-dtype",
        choices=["float16", "float32"],
        default=DEFAULT_STORAGE_DTYPE,
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cuda, cuda:0, mps o cpu.",
    )
    parser.add_argument(
        "--precision",
        choices=["auto", "fp32", "fp16", "bf16"],
        default="auto",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--decode-workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--reserve-gib",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--verify-all-images",
        action="store_true",
        help="Apre e verifica tutte le immagini durante il dry-run.",
    )
    parser.add_argument(
        "--skip-self-check",
        action="store_true",
        help="Salta il confronto con embedder.forward sulla prima immagine.",
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        default=None,
        help=(
            "Numero massimo di nuove immagini da elaborare in questa "
            "esecuzione; utile per smoke test e resume."
        ),
    )
    parser.add_argument(
        "--audit-chunk-rows",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Senza questo flag esegue soltanto audit e stima spazio.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()

    if args.batch_size < 1:
        raise SystemExit("--batch-size deve essere >= 1.")
    if args.decode_workers < 0:
        raise SystemExit("--decode-workers deve essere >= 0.")
    if args.stop_after is not None and args.stop_after < 1:
        raise SystemExit("--stop-after deve essere >= 1.")
    if args.audit_chunk_rows < 1:
        raise SystemExit("--audit-chunk-rows deve essere >= 1.")

    root = args.root.expanduser().resolve()
    index_dir = (
        args.index_dir.expanduser().resolve()
        if args.index_dir is not None
        else root / "zebra_reve_index"
    )
    visuals_path = (
        args.visuals_path.expanduser().resolve()
        if args.visuals_path is not None
        else index_dir / "visuals.csv"
    )
    images_root = (
        args.images_root.expanduser().resolve()
        if args.images_root is not None
        else None
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / "clip_openclip_bigG"
    )
    project_root = (
        args.project_root.expanduser().resolve()
        if args.project_root is not None
        else project_root_from_script()
    )
    cache_dir = (
        args.cache_dir.expanduser().resolve() if args.cache_dir is not None else None
    )

    print(f"[MODE]       {'EXECUTE' if args.execute else 'DRY-RUN'}")
    print(f"[ROOT]       {root}")
    print(f"[INDEX]      {index_dir}")
    print(f"[VISUALS]    {visuals_path}")
    print(f"[OUTPUT]     {output_dir}")
    print(f"[PROJECT]    {project_root}")
    print(f"[MODEL]      {args.arch} / {args.version}")
    print(
        f"[TARGET]     [N,{args.expected_seq_len},"
        f"{args.expected_embed_dim}] {args.storage_dtype}"
    )

    rows = read_visual_rows(
        visuals_path,
        root=root,
        index_dir=index_dir,
        images_root=images_root,
    )
    partition_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    for row in rows:
        partition_counts[row.partition] = partition_counts.get(row.partition, 0) + 1
        split_counts[row.split] = split_counts.get(row.split, 0) + 1

    print(f"[IMAGES]     {len(rows)}")
    print(f"[PARTITION]  {partition_counts}")
    print(f"[SPLIT]      {split_counts}")

    image_signature = image_files_signature(rows)

    device = resolve_device(args.device)
    precision_name, compute_dtype = resolve_compute_dtype(
        args.precision,
        device,
    )
    print(
        f"[DEVICE]     {device} | precision={precision_name} | "
        f"torch={torch.__version__}"
    )

    signature = cache_signature(
        visuals_path=visuals_path,
        rows=rows,
        image_signature=image_signature,
        arch=args.arch,
        version=args.version,
        expected_seq_len=args.expected_seq_len,
        expected_embed_dim=args.expected_embed_dim,
        storage_dtype=args.storage_dtype,
        compute_precision=precision_name,
        compute_device_type=device.type,
    )

    storage_dtype = storage_numpy_dtype(args.storage_dtype)
    expected_shape = (
        len(rows),
        args.expected_seq_len,
        args.expected_embed_dim,
    )
    expected_bytes = int(
        np.prod(expected_shape, dtype=np.int64) * storage_dtype.itemsize
    )

    output_parent = nearest_existing_parent(output_dir)
    disk = shutil.disk_usage(output_parent)
    reserve_bytes = int(args.reserve_gib * 1024**3)
    print(f"[SIZE]       {human_bytes(expected_bytes)}")
    print(
        f"[DISK]       free={human_bytes(disk.free)} | "
        f"reserve={human_bytes(reserve_bytes)}"
    )

    if args.verify_all_images:
        print("[VERIFY]     Verifica header di tutte le immagini")
        verify_image_headers(
            rows,
            workers=args.decode_workers,
        )
        print("[VERIFY]     Tutte le immagini sono leggibili")

    if not args.execute:
        print("\n[DRY-RUN COMPLETE]")
        print("Nessun modello caricato e nessun file scritto.")
        print("Per uno smoke test con resume aggiungi:\n" "  --execute --stop-after 32")
        return 0

    paths = output_paths(output_dir)
    prepare_output_directory(
        output_dir,
        overwrite=args.overwrite,
    )

    # Se la cache non esiste ancora, deve esserci spazio per il memmap pieno.
    if (
        not paths["final_tokens"].exists()
        and not paths["partial_tokens"].exists()
        and disk.free < expected_bytes + reserve_bytes
    ):
        raise RuntimeError(
            "Spazio insufficiente per la cache: "
            f"liberi {human_bytes(disk.free)}, richiesti almeno "
            f"{human_bytes(expected_bytes + reserve_bytes)}."
        )

    write_cache_mappings(rows, paths)
    tokens_memmap, completed_memmap, metadata = initialize_or_resume_cache(
        paths,
        signature=signature,
        shape=expected_shape,
        storage_dtype=storage_dtype,
        overwrite=False,
    )

    already_complete = int(np.count_nonzero(completed_memmap))
    print(
        f"[PROGRESS]   complete={already_complete}/{len(rows)} "
        f"({already_complete / len(rows):.2%})"
    )

    if paths["final_tokens"].is_file():
        print("[CACHE]      Cache già completa.")
        if not paths["audit"].is_file():
            audit = audit_final_cache(
                paths["final_tokens"],
                rows,
                chunk_rows=args.audit_chunk_rows,
                random_seed=args.seed,
            )
            atomic_json(paths["audit"], audit)
        print(f"[COMPLETE]   {paths['final_tokens']}")
        return 0

    # device, precision_name, compute_dtype already resolved above
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    Embedder = import_embedder(project_root)
    embedder, constructor_kwargs, constructor_signature = instantiate_embedder(
        Embedder,
        arch=args.arch,
        version=args.version,
        device=device,
        cache_dir=cache_dir,
    )
    embedder.to(device)
    embedder.device = str(device)

    # Riduce la VRAM del visual encoder. Il preprocess resta float32.
    embedder.model.visual.to(
        device=device,
        dtype=compute_dtype,
    )
    embedder.model.visual.eval()

    try:
        import open_clip

        open_clip_version = getattr(open_clip, "__version__", None)
    except Exception:
        open_clip_version = None

    metadata.update(
        {
            "updated_at_utc": utc_now(),
            "device": str(device),
            "precision": precision_name,
            "compute_dtype": str(compute_dtype),
            "torch_version": torch.__version__,
            "open_clip_version": open_clip_version,
            "embedder_class": (f"{Embedder.__module__}.{Embedder.__name__}"),
            "embedder_constructor_signature": constructor_signature,
            "embedder_constructor_kwargs": constructor_kwargs,
            "visual_parameter_count": int(
                sum(
                    parameter.numel()
                    for parameter in embedder.model.visual.parameters()
                )
            ),
        }
    )

    first_incomplete = np.flatnonzero(~completed_memmap)
    if len(first_incomplete) == 0:
        finalize_cache(
            tokens_memmap,
            completed_memmap,
            paths,
            metadata,
        )
        audit = audit_final_cache(
            paths["final_tokens"],
            rows,
            chunk_rows=args.audit_chunk_rows,
            random_seed=args.seed,
        )
        atomic_json(paths["audit"], audit)
        print(f"[COMPLETE]   {paths['final_tokens']}")
        return 0

    if not args.skip_self_check:
        print("[SELF-CHECK] Confronto batch path vs embedder.forward")
        first_image = load_rgb_minus_one_to_one(
            rows[int(first_incomplete[0])].image_path
        )
        self_check = model_self_check(
            embedder,
            first_image,
            device=device,
            compute_dtype=compute_dtype,
            expected_seq_len=args.expected_seq_len,
            expected_embed_dim=args.expected_embed_dim,
        )
        metadata["self_check"] = self_check
        atomic_json(paths["metadata"], metadata)
        print(
            "[SELF-CHECK] passed | "
            f"reference={self_check['reference_path']} | "
            f"cos={self_check['flattened_cosine_similarity']:.8f} | "
            f"rel_l2={self_check['relative_l2_error']:.6g} | "
            f"max_err={self_check['maximum_absolute_error']:.6g}"
        )
        del first_image
        if device.type == "cuda":
            torch.cuda.empty_cache()

    incomplete_indices = np.flatnonzero(~completed_memmap).tolist()
    if args.stop_after is not None:
        incomplete_indices = incomplete_indices[: args.stop_after]

    processed_this_run = 0
    total_to_process = len(incomplete_indices)
    run_started = time.time()

    print(f"[ENCODE]     nuove immagini in questa esecuzione: " f"{total_to_process}")

    try:
        for batch_number, batch_indices in enumerate(
            batched(incomplete_indices, args.batch_size),
            start=1,
        ):
            batch_rows = [rows[index] for index in batch_indices]
            image_tensors = load_image_batch(
                batch_rows,
                workers=args.decode_workers,
            )
            preprocessed = preprocess_individual_images(
                embedder,
                image_tensors,
                device=device,
                compute_dtype=compute_dtype,
            )
            token_tensor = encode_preprocessed_batch(
                embedder,
                preprocessed,
                device=device,
                compute_dtype=compute_dtype,
                expected_seq_len=args.expected_seq_len,
                expected_embed_dim=args.expected_embed_dim,
            )
            token_array = (
                token_tensor.detach()
                .to(dtype=torch.float32)
                .cpu()
                .numpy()
                .astype(storage_dtype, copy=False)
            )

            if token_array.shape != (
                len(batch_indices),
                args.expected_seq_len,
                args.expected_embed_dim,
            ):
                raise RuntimeError(f"Shape batch token errata: {token_array.shape}")
            if not np.isfinite(token_array).all():
                raise FloatingPointError("NaN/Inf dopo cast storage.")

            tokens_memmap[batch_indices] = token_array
            tokens_memmap.flush()

            # Il completion flag viene scritto soltanto dopo il flush token.
            completed_memmap[batch_indices] = True
            completed_memmap.flush()

            processed_this_run += len(batch_indices)
            completed_total = int(np.count_nonzero(completed_memmap))
            elapsed = time.time() - run_started
            rate = processed_this_run / max(elapsed, 1e-9)
            remaining = len(rows) - completed_total
            eta_seconds = remaining / max(rate, 1e-9)

            metadata.update(
                {
                    "status": "in_progress",
                    "updated_at_utc": utc_now(),
                    "completed_rows": completed_total,
                    "last_completed_rows": batch_indices,
                    "processing_rate_images_per_second": rate,
                }
            )
            atomic_json(paths["metadata"], metadata)

            print(
                f"[BATCH {batch_number:05d}] "
                f"+{len(batch_indices)} | "
                f"{completed_total}/{len(rows)} "
                f"({completed_total / len(rows):.2%}) | "
                f"{rate:.3f} img/s | "
                f"ETA {eta_seconds / 3600.0:.2f} h",
                flush=True,
            )

            del (
                image_tensors,
                preprocessed,
                token_tensor,
                token_array,
            )
            gc.collect()
    except KeyboardInterrupt:
        metadata.update(
            {
                "status": "in_progress",
                "updated_at_utc": utc_now(),
                "completed_rows": int(np.count_nonzero(completed_memmap)),
                "last_exit": "KeyboardInterrupt",
            }
        )
        atomic_json(paths["metadata"], metadata)
        print("\n[INTERRUPTED] Cache salvata e pronta al resume.")
        return 130

    completed_total = int(np.count_nonzero(completed_memmap))
    metadata.update(
        {
            "updated_at_utc": utc_now(),
            "completed_rows": completed_total,
            "last_run_processed_rows": processed_this_run,
            "total_processing_seconds": float(time.time() - started),
        }
    )

    if completed_total < len(rows):
        metadata["status"] = "in_progress"
        atomic_json(paths["metadata"], metadata)
        print("\n===== SMOKE/RESUME CHECKPOINT =====")
        print(
            f"Completate: {completed_total}/{len(rows)} "
            f"({completed_total / len(rows):.2%})"
        )
        print(f"Parziale:   {paths['partial_tokens']}")
        print("Rilancia lo stesso comando senza --stop-after per " "proseguire.")
        return 0

    finalize_cache(
        tokens_memmap,
        completed_memmap,
        paths,
        metadata,
    )
    print("[AUDIT]      Verifica completa della cache")
    audit = audit_final_cache(
        paths["final_tokens"],
        rows,
        chunk_rows=args.audit_chunk_rows,
        random_seed=args.seed,
    )
    audit["metadata_path"] = str(paths["metadata"].resolve())
    audit["signature_hash"] = signature_hash(signature)
    atomic_json(paths["audit"], audit)

    if audit["nonfinite_values"] != 0:
        raise RuntimeError("Audit: valori non finiti.")
    if audit["all_zero_rows"] != 0:
        raise RuntimeError("Audit: righe completamente nulle.")

    print("\n===== CACHE OPENCLIP COMPLETATA =====")
    print(f"Shape:       {audit['shape']}")
    print(f"Dtype:       {audit['dtype']}")
    print(f"Dimensione:  {human_bytes(audit['tokens_file_size_bytes'])}")
    print(
        f"Valori:      mean={audit['value_mean']:.6g} | "
        f"std={audit['value_std']:.6g} | "
        f"min={audit['value_min']:.6g} | "
        f"max={audit['value_max']:.6g}"
    )
    print(f"Token:       {paths['final_tokens']}")
    print(f"Metadata:    {paths['metadata']}")
    print(f"Audit:       {paths['audit']}")
    print("[COMPLETE]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
