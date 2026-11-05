#!/usr/bin/env python3
"""
Costruisce l'indice canonico di THINGS-EEG2 per ZEBRA + REVE.

Questo è il primo blocco della nuova pipeline. Lo script NON materializza le
epoche EEG e NON calcola gli embedding CLIP. Produce invece:

    INDEX_DIR/
    ├── trials.npy
    ├── recordings.json
    ├── visuals.csv
    ├── visual_id_to_index.json
    ├── index_metadata.json
    ├── audit_report.json
    ├── metadata_overview.json
    ├── splits/
    │   ├── train_visual_indices.npy
    │   ├── val_visual_indices.npy
    │   ├── test_visual_indices.npy
    │   ├── train_visual_ids.txt
    │   ├── val_visual_ids.txt
    │   ├── test_visual_ids.txt
    │   └── split_metadata.json
    └── reve_positions/
        ├── things_eeg2_63_positions.npy
        └── things_eeg2_63_positions.json

Principi
--------
- Il manifest conserva tutte le occorrenze EEG reali.
- I catch trial 99999 non diventano esempi di apprendimento, ma restano
  rilevanti come "next event" per troncare clean_early.
- Gli ID sono namespaced:
      training:00001
      test:00001
  perché gli event ID delle due partizioni non identificano la stessa immagine.
- clean_early usa:
      [onset, min(onset + 40 campioni, onset del prossimo evento))
  a 200 Hz. Quindi produce normalmente 40 campioni validi, oppure 39 quando
  il prossimo onset cade a 195 ms.
- post400_right usa 80 campioni reali, salvo i margini della registrazione.
- La validation è definita una sola volta a livello di visual ID ed è quindi
  identica per tutti i soggetti e tutte le sessioni.
- La strategia di partizionamento è selezionabile da riga di comando.

Formato trials.npy
------------------
Structured array NumPy con campi:
    recording_id
    subject
    session
    partition_code       (0=training, 1=test)
    event_index
    event_id
    onset_sample
    next_event_sample
    visual_index
    occurrence_subject_visual
    occurrence_recording_visual
    clean_valid_length
    post400_valid_length
    canonical_test20

Il flag canonical_test20 seleziona deterministicamente al massimo 20
ripetizioni per (soggetto, sessione, immagine test), senza eliminare le altre.

Uso
---
1. Dry-run completo:
    python tools/build_things_eeg2_index.py \
      --root data/THINGS-EEG2

2. Scrittura effettiva:
    python tools/build_things_eeg2_index.py \
      --root data/THINGS-EEG2 \
      --execute

3. Senza interrogare REVE positions:
    python tools/build_things_eeg2_index.py \
      --root data/THINGS-EEG2 \
      --execute \
      --skip-reve-positions

Dipendenze
----------
    python -m pip install -U numpy transformers einops packaging

Nota di sicurezza
-----------------
image_metadata.npy è un oggetto NumPy pickled. Lo script usa allow_pickle=True
soltanto su quel file locale, che deve provenire dalla distribuzione ufficiale
o dal mirror già utilizzato per il download.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np


SCRIPT_VERSION = "1.0.0"
TARGET_SFREQ = 200.0
CATCH_ID = 99999

PARTITION_TO_CODE = {"training": 0, "test": 1}
CODE_TO_PARTITION = {value: key for key, value in PARTITION_TO_CODE.items()}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

TRIAL_DTYPE = np.dtype(
    [
        ("recording_id", "<u2"),
        ("subject", "u1"),
        ("session", "u1"),
        ("partition_code", "u1"),
        ("event_index", "<u4"),
        ("event_id", "<u4"),
        ("onset_sample", "<i8"),
        ("next_event_sample", "<i8"),
        ("visual_index", "<u4"),
        ("occurrence_subject_visual", "<u2"),
        ("occurrence_recording_visual", "<u2"),
        ("clean_valid_length", "<u2"),
        ("post400_valid_length", "<u2"),
        ("canonical_test20", "?"),
    ]
)


@dataclass(frozen=True)
class RecordingKey:
    subject: int
    session: int
    partition: str

    @property
    def short(self) -> str:
        return f"sub-{self.subject:02d}/" f"ses-{self.session:02d}/{self.partition}"


@dataclass
class RecordingDescriptor:
    recording_id: int
    key: RecordingKey
    eeg_path: Path
    events_path: Path
    metadata_path: Path
    eeg_shape: tuple[int, int]
    channel_names: list[str]
    n_events: int
    n_image_events: int
    n_catch_events: int
    min_event_id: int | None
    max_event_id: int | None
    trial_start: int = 0
    trial_stop: int = 0


@dataclass(frozen=True)
class DiscoveredImage:
    path: Path
    relative_path: str
    partition: str


@dataclass
class VisualRecord:
    visual_index: int
    visual_uid: str
    partition: str
    event_id: int
    concept_uid: str
    concept_name: str
    image_path: str
    image_filename: str
    split: str = ""


# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def json_ready(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return json_ready(dataclasses.asdict(value))
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
    return value


def atomic_write_json(path: Path, payload: Any) -> None:
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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def parse_number_spec(
    values: Sequence[str],
    *,
    minimum: int,
    maximum: int,
) -> list[int]:
    result: set[int] = set()

    for raw in values:
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue

            matched = re.fullmatch(r"(\d+)\s*-\s*(\d+)", token)
            if matched:
                start, stop = map(int, matched.groups())
                if start > stop:
                    raise ValueError(f"Intervallo non valido: {token}")
                result.update(range(start, stop + 1))
            elif token.isdigit():
                result.add(int(token))
            else:
                raise ValueError(f"Specifica numerica non valida: {token}")

    invalid = sorted(value for value in result if value < minimum or value > maximum)
    if invalid:
        raise ValueError(f"Valori fuori intervallo [{minimum}, {maximum}]: {invalid}")
    if not result:
        raise ValueError("La selezione non può essere vuota.")
    return sorted(result)


def natural_key(value: str) -> list[Any]:
    return [
        int(token) if token.isdigit() else token.casefold()
        for token in re.split(r"(\d+)", value)
    ]


def relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def normalize_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def visual_uid(partition: str, event_id: int) -> str:
    width = 5 if partition == "training" else 3
    return f"{partition}:{event_id:0{width}d}"


def stable_integer_seed(*values: int, base_seed: int) -> int:
    text = ":".join([str(base_seed), *(str(value) for value in values)])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def ensure_output_dir(
    output_dir: Path,
    *,
    execute: bool,
    overwrite: bool,
) -> None:
    if not execute:
        return

    important_outputs = [
        output_dir / "trials.npy",
        output_dir / "recordings.json",
        output_dir / "visuals.csv",
        output_dir / "index_metadata.json",
    ]
    existing = [path for path in important_outputs if path.exists()]

    if existing and not overwrite:
        raise FileExistsError(
            "L'indice sembra già esistere. Usa --overwrite per "
            f"rigenerarlo: {existing}"
        )

    if overwrite and output_dir.exists():
        backup = output_dir.with_name(output_dir.name + ".previous")
        if backup.exists():
            shutil.rmtree(backup)
        output_dir.replace(backup)
        print(f"[BACKUP] Vecchio indice spostato in: {backup}")

    output_dir.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Metadata immagini
# -----------------------------------------------------------------------------


def metadata_overview(value: Any, depth: int = 0) -> Any:
    if depth > 5:
        return {"type": type(value).__name__, "truncated": True}

    if isinstance(value, dict):
        return {
            "type": "dict",
            "keys": {
                normalize_string(key): metadata_overview(item, depth + 1)
                for key, item in value.items()
            },
        }

    if isinstance(value, np.ndarray):
        preview: list[Any] = []
        flattened = value.reshape(-1)
        for item in flattened[:5]:
            if np.isscalar(item) or isinstance(item, (str, bytes)):
                preview.append(normalize_string(item))
            else:
                preview.append(type(item).__name__)
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "preview": preview,
        }

    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "length": len(value),
            "preview": [
                (
                    normalize_string(item)
                    if np.isscalar(item) or isinstance(item, (str, bytes))
                    else type(item).__name__
                )
                for item in value[:5]
            ],
        }

    return {
        "type": type(value).__name__,
        "value": normalize_string(value)[:200],
    }


def load_image_metadata(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"image_metadata.npy non trovato: {path}")

    loaded = np.load(path, allow_pickle=True)

    if isinstance(loaded, np.ndarray) and loaded.shape == ():
        return loaded.item()

    if isinstance(loaded, np.ndarray) and loaded.dtype == object and loaded.size == 1:
        return loaded.reshape(()).item()

    return loaded


def flatten_metadata(
    value: Any,
    prefix: str = "",
    depth: int = 0,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    if depth > 5:
        return output

    if isinstance(value, dict):
        for key, item in value.items():
            key_string = normalize_string(key)
            path = f"{prefix}.{key_string}" if prefix else key_string
            output[path] = item
            output.update(flatten_metadata(item, path, depth + 1))

    return output


def scalar_sequence(value: Any) -> list[Any] | None:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return None
        flattened = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        array = np.asarray(value, dtype=object)
        flattened = array.reshape(-1).tolist()
    else:
        return None

    if not flattened:
        return []

    for item in flattened:
        if isinstance(item, (dict, list, tuple, np.ndarray)):
            return None
    return flattened


def string_sequence(value: Any) -> list[str] | None:
    sequence = scalar_sequence(value)
    if sequence is None:
        return None

    result: list[str] = []
    for item in sequence:
        if isinstance(item, (str, bytes, np.str_, np.bytes_)):
            result.append(normalize_string(item))
        else:
            return None
    return result


def integer_sequence(value: Any) -> list[int] | None:
    sequence = scalar_sequence(value)
    if sequence is None:
        return None

    result: list[int] = []
    for item in sequence:
        if isinstance(item, (bool, np.bool_)):
            return None
        if isinstance(item, (int, np.integer)):
            result.append(int(item))
        else:
            return None
    return result


def candidate_score(
    key_path: str,
    *,
    partition: str,
    kind: str,
    length: int,
    expected_images: int,
    expected_concepts: int | None,
) -> int:
    key = normalized_key(key_path)
    tokens = set(key.split("_"))

    partition_tokens = {"train", "training"} if partition == "training" else {"test"}
    file_tokens = {"file", "files", "filename", "filenames", "path", "paths"}
    image_tokens = {"img", "image", "images", "stim", "stimuli"}
    concept_tokens = {
        "concept",
        "concepts",
        "category",
        "categories",
        "class",
        "classes",
        "label",
        "labels",
    }

    score = 0
    if tokens & partition_tokens:
        score += 12
    else:
        score -= 8

    if kind == "files":
        if tokens & file_tokens:
            score += 10
        if tokens & image_tokens:
            score += 4
        if tokens & concept_tokens:
            score -= 4
        if length == expected_images:
            score += 15
    elif kind == "concepts":
        if tokens & concept_tokens:
            score += 10
        if tokens & image_tokens:
            score += 2
        if tokens & file_tokens:
            score -= 5
        if length == expected_images:
            score += 10
        if expected_concepts is not None and length == expected_concepts:
            score += 15
    else:
        raise ValueError(kind)

    exact_aliases = {
        ("training", "files"): {
            "train_img_files",
            "training_img_files",
            "train_image_files",
            "training_image_files",
            "train_files",
            "training_files",
        },
        ("test", "files"): {
            "test_img_files",
            "test_image_files",
            "test_files",
        },
        ("training", "concepts"): {
            "train_img_concepts",
            "training_img_concepts",
            "train_concepts",
            "training_concepts",
            "train_categories",
            "training_categories",
        },
        ("test", "concepts"): {
            "test_img_concepts",
            "test_concepts",
            "test_categories",
        },
    }
    leaf_key = normalized_key(key_path.split(".")[-1])
    if leaf_key in exact_aliases[(partition, kind)]:
        score += 30

    return score


def choose_metadata_sequence(
    flattened: dict[str, Any],
    *,
    partition: str,
    kind: str,
    expected_images: int,
    expected_concepts: int | None,
) -> tuple[list[str] | None, dict[str, Any]]:
    candidates: list[tuple[int, str, list[str]]] = []

    for key_path, value in flattened.items():
        sequence = string_sequence(value)
        if sequence is None:
            continue

        score = candidate_score(
            key_path,
            partition=partition,
            kind=kind,
            length=len(sequence),
            expected_images=expected_images,
            expected_concepts=expected_concepts,
        )
        candidates.append((score, key_path, sequence))

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

    report_candidates = [
        {
            "score": score,
            "key": key,
            "length": len(sequence),
            "preview": sequence[:3],
        }
        for score, key, sequence in candidates[:20]
    ]

    if not candidates or candidates[0][0] <= 0:
        return None, {
            "selected": None,
            "candidates": report_candidates,
        }

    score, key, sequence = candidates[0]
    return sequence, {
        "selected": {
            "score": score,
            "key": key,
            "length": len(sequence),
            "preview": sequence[:5],
        },
        "candidates": report_candidates,
    }


def detect_partition_from_path(path: Path, images_root: Path) -> str | None:
    try:
        relative = path.relative_to(images_root)
    except ValueError:
        relative = path

    tokens = [normalized_key(part) for part in relative.parts]

    for token in tokens:
        if "training" in token or token in {"train", "train_images"}:
            return "training"
        if token == "test" or "test_images" in token:
            return "test"

    return None


def discover_images(images_root: Path) -> list[DiscoveredImage]:
    if not images_root.is_dir():
        raise FileNotFoundError(
            f"Cartella immagini estratte non trovata: {images_root}"
        )

    discovered: list[DiscoveredImage] = []
    unknown: list[Path] = []

    for path in images_root.rglob("*"):
        if not path.is_file() or path.suffix.casefold() not in IMAGE_EXTENSIONS:
            continue

        partition = detect_partition_from_path(path, images_root)
        if partition is None:
            unknown.append(path)
            continue

        discovered.append(
            DiscoveredImage(
                path=path.resolve(),
                relative_path=path.resolve()
                .relative_to(images_root.resolve())
                .as_posix(),
                partition=partition,
            )
        )

    if unknown:
        preview = "\n".join(f"  - {path}" for path in unknown[:20])
        raise RuntimeError(
            "Sono state trovate immagini senza partizione training/test. "
            "La cartella estratta deve mantenere i nomi training_images e "
            f"test_images.\nEsempi:\n{preview}"
        )

    discovered.sort(key=lambda item: natural_key(item.relative_path))
    if not discovered:
        raise RuntimeError(f"Nessuna immagine trovata in {images_root}")
    return discovered


class ImageResolver:
    def __init__(
        self,
        images: Sequence[DiscoveredImage],
        images_root: Path,
    ) -> None:
        self.images_root = images_root.resolve()
        self.by_partition: dict[str, list[DiscoveredImage]] = defaultdict(list)
        self.by_relative: dict[tuple[str, str], DiscoveredImage] = {}
        self.by_basename: dict[
            tuple[str, str],
            list[DiscoveredImage],
        ] = defaultdict(list)
        self.by_stem: dict[
            tuple[str, str],
            list[DiscoveredImage],
        ] = defaultdict(list)

        for image in images:
            self.by_partition[image.partition].append(image)
            relative = image.relative_path.replace("\\", "/").lstrip("./")
            self.by_relative[(image.partition, relative.casefold())] = image
            self.by_basename[(image.partition, image.path.name.casefold())].append(
                image
            )
            self.by_stem[(image.partition, image.path.stem.casefold())].append(image)

        for partition in self.by_partition:
            self.by_partition[partition].sort(
                key=lambda item: natural_key(item.relative_path)
            )

    def resolve(
        self,
        partition: str,
        metadata_value: str,
    ) -> DiscoveredImage | None:
        raw = metadata_value.replace("\\", "/").strip().lstrip("./")
        candidates = [
            raw.casefold(),
            f"{partition}_images/{raw}".casefold(),
            f"{partition}/{raw}".casefold(),
        ]

        for candidate in candidates:
            exact = self.by_relative.get((partition, candidate))
            if exact is not None:
                return exact

        basename = Path(raw).name.casefold()
        basename_matches = self.by_basename.get((partition, basename), [])
        if len(basename_matches) == 1:
            return basename_matches[0]

        stem = Path(raw).stem.casefold()
        stem_matches = self.by_stem.get((partition, stem), [])
        if len(stem_matches) == 1:
            return stem_matches[0]

        suffix_matches = [
            image
            for image in self.by_partition[partition]
            if image.relative_path.casefold().endswith(raw.casefold())
        ]
        if len(suffix_matches) == 1:
            return suffix_matches[0]

        return None


def infer_concept_from_image(image: DiscoveredImage) -> tuple[str, str]:
    parts = Path(image.relative_path).parts

    candidates = list(reversed(parts[:-1])) + [image.path.stem]
    for candidate in candidates:
        cleaned = Path(candidate).stem
        matched = re.match(r"^(\d{1,6})[_-](.+)$", cleaned)
        if matched:
            numeric, name = matched.groups()
            concept_uid = f"{int(numeric):05d}"
            concept_name = name.replace("_", " ").strip()
            return concept_uid, concept_name

    parent = image.path.parent.name
    if parent and normalized_key(parent) not in {
        "training_images",
        "test_images",
        "train",
        "training",
        "test",
    }:
        return normalized_key(parent), parent.replace("_", " ")

    stem = image.path.stem
    return normalized_key(stem), stem.replace("_", " ")


def expand_concepts(
    concepts: list[str] | None,
    *,
    n_images: int,
    expected_concepts: int | None,
) -> list[str] | None:
    if concepts is None:
        return None

    if len(concepts) == n_images:
        return concepts

    if (
        expected_concepts is not None
        and len(concepts) == expected_concepts
        and n_images % expected_concepts == 0
    ):
        repetitions = n_images // expected_concepts
        return [concept for concept in concepts for _ in range(repetitions)]

    raise ValueError(
        "Lunghezza concept metadata incompatibile: "
        f"{len(concepts)} per {n_images} immagini."
    )


def build_visual_records(
    *,
    metadata: Any,
    metadata_path: Path,
    images_root: Path,
    expected_train_images: int,
    expected_test_images: int,
    expected_train_concepts: int,
    allow_sorted_fallback: bool,
    allow_unexpected_counts: bool,
) -> tuple[
    list[VisualRecord],
    dict[tuple[str, int], int],
    dict[str, Any],
]:
    discovered = discover_images(images_root)
    resolver = ImageResolver(discovered, images_root)

    discovered_counts = {
        partition: len(resolver.by_partition.get(partition, []))
        for partition in PARTITION_TO_CODE
    }

    expected_by_partition = {
        "training": expected_train_images,
        "test": expected_test_images,
    }
    expected_concepts_by_partition: dict[str, int | None] = {
        "training": expected_train_concepts,
        "test": expected_test_images,
    }

    if not allow_unexpected_counts:
        for partition, expected in expected_by_partition.items():
            observed = discovered_counts.get(partition, 0)
            if observed != expected:
                raise RuntimeError(
                    f"Numero immagini {partition} inatteso: "
                    f"{observed}, atteso {expected}."
                )

    flattened = flatten_metadata(metadata)
    parser_report: dict[str, Any] = {
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": sha256_file(metadata_path),
        "discovered_image_counts": discovered_counts,
        "partitions": {},
    }

    all_visuals: list[VisualRecord] = []
    event_to_visual: dict[tuple[str, int], int] = {}
    used_paths: set[str] = set()

    next_visual_index = 0

    for partition in ("training", "test"):
        expected_images = expected_by_partition[partition]
        expected_concepts = expected_concepts_by_partition[partition]

        file_values, file_report = choose_metadata_sequence(
            flattened,
            partition=partition,
            kind="files",
            expected_images=expected_images,
            expected_concepts=expected_concepts,
        )
        concept_values, concept_report = choose_metadata_sequence(
            flattened,
            partition=partition,
            kind="concepts",
            expected_images=expected_images,
            expected_concepts=expected_concepts,
        )

        parser_report["partitions"][partition] = {
            "files": file_report,
            "concepts": concept_report,
        }

        partition_images: list[DiscoveredImage]

        if file_values is not None:
            if len(file_values) != expected_images and not allow_unexpected_counts:
                raise RuntimeError(
                    f"{partition}: la sequenza file scelta contiene "
                    f"{len(file_values)} elementi, attesi {expected_images}."
                )

            unresolved: list[str] = []
            partition_images = []
            for value in file_values:
                resolved = resolver.resolve(partition, value)
                if resolved is None:
                    unresolved.append(value)
                else:
                    partition_images.append(resolved)

            if unresolved:
                preview = "\n".join(f"  - {value}" for value in unresolved[:30])
                raise RuntimeError(
                    f"{partition}: {len(unresolved)} file metadata non "
                    f"risolti sotto {images_root}.\nEsempi:\n{preview}"
                )
        else:
            if not allow_sorted_fallback:
                chosen = parser_report["partitions"][partition]["files"]
                raise RuntimeError(
                    f"Non è stato possibile identificare la lista ufficiale "
                    f"dei file {partition} in image_metadata.npy. "
                    "Esegui prima il dry-run e controlla i candidati "
                    "riportati nell'errore, oppure usa esplicitamente "
                    "--allow-sorted-image-fallback dopo aver verificato "
                    "l'ordinamento.\n"
                    f"Candidati: {chosen['candidates'][:5]}"
                )
            print(
                f"[WARN] {partition}: uso fallback con ordinamento naturale "
                "dei file immagine.",
                file=sys.stderr,
            )
            partition_images = list(resolver.by_partition[partition])

        if not partition_images:
            raise RuntimeError(f"Nessuna immagine risolta per {partition}.")

        concepts_expanded = expand_concepts(
            concept_values,
            n_images=len(partition_images),
            expected_concepts=expected_concepts,
        )

        if concepts_expanded is None:
            concept_pairs = [
                infer_concept_from_image(image) for image in partition_images
            ]
        else:
            concept_pairs = []
            concept_to_uid: dict[str, str] = {}
            for image, concept_raw in zip(
                partition_images,
                concepts_expanded,
                strict=True,
            ):
                concept_text = concept_raw.strip()
                if concept_text not in concept_to_uid:
                    inferred_uid, _ = infer_concept_from_image(image)
                    concept_to_uid[concept_text] = inferred_uid
                concept_pairs.append(
                    (
                        concept_to_uid[concept_text],
                        concept_text.replace("_", " "),
                    )
                )

        if len(set(image.relative_path for image in partition_images)) != len(
            partition_images
        ):
            raise RuntimeError(f"{partition}: la lista immagini contiene duplicati.")

        for offset, (image, concept_pair) in enumerate(
            zip(partition_images, concept_pairs, strict=True),
            start=1,
        ):
            uid = visual_uid(partition, offset)
            concept_uid, concept_name = concept_pair
            image_path = relative_or_absolute(image.path, images_root.parent)

            if image_path in used_paths:
                raise RuntimeError(f"Immagine usata più volte: {image_path}")
            used_paths.add(image_path)

            record = VisualRecord(
                visual_index=next_visual_index,
                visual_uid=uid,
                partition=partition,
                event_id=offset,
                concept_uid=concept_uid,
                concept_name=concept_name,
                image_path=image_path,
                image_filename=image.path.name,
            )
            all_visuals.append(record)
            event_to_visual[(partition, offset)] = next_visual_index
            next_visual_index += 1

        unique_concepts = len(
            {
                record.concept_uid
                for record in all_visuals
                if record.partition == partition
            }
        )
        parser_report["partitions"][partition]["resolved_image_count"] = len(
            partition_images
        )
        parser_report["partitions"][partition]["unique_concept_count"] = unique_concepts

        if (
            partition == "training"
            and not allow_unexpected_counts
            and unique_concepts != expected_train_concepts
        ):
            raise RuntimeError(
                "Numero concetti training inatteso dopo il parsing: "
                f"{unique_concepts}, atteso {expected_train_concepts}. "
                "Controlla la lista concepts o la struttura delle cartelle."
            )

    return all_visuals, event_to_visual, parser_report


# -----------------------------------------------------------------------------
# Registrazioni ed eventi
# -----------------------------------------------------------------------------


def canonical_paths(
    canonical_root: Path,
    key: RecordingKey,
) -> tuple[Path, Path, Path]:
    folder = canonical_root / f"sub-{key.subject:02d}" / f"ses-{key.session:02d}"
    stem = f"raw_eeg_{key.partition}"
    return (
        folder / f"{stem}_200hz.npy",
        folder / f"{stem}_events.npz",
        folder / f"{stem}_200hz.json",
    )


def read_event_arrays(
    events_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(events_path) as events:
        samples = np.asarray(events["sample_200hz"], dtype=np.int64)
        event_ids = np.asarray(events["event_id"], dtype=np.int64)
    return samples, event_ids


def inspect_recording(
    canonical_root: Path,
    key: RecordingKey,
    recording_id: int,
) -> RecordingDescriptor:
    eeg_path, events_path, metadata_path = canonical_paths(
        canonical_root,
        key,
    )
    missing = [
        path for path in (eeg_path, events_path, metadata_path) if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"{key.short}: file mancanti: {missing}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise ValueError(f"{key.short}: status non complete.")

    sfreq = float(metadata["target_sfreq"])
    if not math.isclose(
        sfreq,
        TARGET_SFREQ,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(f"{key.short}: target_sfreq={sfreq}, atteso {TARGET_SFREQ}.")

    eeg = np.load(eeg_path, mmap_mode="r")
    if eeg.ndim != 2 or eeg.shape[0] != 63:
        raise ValueError(f"{key.short}: shape EEG inattesa {eeg.shape}, attesa [63,T].")
    if eeg.dtype != np.float32:
        raise ValueError(f"{key.short}: dtype {eeg.dtype}, atteso float32.")

    channel_names = [normalize_string(name) for name in metadata["eeg_channel_names"]]
    if len(channel_names) != 63:
        raise ValueError(f"{key.short}: attesi 63 channel_names.")
    if len(set(channel_names)) != 63:
        raise ValueError(f"{key.short}: channel_names duplicati.")

    samples, event_ids = read_event_arrays(events_path)
    if samples.ndim != 1 or event_ids.ndim != 1:
        raise ValueError(f"{key.short}: eventi non 1D.")
    if len(samples) != len(event_ids):
        raise ValueError(f"{key.short}: lunghezze eventi incoerenti.")
    if len(samples) != int(metadata["n_events"]):
        raise ValueError(f"{key.short}: n_events incoerente.")
    if len(samples) == 0:
        raise ValueError(f"{key.short}: nessun evento.")
    if not np.all(np.diff(samples) > 0):
        raise ValueError(f"{key.short}: sample eventi non strettamente crescente.")
    if samples[0] < 0 or samples[-1] >= eeg.shape[1]:
        raise ValueError(f"{key.short}: evento fuori dal segnale.")

    image_mask = (event_ids > 0) & (event_ids != CATCH_ID)
    image_ids = event_ids[image_mask]

    return RecordingDescriptor(
        recording_id=recording_id,
        key=key,
        eeg_path=eeg_path.resolve(),
        events_path=events_path.resolve(),
        metadata_path=metadata_path.resolve(),
        eeg_shape=(int(eeg.shape[0]), int(eeg.shape[1])),
        channel_names=channel_names,
        n_events=len(event_ids),
        n_image_events=int(image_mask.sum()),
        n_catch_events=int(np.sum(event_ids == CATCH_ID)),
        min_event_id=int(image_ids.min()) if len(image_ids) else None,
        max_event_id=int(image_ids.max()) if len(image_ids) else None,
    )


def inspect_recordings(
    canonical_root: Path,
    subjects: Sequence[int],
    sessions: Sequence[int],
) -> list[RecordingDescriptor]:
    descriptors: list[RecordingDescriptor] = []
    recording_id = 0

    for subject in subjects:
        for session in sessions:
            for partition in ("training", "test"):
                key = RecordingKey(subject, session, partition)
                descriptor = inspect_recording(
                    canonical_root,
                    key,
                    recording_id,
                )
                descriptors.append(descriptor)
                recording_id += 1

                print(
                    f"[RECORDING {recording_id:02d}] {key.short} | "
                    f"shape={descriptor.eeg_shape} | "
                    f"events={descriptor.n_events} | "
                    f"images={descriptor.n_image_events} | "
                    f"catch={descriptor.n_catch_events}"
                )

    reference_channels = descriptors[0].channel_names
    for descriptor in descriptors[1:]:
        if descriptor.channel_names != reference_channels:
            raise RuntimeError(f"Ordine canali differente in {descriptor.key.short}.")

    return descriptors


def fill_trial_manifest(
    descriptors: Sequence[RecordingDescriptor],
    event_to_visual: dict[tuple[str, int], int],
    *,
    canonical_test_repetitions: int,
    canonical_test_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    total_trials = sum(descriptor.n_image_events for descriptor in descriptors)
    trials = np.empty(total_trials, dtype=TRIAL_DTYPE)

    subject_occurrence: dict[tuple[int, int, int], int] = defaultdict(int)
    clean_lengths = Counter()
    post400_lengths = Counter()
    partition_counts = Counter()
    subject_counts = Counter()
    session_counts = Counter()
    event_id_counts: dict[str, Counter[int]] = {
        "training": Counter(),
        "test": Counter(),
    }

    cursor = 0

    for descriptor in descriptors:
        samples, event_ids = read_event_arrays(descriptor.events_path)
        recording_start = cursor
        recording_occurrence: dict[int, int] = defaultdict(int)
        partition_code = PARTITION_TO_CODE[descriptor.key.partition]
        recording_length = descriptor.eeg_shape[1]

        for event_index, (onset, event_id_raw) in enumerate(
            zip(samples, event_ids, strict=True)
        ):
            event_id = int(event_id_raw)
            if event_id <= 0 or event_id == CATCH_ID:
                continue

            visual_index = event_to_visual.get((descriptor.key.partition, event_id))
            if visual_index is None:
                raise KeyError(
                    f"{descriptor.key.short}: event_id {event_id} non "
                    "presente nell'indice immagini."
                )

            next_event_sample = (
                int(samples[event_index + 1])
                if event_index + 1 < len(samples)
                else recording_length
            )

            clean_stop = min(
                int(onset) + 40,
                next_event_sample,
                recording_length,
            )
            clean_valid_length = max(0, clean_stop - int(onset))

            post400_stop = min(int(onset) + 80, recording_length)
            post400_valid_length = max(
                0,
                post400_stop - int(onset),
            )

            if clean_valid_length <= 0:
                raise RuntimeError(
                    f"{descriptor.key.short}: clean_valid_length={clean_valid_length} "
                    f"all'evento {event_index}."
                )
            if post400_valid_length <= 0:
                raise RuntimeError(
                    f"{descriptor.key.short}: post400_valid_length={post400_valid_length} "
                    f"all'evento {event_index}."
                )

            subject_key = (
                descriptor.key.subject,
                partition_code,
                event_id,
            )
            subject_index = subject_occurrence[subject_key]
            recording_index = recording_occurrence[event_id]

            trials[cursor] = (
                descriptor.recording_id,
                descriptor.key.subject,
                descriptor.key.session,
                partition_code,
                event_index,
                event_id,
                int(onset),
                next_event_sample,
                visual_index,
                subject_index,
                recording_index,
                clean_valid_length,
                post400_valid_length,
                False,
            )

            subject_occurrence[subject_key] += 1
            recording_occurrence[event_id] += 1
            clean_lengths[clean_valid_length] += 1
            post400_lengths[post400_valid_length] += 1
            partition_counts[descriptor.key.partition] += 1
            subject_counts[descriptor.key.subject] += 1
            session_counts[descriptor.key.session] += 1
            event_id_counts[descriptor.key.partition][event_id] += 1
            cursor += 1

        descriptor.trial_start = recording_start
        descriptor.trial_stop = cursor

        # Selezione deterministica delle 20 ripetizioni canoniche nel test.
        if descriptor.key.partition == "test":
            recording_rows = np.arange(
                recording_start,
                cursor,
                dtype=np.int64,
            )
            recording_event_ids = trials["event_id"][recording_rows]

            for event_id in np.unique(recording_event_ids):
                rows = recording_rows[recording_event_ids == event_id]
                n_keep = min(canonical_test_repetitions, len(rows))
                rng = np.random.default_rng(
                    stable_integer_seed(
                        descriptor.key.subject,
                        descriptor.key.session,
                        int(event_id),
                        base_seed=canonical_test_seed,
                    )
                )
                chosen = np.sort(rng.choice(rows, size=n_keep, replace=False))
                trials["canonical_test20"][chosen] = True

    if cursor != total_trials:
        raise RuntimeError(f"Riempiti {cursor}/{total_trials} trial.")

    audit = {
        "total_trials": total_trials,
        "partition_trial_counts": dict(partition_counts),
        "subject_trial_counts": {
            str(key): value for key, value in sorted(subject_counts.items())
        },
        "session_trial_counts": {
            str(key): value for key, value in sorted(session_counts.items())
        },
        "clean_valid_length_distribution": {
            str(key): value for key, value in sorted(clean_lengths.items())
        },
        "post400_valid_length_distribution": {
            str(key): value for key, value in sorted(post400_lengths.items())
        },
        "event_id_coverage": {
            partition: {
                "n_unique": len(counter),
                "min": min(counter) if counter else None,
                "max": max(counter) if counter else None,
                "min_occurrences": min(counter.values()) if counter else None,
                "max_occurrences": max(counter.values()) if counter else None,
            }
            for partition, counter in event_id_counts.items()
        },
        "canonical_test20_count": int(trials["canonical_test20"].sum()),
    }
    return trials, audit


# -----------------------------------------------------------------------------
# Split visuali
# -----------------------------------------------------------------------------


def assign_visual_splits(
    visuals: list[VisualRecord],
    *,
    val_ratio: float,
    val_mode: str,
    split_seed: int,
) -> dict[str, np.ndarray]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio deve essere fra 0 e 1.")

    training_visuals = [record for record in visuals if record.partition == "training"]
    test_visuals = [record for record in visuals if record.partition == "test"]

    by_concept: dict[str, list[VisualRecord]] = defaultdict(list)
    for record in training_visuals:
        by_concept[record.concept_uid].append(record)

    rng = np.random.default_rng(split_seed)
    train_indices: list[int] = []
    val_indices: list[int] = []

    if val_mode == "concept_holdout":
        concepts = np.asarray(sorted(by_concept), dtype=object)
        n_val_concepts = max(1, int(round(len(concepts) * val_ratio)))
        permutation = rng.permutation(len(concepts))
        val_concepts = set(concepts[permutation[:n_val_concepts]].tolist())

        for concept_uid, records in by_concept.items():
            destination = val_indices if concept_uid in val_concepts else train_indices
            destination.extend(record.visual_index for record in records)

    elif val_mode == "image_within_concept":
        for concept_uid in sorted(by_concept):
            records = sorted(
                by_concept[concept_uid],
                key=lambda item: item.event_id,
            )
            if len(records) < 2:
                train_indices.extend(record.visual_index for record in records)
                continue

            local = np.arange(len(records))
            local_rng = np.random.default_rng(
                stable_integer_seed(
                    int(hashlib.sha256(concept_uid.encode()).hexdigest()[:8], 16),
                    base_seed=split_seed,
                )
            )
            local_rng.shuffle(local)

            n_val = max(1, int(round(len(records) * val_ratio)))
            n_val = min(n_val, len(records) - 1)
            val_local = set(local[:n_val].tolist())

            for index, record in enumerate(records):
                if index in val_local:
                    val_indices.append(record.visual_index)
                else:
                    train_indices.append(record.visual_index)
    else:
        raise ValueError(f"val_mode non valido: {val_mode}")

    test_indices = [record.visual_index for record in test_visuals]

    arrays = {
        "train": np.asarray(sorted(train_indices), dtype=np.int32),
        "val": np.asarray(sorted(val_indices), dtype=np.int32),
        "test": np.asarray(sorted(test_indices), dtype=np.int32),
    }

    train_set = set(arrays["train"].tolist())
    val_set = set(arrays["val"].tolist())
    test_set = set(arrays["test"].tolist())

    if train_set & val_set:
        raise RuntimeError("Leakage visual ID fra train e validation.")
    if train_set & test_set or val_set & test_set:
        raise RuntimeError("Leakage visual ID fra training partition e test.")

    split_lookup = {}
    for split, array in arrays.items():
        for visual_index in array:
            split_lookup[int(visual_index)] = split

    for record in visuals:
        record.split = split_lookup[record.visual_index]

    return arrays


def split_report(
    visuals: Sequence[VisualRecord],
    splits: dict[str, np.ndarray],
    *,
    val_mode: str,
    val_ratio: float,
    split_seed: int,
) -> dict[str, Any]:
    by_index = {record.visual_index: record for record in visuals}

    report: dict[str, Any] = {
        "val_mode": val_mode,
        "val_ratio_requested": val_ratio,
        "split_seed": split_seed,
        "splits": {},
    }

    concept_sets: dict[str, set[str]] = {}
    for split, indices in splits.items():
        records = [by_index[int(index)] for index in indices]
        concepts = {record.concept_uid for record in records}
        concept_sets[split] = concepts

        report["splits"][split] = {
            "n_visuals": len(records),
            "n_concepts": len(concepts),
            "first_visual_ids": [record.visual_uid for record in records[:10]],
        }

    report["concept_overlap"] = {
        "train_val": len(concept_sets["train"] & concept_sets["val"]),
        "train_test": len(concept_sets["train"] & concept_sets["test"]),
        "val_test": len(concept_sets["val"] & concept_sets["test"]),
    }
    return report


# -----------------------------------------------------------------------------
# REVE positions
# -----------------------------------------------------------------------------


def export_reve_positions(
    output_dir: Path,
    channel_names: Sequence[str],
    *,
    positions_model_id: str,
    local_files_only: bool,
) -> dict[str, Any]:
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "Manca transformers. Installa con:\n"
            "  python -m pip install -U transformers einops packaging"
        ) from exc

    print(f"[REVE] Caricamento position bank: {positions_model_id}")
    position_bank = AutoModel.from_pretrained(
        positions_model_id,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )

    positions = position_bank(list(channel_names))
    if not isinstance(positions, np.ndarray):
        try:
            import torch

            if torch.is_tensor(positions):
                positions = positions.detach().cpu().numpy()
            else:
                positions = np.asarray(positions)
        except ImportError:
            positions = np.asarray(positions)

    positions = np.asarray(positions, dtype=np.float32)
    if positions.shape != (len(channel_names), 3):
        raise RuntimeError(f"Shape REVE positions inattesa: {positions.shape}")
    if not np.isfinite(positions).all():
        raise RuntimeError("REVE positions contiene NaN/Inf.")
    if np.unique(positions, axis=0).shape[0] != len(channel_names):
        raise RuntimeError("Coordinate REVE duplicate.")

    position_dir = output_dir / "reve_positions"
    position_dir.mkdir(parents=True, exist_ok=True)
    npy_path = position_dir / "things_eeg2_63_positions.npy"
    json_path = position_dir / "things_eeg2_63_positions.json"

    atomic_save_npy(npy_path, positions)
    payload = {
        "created_at_utc": utc_now(),
        "positions_model_id": positions_model_id,
        "model_class": type(position_bank).__name__,
        "commit_hash": getattr(
            getattr(position_bank, "config", None),
            "_commit_hash",
            None,
        ),
        "channel_names": list(channel_names),
        "shape": list(positions.shape),
        "dtype": str(positions.dtype),
        "unique_positions": int(np.unique(positions, axis=0).shape[0]),
        "positions_sha256": sha256_file(npy_path),
    }
    atomic_write_json(json_path, payload)
    return payload


# -----------------------------------------------------------------------------
# Scrittura
# -----------------------------------------------------------------------------


def write_visuals_csv(
    path: Path,
    visuals: Sequence[VisualRecord],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")

    fields = [
        "visual_index",
        "visual_uid",
        "partition",
        "event_id",
        "concept_uid",
        "concept_name",
        "image_path",
        "image_filename",
        "split",
    ]
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in visuals:
            writer.writerow(dataclasses.asdict(record))

    temporary.replace(path)


def write_text_lines(path: Path, values: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        "\n".join(values) + ("\n" if values else ""),
        encoding="utf-8",
    )
    temporary.replace(path)


def recording_payload(
    descriptor: RecordingDescriptor,
    root: Path,
) -> dict[str, Any]:
    return {
        "recording_id": descriptor.recording_id,
        "subject": descriptor.key.subject,
        "session": descriptor.key.session,
        "partition": descriptor.key.partition,
        "eeg_path": relative_or_absolute(descriptor.eeg_path, root),
        "events_path": relative_or_absolute(
            descriptor.events_path,
            root,
        ),
        "metadata_path": relative_or_absolute(
            descriptor.metadata_path,
            root,
        ),
        "eeg_shape": list(descriptor.eeg_shape),
        "channel_names": descriptor.channel_names,
        "n_events": descriptor.n_events,
        "n_image_events": descriptor.n_image_events,
        "n_catch_events": descriptor.n_catch_events,
        "min_event_id": descriptor.min_event_id,
        "max_event_id": descriptor.max_event_id,
        "trial_start": descriptor.trial_start,
        "trial_stop": descriptor.trial_stop,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Costruisce manifest, split e coordinate REVE per "
            "THINGS-EEG2 canonico a 200 Hz."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Radice data/THINGS-EEG2.",
    )
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=None,
        help="Default: ROOT/canonical_200hz.",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=None,
        help="Default: ROOT/imgs.",
    )
    parser.add_argument(
        "--image-metadata",
        type=Path,
        default=None,
        help="Default: ROOT/image_metadata.npy.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: ROOT/zebra_reve_index.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=["1-10"],
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        default=["1-4"],
    )
    parser.add_argument(
        "--val-mode",
        choices=["concept_holdout", "image_within_concept"],
        default="concept_holdout",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--canonical-test-repetitions",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--canonical-test-seed",
        type=int,
        default=20200220,
    )
    parser.add_argument(
        "--expected-train-images",
        type=int,
        default=16540,
    )
    parser.add_argument(
        "--expected-test-images",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--expected-train-concepts",
        type=int,
        default=1654,
    )
    parser.add_argument(
        "--allow-unexpected-counts",
        action="store_true",
    )
    parser.add_argument(
        "--allow-sorted-image-fallback",
        action="store_true",
        help=(
            "Usa l'ordinamento naturale delle immagini soltanto quando "
            "image_metadata.npy non contiene una lista file identificabile."
        ),
    )
    parser.add_argument(
        "--positions-model",
        default="brain-bzh/reve-positions",
    )
    parser.add_argument(
        "--skip-reve-positions",
        action="store_true",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Scrive realmente l'indice. Senza è un dry-run completo.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        subjects = parse_number_spec(
            args.subjects,
            minimum=1,
            maximum=10,
        )
        sessions = parse_number_spec(
            args.sessions,
            minimum=1,
            maximum=4,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.canonical_test_repetitions < 1:
        raise SystemExit("--canonical-test-repetitions deve essere >= 1.")

    root = args.root.expanduser().resolve()
    canonical_root = (
        args.canonical_root.expanduser().resolve()
        if args.canonical_root is not None
        else root / "canonical_200hz"
    )
    images_root = (
        args.images_root.expanduser().resolve()
        if args.images_root is not None
        else root / "imgs"
    )
    metadata_path = (
        args.image_metadata.expanduser().resolve()
        if args.image_metadata is not None
        else root / "image_metadata.npy"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / "zebra_reve_index"
    )

    print(f"[MODE]       {'EXECUTE' if args.execute else 'DRY-RUN'}")
    print(f"[ROOT]       {root}")
    print(f"[CANONICAL]  {canonical_root}")
    print(f"[IMAGES]     {images_root}")
    print(f"[METADATA]   {metadata_path}")
    print(f"[OUTPUT]     {output_dir}")
    print(f"[SUBJECTS]   {subjects}")
    print(f"[SESSIONS]   {sessions}")
    print(
        f"[VAL]        {args.val_mode}, ratio={args.val_ratio}, "
        f"seed={args.split_seed}"
    )

    if not root.is_dir():
        raise SystemExit(f"Root non trovata: {root}")
    if not canonical_root.is_dir():
        raise SystemExit(f"Canonical root non trovata: {canonical_root}")

    ensure_output_dir(
        output_dir,
        execute=args.execute,
        overwrite=args.overwrite,
    )

    print("\n[1/5] Lettura image_metadata.npy")
    metadata = load_image_metadata(metadata_path)
    overview = metadata_overview(metadata)

    print("[2/5] Costruzione indice immagini")
    visuals, event_to_visual, parser_report = build_visual_records(
        metadata=metadata,
        metadata_path=metadata_path,
        images_root=images_root,
        expected_train_images=args.expected_train_images,
        expected_test_images=args.expected_test_images,
        expected_train_concepts=args.expected_train_concepts,
        allow_sorted_fallback=args.allow_sorted_image_fallback,
        allow_unexpected_counts=args.allow_unexpected_counts,
    )
    print(
        f"[VISUALS] training="
        f"{sum(v.partition == 'training' for v in visuals)} | "
        f"test={sum(v.partition == 'test' for v in visuals)}"
    )

    print("[3/5] Audit delle registrazioni canoniche")
    descriptors = inspect_recordings(
        canonical_root,
        subjects,
        sessions,
    )

    print("[4/5] Costruzione trial manifest")
    trials, trial_audit = fill_trial_manifest(
        descriptors,
        event_to_visual,
        canonical_test_repetitions=args.canonical_test_repetitions,
        canonical_test_seed=args.canonical_test_seed,
    )

    splits = assign_visual_splits(
        visuals,
        val_ratio=args.val_ratio,
        val_mode=args.val_mode,
        split_seed=args.split_seed,
    )
    split_metadata = split_report(
        visuals,
        splits,
        val_mode=args.val_mode,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
    )

    print(
        "[SPLITS] "
        f"train={len(splits['train'])} | "
        f"val={len(splits['val'])} | "
        f"test={len(splits['test'])}"
    )
    print("[CLEAN LENGTHS] " f"{trial_audit['clean_valid_length_distribution']}")

    channel_names = descriptors[0].channel_names

    audit_report = {
        "created_at_utc": utc_now(),
        "script_version": SCRIPT_VERSION,
        "root": str(root),
        "canonical_root": str(canonical_root),
        "images_root": str(images_root),
        "subjects": subjects,
        "sessions": sessions,
        "recording_count": len(descriptors),
        "visual_count": len(visuals),
        "channel_count": len(channel_names),
        "channel_names": channel_names,
        "image_parser": parser_report,
        "trials": trial_audit,
        "splits": split_metadata,
    }

    if not args.execute:
        print("\n[DRY-RUN COMPLETE]")
        print("Nessun file scritto. Per creare l'indice aggiungi --execute.")
        print(
            "Il parser metadata selezionato è:\n"
            + json.dumps(
                {
                    partition: {
                        "files": data["files"]["selected"],
                        "concepts": data["concepts"]["selected"],
                        "resolved_image_count": data["resolved_image_count"],
                        "unique_concept_count": data["unique_concept_count"],
                    }
                    for partition, data in parser_report["partitions"].items()
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    print("[5/5] Scrittura indice")

    atomic_save_npy(output_dir / "trials.npy", trials)
    atomic_write_json(
        output_dir / "recordings.json",
        [recording_payload(descriptor, root) for descriptor in descriptors],
    )
    write_visuals_csv(output_dir / "visuals.csv", visuals)
    atomic_write_json(
        output_dir / "visual_id_to_index.json",
        {record.visual_uid: record.visual_index for record in visuals},
    )
    atomic_write_json(
        output_dir / "metadata_overview.json",
        overview,
    )

    split_dir = output_dir / "splits"
    by_index = {record.visual_index: record for record in visuals}

    for split, indices in splits.items():
        atomic_save_npy(
            split_dir / f"{split}_visual_indices.npy",
            indices,
        )
        write_text_lines(
            split_dir / f"{split}_visual_ids.txt",
            [by_index[int(index)].visual_uid for index in indices],
        )

    atomic_write_json(
        split_dir / "split_metadata.json",
        split_metadata,
    )

    positions_report = None
    if not args.skip_reve_positions:
        positions_report = export_reve_positions(
            output_dir,
            channel_names,
            positions_model_id=args.positions_model,
            local_files_only=args.local_files_only,
        )

    audit_report["reve_positions"] = positions_report
    atomic_write_json(
        output_dir / "audit_report.json",
        audit_report,
    )

    index_metadata = {
        "created_at_utc": utc_now(),
        "script_version": SCRIPT_VERSION,
        "trial_dtype": trials.dtype.descr,
        "trial_count": len(trials),
        "recording_count": len(descriptors),
        "visual_count": len(visuals),
        "partition_codes": PARTITION_TO_CODE,
        "catch_id": CATCH_ID,
        "target_sfreq": TARGET_SFREQ,
        "window_modes": {
            "clean_early": {
                "nominal_start_ms": 0,
                "nominal_end_ms": 200,
                "nominal_samples": 40,
                "stop_rule": "min(onset+40,next_event,recording_end)",
                "padding": "right_to_200",
            },
            "post400_right": {
                "start_ms": 0,
                "end_ms": 400,
                "nominal_samples": 80,
                "stop_rule": "min(onset+80,recording_end)",
                "padding": "right_to_200",
            },
        },
        "canonical_test_repetitions": (args.canonical_test_repetitions),
        "canonical_test_seed": args.canonical_test_seed,
        "split_metadata_path": "splits/split_metadata.json",
        "trials_sha256": sha256_file(output_dir / "trials.npy"),
        "recordings_sha256": sha256_file(output_dir / "recordings.json"),
        "visuals_sha256": sha256_file(output_dir / "visuals.csv"),
    }
    atomic_write_json(
        output_dir / "index_metadata.json",
        index_metadata,
    )

    print("\n===== INDICE COMPLETATO =====")
    print(f"Trial:          {len(trials)}")
    print(f"Registrazioni:  {len(descriptors)}")
    print(f"Visual ID:      {len(visuals)}")
    print("Clean lengths:  " f"{trial_audit['clean_valid_length_distribution']}")
    print(f"Train visuals:  {len(splits['train'])}")
    print(f"Val visuals:    {len(splits['val'])}")
    print(f"Test visuals:   {len(splits['test'])}")
    print(f"Output:         {output_dir}")
    print("[COMPLETE]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
