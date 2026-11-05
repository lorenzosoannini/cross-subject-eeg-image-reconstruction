"""Stable identity checks for THINGS-EEG2 OpenCLIP caches."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence


IDENTITY_FIELDS = (
    "visual_index",
    "visual_uid",
    "partition",
    "event_id",
    "image_path",
)


def _normalized_path(value: str) -> str:
    """Normalize separators without tying an identity to a machine root."""
    return PurePosixPath(str(value).strip().replace("\\", "/")).as_posix()


def _canonical_identity_line(row: Mapping[str, object]) -> str:
    values = [
        str(int(row["visual_index"])),
        str(row["visual_uid"]).strip(),
        str(row["partition"]).strip(),
        str(int(row["event_id"])),
        _normalized_path(str(row["image_path"])),
    ]
    return "\t".join(values) + "\n"


def visual_identity_sha256_from_rows(
    rows: Sequence[Mapping[str, object]],
) -> str:
    """Hash ordered visual identities while deliberately excluding split."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(_canonical_identity_line(row).encode("utf-8"))
    return digest.hexdigest()


def load_visual_rows(path: str | Path) -> list[dict[str, str]]:
    resolved = Path(path)
    with resolved.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    missing = [field for field in IDENTITY_FIELDS if not rows or field not in rows[0]]
    if missing:
        raise ValueError(f"Campi identità mancanti in {resolved}: {missing}")

    indices = [int(row["visual_index"]) for row in rows]
    if indices != list(range(len(rows))):
        raise ValueError(f"visual_index non contiguo/ordinato in {resolved}.")
    return rows


def visual_identity_sha256(path: str | Path) -> str:
    return visual_identity_sha256_from_rows(load_visual_rows(path))


def _cache_path_matches_manifest(
    cache_path: str,
    manifest_path: str,
) -> bool:
    cached = _normalized_path(cache_path)
    manifest = _normalized_path(manifest_path).lstrip("./")
    return cached == manifest or cached.endswith("/" + manifest)


def validate_legacy_cache_rows(
    *,
    visuals_path: str | Path,
    cache_rows_path: str | Path,
) -> None:
    """
    Prove that a legacy cache still maps row-for-row to the same images.

    Legacy cache rows contain an absolute image path and a mutable split.
    The absolute root and split are ignored; every identity-bearing field is
    checked exactly, including the relative image-path suffix.
    """
    visuals = load_visual_rows(visuals_path)
    cache_path = Path(cache_rows_path)
    with cache_path.open(newline="", encoding="utf-8") as handle:
        cached = list(csv.DictReader(handle))

    if len(cached) != len(visuals):
        raise ValueError(
            "Numero righe cache/visuals diverso: " f"{len(cached)} != {len(visuals)}."
        )

    for expected_row, (visual, cache_row) in enumerate(zip(visuals, cached)):
        if int(cache_row.get("row", -1)) != expected_row:
            raise ValueError(f"Ordine cache non valido alla riga {expected_row}.")
        for field in (
            "visual_index",
            "visual_uid",
            "partition",
            "event_id",
        ):
            if str(cache_row.get(field, "")).strip() != str(visual[field]).strip():
                raise ValueError(
                    f"Identità cache diversa alla riga {expected_row}: "
                    f"campo {field}."
                )
        if not _cache_path_matches_manifest(
            cache_row.get("image_path", ""),
            visual["image_path"],
        ):
            raise ValueError(
                f"Percorso immagine cache diverso alla riga {expected_row}."
            )
