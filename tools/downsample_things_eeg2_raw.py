"""
Downsampling conservativo di THINGS-EEG2 da 1000 Hz a 200 Hz.

Input atteso:
  ROOT/raw-eeg/sub-XX/ses-YY/raw_eeg_training.npy
  ROOT/raw-eeg/sub-XX/ses-YY/raw_eeg_test.npy

Output:
  ROOT/canonical_200hz/sub-XX/ses-YY/
    raw_eeg_training_200hz.npy
    raw_eeg_training_events.npz
    raw_eeg_training_200hz.json
    raw_eeg_test_200hz.npy
    raw_eeg_test_events.npz
    raw_eeg_test_200hz.json

Lo script:
  - conserva tutti i 63 canali EEG e il loro ordine;
  - estrae il canale stim prima del resampling;
  - applica resample_poly con filtro anti-alias;
  - salva l'EEG continuo in float32 e formato .npy memmappabile;
  - conserva tutti gli eventi, incluso l'eventuale catch trial 99999;
  - non applica epoching, crop, padding, baseline, MVNN o normalizzazione;
  - processa un solo file e un solo canale alla volta;
  - non cancella mai il raw senza due flag espliciti.

Requisiti:
  python -m pip install -U numpy scipy

Esempi:
  # Dry run del test della prima sessione
  python tools/downsample_things_eeg2_raw.py     --root data/THINGS-EEG2     --subjects 1 --sessions 1 --partitions test

  # Downsampling pilota
  python tools/downsample_things_eeg2_raw.py     --root data/THINGS-EEG2     --subjects 1 --sessions 1 --partitions test     --execute

  # Tutto il dataset
  python tools/downsample_things_eeg2_raw.py     --root data/THINGS-EEG2     --execute
"""

from __future__ import annotations
import argparse
import datetime as dt
import gc
import json
import math
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import scipy
from scipy.signal import resample_poly

SOURCE_SFREQ = 1000.0
DEFAULT_TARGET_SFREQ = 200.0
REQUIRED_KEYS = {"raw_eeg_data", "ch_names", "ch_types", "sfreq"}


@dataclass(frozen=True)
class WorkItem:
    subject: int
    session: int
    partition: str
    source_path: Path


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TiB"


def parse_numbers(values: Sequence[str], *, minimum: int, maximum: int) -> list[int]:
    result: set[int] = set()
    for raw in values:
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            match = re.fullmatch("(\\d+)\\s*-\\s*(\\d+)", token)
            if match:
                (start, end) = map(int, match.groups())
                if start > end:
                    raise ValueError(f"Intervallo non valido: {token}")
                result.update(range(start, end + 1))
            elif token.isdigit():
                result.add(int(token))
            else:
                raise ValueError(f"Specifica non valida: {token}")
    invalid = sorted((x for x in result if x < minimum or x > maximum))
    if invalid:
        raise ValueError(f"Valori fuori intervallo [{minimum}, {maximum}]: {invalid}")
    if not result:
        raise ValueError("Selezione vuota.")
    return sorted(result)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def output_paths(
    output_root: Path, *, subject: int, session: int, partition: str
) -> tuple[Path, Path, Path]:
    folder = output_root / f"sub-{subject:02d}" / f"ses-{session:02d}"
    stem = f"raw_eeg_{partition}"
    return (
        folder / f"{stem}_200hz.npy",
        folder / f"{stem}_events.npz",
        folder / f"{stem}_200hz.json",
    )


def discover(
    raw_root: Path,
    subjects: Sequence[int],
    sessions: Sequence[int],
    partitions: Sequence[str],
) -> list[WorkItem]:
    items: list[WorkItem] = []
    missing: list[Path] = []
    for subject in subjects:
        for session in sessions:
            folder = raw_root / f"sub-{subject:02d}" / f"ses-{session:02d}"
            for partition in partitions:
                source = folder / f"raw_eeg_{partition}.npy"
                if source.is_file():
                    items.append(WorkItem(subject, session, partition, source))
                else:
                    missing.append(source)
    if missing:
        preview = "\n".join((f"  - {p}" for p in missing[:20]))
        more = f"\n  ... e altri {len(missing) - 20}" if len(missing) > 20 else ""
        raise FileNotFoundError("File raw mancanti:\n" + preview + more)
    return items


def load_dict(path: Path) -> dict[str, Any]:
    try:
        data = np.load(path, allow_pickle=True).item()
    except Exception as exc:
        raise RuntimeError(f"Errore leggendo {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise TypeError(f"{path} non contiene un dizionario NumPy.")
    missing = REQUIRED_KEYS - set(data)
    if missing:
        raise KeyError(f"Chiavi mancanti in {path}: {sorted(missing)}")
    return data


def validate_layout(
    data: dict[str, Any],
) -> tuple[np.ndarray, list[str], list[str], list[int], int]:
    matrix = np.asarray(data["raw_eeg_data"])
    names = [str(x) for x in data["ch_names"]]
    types = [str(x).lower() for x in data["ch_types"]]
    if matrix.ndim != 2:
        raise ValueError(f"Shape non 2D: {matrix.shape}")
    if matrix.shape[0] != len(names) or len(names) != len(types):
        raise ValueError(
            f"Incoerenza fra righe, ch_names e ch_types: {matrix.shape[0]}, {len(names)}, {len(types)}"
        )
    stim_candidates = [
        i
        for (i, (name, kind)) in enumerate(zip(names, types))
        if name.lower() == "stim" or kind == "stim"
    ]
    if len(stim_candidates) != 1:
        raise ValueError(f"Atteso un solo canale stim, trovati {stim_candidates}")
    stim_index = stim_candidates[0]
    eeg_indices = [i for i in range(len(names)) if i != stim_index]
    unexpected = [(i, names[i], types[i]) for i in eeg_indices if types[i] != "eeg"]
    if unexpected:
        raise ValueError(f"Canali non EEG inattesi: {unexpected}")
    return (matrix, names, types, eeg_indices, stim_index)


def extract_events(
    stim: np.ndarray, *, source_sfreq: float, target_sfreq: float, output_n_times: int
) -> dict[str, np.ndarray]:
    if stim.ndim != 1:
        raise ValueError(f"Stim non 1D: {stim.shape}")
    if not np.isfinite(stim).all():
        raise ValueError("Il canale stim contiene NaN/Inf.")
    sample_source = np.flatnonzero(stim != 0).astype(np.int64)
    values = stim[sample_source]
    rounded = np.rint(values)
    if not np.allclose(values, rounded, rtol=0.0, atol=1e-06):
        raise ValueError("Il canale stim contiene valori non interi.")
    event_id = rounded.astype(np.int64)
    time_seconds = sample_source.astype(np.float64) / source_sfreq
    target_float = time_seconds * target_sfreq
    sample_target = np.floor(target_float + 0.5).astype(np.int64)
    sample_target = np.clip(sample_target, 0, output_n_times - 1)
    if len(np.unique(sample_target)) != len(sample_target):
        raise ValueError("Collisione fra eventi dopo il downsampling.")
    timing_error_seconds = (
        sample_target.astype(np.float64) / target_sfreq - time_seconds
    )
    return {
        "sample_source": sample_source,
        "sample_200hz": sample_target,
        "time_seconds": time_seconds,
        "event_id": event_id,
        "timing_error_seconds": timing_error_seconds,
    }


def expected_output_length(n_times: int, up: int, down: int) -> int:
    return (n_times * up + down - 1) // down


def is_complete(
    eeg_path: Path,
    events_path: Path,
    metadata_path: Path,
    *,
    source_path: Path,
    target_sfreq: float,
) -> bool:
    if not (eeg_path.is_file() and events_path.is_file() and metadata_path.is_file()):
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        eeg = np.load(eeg_path, mmap_mode="r")
        with np.load(events_path) as events:
            n_events = len(events["event_id"])
        return (
            metadata.get("status") == "complete"
            and Path(metadata["source_path"]).resolve() == source_path.resolve()
            and math.isclose(
                float(metadata["target_sfreq"]),
                target_sfreq,
                abs_tol=1e-09,
                rel_tol=0.0,
            )
            and (list(eeg.shape) == metadata["output_shape"])
            and (str(eeg.dtype) == metadata["output_dtype"])
            and (n_events == metadata["n_events"])
        )
    except Exception:
        return False


def ensure_space(folder: Path, *, output_bytes: int, reserve_gib: float) -> None:
    usage = shutil.disk_usage(folder)
    reserve_bytes = int(reserve_gib * 1024**3)
    required = output_bytes + reserve_bytes
    if usage.free < required:
        raise RuntimeError(
            f"Spazio insufficiente: liberi {human_bytes(usage.free)}, richiesti {human_bytes(required)}."
        )


def process_item(
    item: WorkItem,
    *,
    output_root: Path,
    target_sfreq: float,
    reserve_gib: float,
    overwrite: bool,
    delete_source: bool,
) -> dict[str, Any]:
    (eeg_path, events_path, metadata_path) = output_paths(
        output_root,
        subject=item.subject,
        session=item.session,
        partition=item.partition,
    )
    eeg_path.parent.mkdir(parents=True, exist_ok=True)
    if is_complete(
        eeg_path,
        events_path,
        metadata_path,
        source_path=item.source_path,
        target_sfreq=target_sfreq,
    ) and (not overwrite):
        print(f"[SKIP COMPLETE] {item.source_path}")
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    existing = [p for p in (eeg_path, events_path, metadata_path) if p.exists()]
    if existing and (not overwrite):
        raise FileExistsError(
            "Output presente ma non verificabile. Usa --overwrite oppure rimuovi:\n  "
            + "\n  ".join((str(p) for p in existing))
        )
    if overwrite:
        for path in existing:
            path.unlink()
    temp_eeg = eeg_path.with_suffix(".partial.npy")
    temp_events = events_path.with_suffix(".partial.npz")
    temp_metadata = metadata_path.with_suffix(".partial.json")
    for path in (temp_eeg, temp_events, temp_metadata):
        if path.exists():
            path.unlink()
    started = time.time()
    print(f"[LOAD] {item.source_path}")
    data = load_dict(item.source_path)
    (matrix, names, types, eeg_indices, stim_index) = validate_layout(data)
    source_sfreq = float(data["sfreq"])
    if not math.isclose(source_sfreq, SOURCE_SFREQ, abs_tol=1e-09, rel_tol=0.0):
        raise ValueError(f"Sampling rate inatteso: {source_sfreq} Hz.")
    ratio = source_sfreq / target_sfreq
    down = int(round(ratio))
    if down < 1 or not math.isclose(ratio, down, abs_tol=1e-09, rel_tol=0.0):
        raise ValueError(
            f"Questo script richiede un fattore intero di downsampling; rapporto trovato {ratio}."
        )
    up = 1
    input_n_times = int(matrix.shape[1])
    output_n_times = expected_output_length(input_n_times, up, down)
    output_shape = (len(eeg_indices), output_n_times)
    output_bytes = (
        int(np.prod(output_shape, dtype=np.int64)) * np.dtype(np.float32).itemsize
    )
    ensure_space(eeg_path.parent, output_bytes=output_bytes, reserve_gib=reserve_gib)
    events = extract_events(
        np.asarray(matrix[stim_index]),
        source_sfreq=source_sfreq,
        target_sfreq=target_sfreq,
        output_n_times=output_n_times,
    )
    print(
        f"[PLAN] {matrix.shape} {matrix.dtype} @ {source_sfreq:g} Hz -> {output_shape} float32 @ {target_sfreq:g} Hz\n       EEG output: {human_bytes(output_bytes)} | eventi: {len(events['event_id'])} | errore temporale max: {np.max(np.abs(events['timing_error_seconds'])) * 1000:.3f} ms"
    )
    destination = np.lib.format.open_memmap(
        temp_eeg, mode="w+", dtype=np.float32, shape=output_shape
    )
    stats: list[dict[str, Any]] = []
    try:
        for out_idx, src_idx in enumerate(eeg_indices):
            name = names[src_idx]
            signal = np.asarray(matrix[src_idx], dtype=np.float64)
            if not np.isfinite(signal).all():
                raise ValueError(f"NaN/Inf nel canale {name}.")
            resampled = resample_poly(
                signal,
                up=up,
                down=down,
                axis=-1,
                window=("kaiser", 5.0),
                padtype="line",
            )
            if resampled.shape != (output_n_times,):
                raise RuntimeError(f"Shape inattesa per {name}: {resampled.shape}")
            if not np.isfinite(resampled).all():
                raise ValueError(f"NaN/Inf dopo resampling nel canale {name}.")
            destination[out_idx] = resampled.astype(np.float32, copy=False)
            stats.append(
                {
                    "name": name,
                    "source_index": src_idx,
                    "output_index": out_idx,
                    "input_rms": float(np.sqrt(np.mean(signal * signal))),
                    "output_rms": float(np.sqrt(np.mean(resampled * resampled))),
                    "input_min": float(signal.min()),
                    "input_max": float(signal.max()),
                    "output_min": float(resampled.min()),
                    "output_max": float(resampled.max()),
                }
            )
            print(
                f"[CHANNEL {out_idx + 1:02d}/{len(eeg_indices):02d}] {name}", flush=True
            )
            del signal, resampled
        destination.flush()
    finally:
        del destination
    np.savez_compressed(
        temp_events,
        sample_source=events["sample_source"],
        sample_200hz=events["sample_200hz"],
        time_seconds=events["time_seconds"],
        event_id=events["event_id"],
        timing_error_seconds=events["timing_error_seconds"],
    )
    metadata: dict[str, Any] = {
        "status": "complete",
        "created_at_utc": utc_now(),
        "subject": item.subject,
        "session": item.session,
        "partition": item.partition,
        "source_path": str(item.source_path.resolve()),
        "source_file_size_bytes": item.source_path.stat().st_size,
        "source_shape": list(matrix.shape),
        "source_dtype": str(matrix.dtype),
        "source_sfreq": source_sfreq,
        "target_sfreq": target_sfreq,
        "resampling_up": up,
        "resampling_down": down,
        "resampler": "scipy.signal.resample_poly",
        "resampling_window": ["kaiser", 5.0],
        "resampling_padtype": "line",
        "output_path": str(eeg_path.resolve()),
        "output_shape": list(output_shape),
        "output_dtype": "float32",
        "events_path": str(events_path.resolve()),
        "n_events": int(len(events["event_id"])),
        "n_unique_event_ids": int(len(np.unique(events["event_id"]))),
        "max_abs_event_timing_error_ms": float(
            np.max(np.abs(events["timing_error_seconds"])) * 1000
        ),
        "eeg_channel_names": [names[i] for i in eeg_indices],
        "source_channel_names": names,
        "source_channel_types": types,
        "stim_channel_index": stim_index,
        "stim_channel_name": names[stim_index],
        "source_highpass": data.get("highpass"),
        "source_lowpass": data.get("lowpass"),
        "units_note": "Nessun riscalamento applicato; le unità numeriche del mirror HF sono preservate.",
        "applied": ["polyphase anti-alias resampling", "float32 cast"],
        "not_applied": [
            "channel selection",
            "epoching",
            "crop",
            "padding",
            "baseline correction",
            "MVNN",
            "rereferencing",
            "normalization",
            "artifact rejection",
        ],
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "processing_seconds": float(time.time() - started),
        "channel_stats": stats,
    }
    temp_metadata.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_eeg.replace(eeg_path)
    temp_events.replace(events_path)
    temp_metadata.replace(metadata_path)
    if not is_complete(
        eeg_path,
        events_path,
        metadata_path,
        source_path=item.source_path,
        target_sfreq=target_sfreq,
    ):
        raise RuntimeError("Verifica finale degli output fallita.")
    print(f"[VERIFIED] {eeg_path} ({human_bytes(eeg_path.stat().st_size)})")
    del matrix, data
    gc.collect()
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downsampling THINGS-EEG2 raw continuo 1000 -> 200 Hz."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Radice del dataset, es. data/THINGS-EEG2.",
    )
    parser.add_argument(
        "--raw-root", type=Path, default=None, help="Default: ROOT/raw-eeg."
    )
    parser.add_argument(
        "--output-root", type=Path, default=None, help="Default: ROOT/canonical_200hz."
    )
    parser.add_argument(
        "--subjects", nargs="+", default=["1-10"], help="Esempi: 1, 1-5, 1 3 8-10."
    )
    parser.add_argument(
        "--sessions", nargs="+", default=["1-4"], help="Esempi: 1, 1-4, 1 3."
    )
    parser.add_argument(
        "--partitions",
        nargs="+",
        choices=["training", "test"],
        default=["training", "test"],
    )
    parser.add_argument("--target-sfreq", type=float, default=DEFAULT_TARGET_SFREQ)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Senza questo flag viene eseguito solo il dry run.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reserve-gib", type=float, default=2.0)
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.target_sfreq <= 0:
        raise SystemExit("--target-sfreq deve essere > 0.")
    if args.reserve_gib < 0:
        raise SystemExit("--reserve-gib deve essere >= 0.")
    try:
        subjects = parse_numbers(args.subjects, minimum=1, maximum=10)
        sessions = parse_numbers(args.sessions, minimum=1, maximum=4)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    root = args.root.expanduser().resolve()
    raw_root = (
        args.raw_root.expanduser().resolve() if args.raw_root else root / "raw-eeg"
    )
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else root / "canonical_200hz"
    )
    if not raw_root.is_dir():
        raise SystemExit(f"Cartella raw inesistente: {raw_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        items = discover(raw_root, subjects, sessions, args.partitions)
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2
    print(f"[ROOT]         {root}")
    print(f"[RAW ROOT]     {raw_root}")
    print(f"[OUTPUT ROOT]  {output_root}")
    print(f"[SUBJECTS]     {subjects}")
    print(f"[SESSIONS]     {sessions}")
    print(f"[PARTITIONS]   {args.partitions}")
    print(f"[TARGET]       {args.target_sfreq:g} Hz")
    print(f"[MODE]         {('EXECUTE' if args.execute else 'DRY-RUN')}")
    print(f"[FILES]        {len(items)}")
    print(
        f"[SOURCE SIZE]  {human_bytes(sum((x.source_path.stat().st_size for x in items)))}"
    )
    if not args.execute:
        for item in items:
            (eeg_path, events_path, metadata_path) = output_paths(
                output_root,
                subject=item.subject,
                session=item.session,
                partition=item.partition,
            )
            status = (
                "complete"
                if is_complete(
                    eeg_path,
                    events_path,
                    metadata_path,
                    source_path=item.source_path,
                    target_sfreq=args.target_sfreq,
                )
                else "pending"
            )
            print(
                f"  [{status:8}] {human_bytes(item.source_path.stat().st_size):>11} {item.source_path}"
            )
        print("\n[DRY-RUN] Aggiungi --execute per iniziare.")
        return 0
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, item in enumerate(items, start=1):
        print(
            f"\n===== [{index}/{len(items)}] sub-{item.subject:02d} ses-{item.session:02d} {item.partition} ====="
        )
        try:
            results.append(
                process_item(
                    item,
                    output_root=output_root,
                    target_sfreq=args.target_sfreq,
                    reserve_gib=args.reserve_gib,
                    overwrite=args.overwrite,
                    delete_source=False,
                )
            )
        except Exception as exc:
            message = (
                f"sub-{item.subject:02d}/ses-{item.session:02d}/{item.partition}: {exc}"
            )
            errors.append(message)
            print(f"[ERROR] {message}", file=sys.stderr)
            if not args.keep_going:
                break
    manifest = {
        "generated_at_utc": utc_now(),
        "dataset_root": str(root),
        "target_sfreq": args.target_sfreq,
        "completed": [
            {
                "subject": x.get("subject"),
                "session": x.get("session"),
                "partition": x.get("partition"),
                "source_path": x.get("source_path"),
                "output_path": x.get("output_path"),
                "output_shape": x.get("output_shape"),
                "n_events": x.get("n_events"),
            }
            for x in results
        ],
        "errors": errors,
    }
    manifest_path = output_root / "downsample_manifest.json"
    atomic_json(manifest_path, manifest)
    print(f"\n[MANIFEST] {manifest_path}")
    if errors:
        return 3
    print("[COMPLETE] Operazione terminata senza errori.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
