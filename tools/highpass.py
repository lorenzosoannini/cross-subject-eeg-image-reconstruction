"""High-pass the canonical continuous EEG before indexing; never fit session statistics."""

import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("Use a new output directory")
    files = sorted(a.input.glob("sub-*/ses-*/raw_eeg_*_200hz.npy"))
    if not files:
        raise FileNotFoundError("No canonical 200 Hz EEG recordings")
    sos = butter(4, 0.5, btype="highpass", fs=200, output="sos")
    for source in files:
        rel = source.relative_to(a.input)
        destination = a.output / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        meta = json.loads(source.with_suffix(".json").read_text())
        values = np.load(source, mmap_mode="r")
        if values.ndim != 2 or values.shape[0] != 63:
            raise ValueError("Expected 63 EEG channels")
        result = np.lib.format.open_memmap(
            destination, mode="w+", dtype=np.float32, shape=values.shape
        )
        for channel in range(63):
            result[channel] = sosfiltfilt(
                sos, np.asarray(values[channel], dtype=np.float64)
            ).astype(np.float32)
        result.flush()
        del result
        events = source.with_name(source.name.replace("_200hz.npy", "_events.npz"))
        shutil.copy2(events, destination.with_name(events.name))
        meta.update(
            highpass={
                "type": "Butterworth",
                "order": 4,
                "cutoff_hz": 0.5,
                "application": "sosfiltfilt",
                "normalization": "none",
            },
            output_path=str(destination.resolve()),
        )
        destination.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
        print(destination, flush=True)


if __name__ == "__main__":
    main()
