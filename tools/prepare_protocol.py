"""Create a separate 8/1/1 index and fit frozen source-only statistics."""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_loader.protocol import partition_images, validate_partitions, fit_statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--input-index", required=True, type=Path)
    p.add_argument("--output-index", required=True, type=Path)
    p.add_argument("--statistics", required=True, type=Path)
    p.add_argument("--categories", type=int, choices=[100, 1654], default=1654)
    a = p.parse_args()
    if a.output_index.exists():
        raise FileExistsError(
            "Use a new output index; input indexes are never overwritten"
        )
    with (a.input_index / "visuals.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    splits = partition_images(rows)
    a.output_index.mkdir(parents=True)
    for name in ["trials.npy", "recordings.json", "index_metadata.json"]:
        shutil.copy2(a.input_index / name, a.output_index / name)
    shutil.copytree(a.input_index / "reve_positions", a.output_index / "reve_positions")
    assignment = {v: s for s, values in splits.items() for v in values}
    for row in rows:
        row["split"] = assignment.get(int(row["visual_index"]), "native_test_unused")
    with (a.output_index / "visuals.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    folder = a.output_index / "splits"
    folder.mkdir()
    for name, values in splits.items():
        np.save(
            folder / f"{name}_visual_indices.npy", np.asarray(values, dtype=np.int32)
        )
    (folder / "split_metadata.json").write_text(
        json.dumps(
            {
                "seed": 42,
                "protocol": "8/1/1",
                "counts": {s: len(v) for s, v in splits.items()},
                "source_subjects": list(range(1, 10)),
                "heldout_subject": 10,
            },
            indent=2,
        )
        + "\n"
    )
    # Refresh the modified CSV hash expected by strict readers.
    from data_loader.protocol import digest

    meta = json.loads((a.output_index / "index_metadata.json").read_text())
    meta["visuals_sha256"] = digest(a.output_index / "visuals.csv")
    (a.output_index / "index_metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )
    validate_partitions(a.output_index)
    concepts = sorted({r["concept_uid"] for r in rows if r["partition"] == "training"})
    if a.categories == 100:
        concepts = sorted(
            np.random.default_rng(42).choice(concepts, 100, replace=False).tolist()
        )
    fit_statistics(a.root, a.output_index, a.statistics, concepts)


if __name__ == "__main__":
    main()
