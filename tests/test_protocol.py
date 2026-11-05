import csv, json, tempfile, unittest
from pathlib import Path
import numpy as np
from data_loader.protocol import (
    Moments,
    partition_images,
    validate_partitions,
    fit_statistics,
    load_statistics,
)


class ProtocolTests(unittest.TestCase):
    def rows(self):
        return [
            dict(visual_index=str(i), concept_uid="cat", partition="training")
            for i in range(10)
        ]

    def fixture(self, base):
        index = base / "index"
        index.mkdir()
        (index / "splits").mkdir()
        rows = self.rows()
        splits = partition_images(rows)
        with (index / "visuals.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        for s, v in splits.items():
            np.save(index / "splits" / f"{s}_visual_indices.npy", v)
        ids = [
            splits["train"][0],
            splits["val"][0],
            splits["evaluation"][0],
            splits["train"][1],
        ]
        dtype = [
            ("recording_id", "i4"),
            ("visual_index", "i4"),
            ("clean_valid_length", "i4"),
            ("onset_sample", "i4"),
        ]
        np.save(
            index / "trials.npy",
            np.array(
                [
                    (0, ids[0], 40, 0),
                    (0, ids[1], 40, 40),
                    (0, ids[2], 40, 80),
                    (1, ids[3], 40, 0),
                ],
                dtype=dtype,
            ),
        )
        recordings = []
        for subject in [1, 10]:
            name = f"eeg_{subject}.npy"
            values = np.full((2, 120), 1e8 if subject == 10 else 1e6, dtype=np.float32)
            if subject == 1:
                values[:, :40] = np.arange(80).reshape(2, 40)
            np.save(base / name, values)
            recordings.append(
                dict(
                    subject=subject,
                    partition="training",
                    channel_names=["a", "b"],
                    recording_id=len(recordings),
                    eeg_path=name,
                )
            )
        (index / "recordings.json").write_text(json.dumps(recordings))
        return index, splits

    def test_partition_is_reproducible_and_disjoint(self):
        a = partition_images(self.rows())
        self.assertEqual(a, partition_images(self.rows()[::-1]))
        self.assertEqual([len(v) for v in a.values()], [8, 1, 1])
        self.assertEqual(len(set(sum(a.values(), []))), 10)

    def test_population_moments_weight_valid_samples(self):
        windows = [np.array([[1.0, 3.0], [2.0, 8.0]]), np.array([[9.0], [5.0]])]
        m = Moments(2)
        for x in windows:
            m.update(x)
        mean, std = m.result()
        expected = np.concatenate(windows, axis=1)
        np.testing.assert_allclose(mean, expected.mean(axis=1))
        np.testing.assert_allclose(std, expected.std(axis=1))
        m = Moments(1)
        m.update(np.ones((1, 40)))
        self.assertEqual(m.result()[1][0], 1e-6)

    def test_statistics_exclude_development_evaluation_and_target(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            index, _ = self.fixture(base)
            fit_statistics(base, index, base / "stats")
            mean, std = load_statistics(base / "stats", ["a", "b"], index)
            expected = np.arange(80).reshape(2, 40)
            np.testing.assert_allclose(mean, expected.mean(axis=1))
            np.testing.assert_allclose(std, expected.std(axis=1))
            self.assertEqual(
                json.loads((base / "stats/source_statistics.json").read_text())[
                    "trial_count"
                ],
                1,
            )

    def test_statistics_reject_modified_partition(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            index, _ = self.fixture(base)
            fit_statistics(base, index, base / "stats")
            with (index / "splits/train_visual_indices.npy").open("ab") as f:
                f.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "provenance"):
                load_statistics(base / "stats", ["a", "b"], index)

    def test_validation_evaluation_photo_reuse_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            index, splits = self.fixture(Path(d))
            np.save(index / "splits/evaluation_visual_indices.npy", splits["val"])
            with self.assertRaisesRegex(ValueError, "leakage"):
                validate_partitions(index)


class LoaderIntegrationTests(unittest.TestCase):
    fixture = ProtocolTests.fixture
    rows = ProtocolTests.rows

    def test_target_uses_frozen_source_transform_and_zero_padding(self):
        from data_loader.things_eeg2_reve_dataset import ThingsEEG2ReveDataset
        from tools.build_things_eeg2_index import TRIAL_DTYPE

        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            index, splits = self.fixture(base)
            old = np.load(index / "trials.npy")
            trials = np.zeros(len(old), dtype=TRIAL_DTYPE)
            for name in old.dtype.names:
                trials[name] = old[name]
            trials["subject"] = [1, 1, 1, 10]
            trials["session"] = 1
            trials["post400_valid_length"] = 40
            trials["next_event_sample"] = 120
            trials["event_id"] = 1
            trials["visual_index"][-1] = splits["evaluation"][0]
            np.save(index / "trials.npy", trials)
            channels = [f"c{i}" for i in range(63)]
            recordings = json.loads((index / "recordings.json").read_text())
            for recording in recordings:
                subject = recording["subject"]
                values = np.full((63, 120), 1e8, dtype=np.float32)
                if subject == 1:
                    values[:, :40] = np.arange(63 * 40).reshape(63, 40)
                np.save(base / recording["eeg_path"], values)
                (base / f"events_{subject}.npz").touch()
                (base / f"meta_{subject}.json").write_text("{}")
                recording.update(
                    channel_names=channels,
                    eeg_shape=[63, 120],
                    session=1,
                    trial_start=0,
                    trial_stop=len(trials),
                    events_path=f"events_{subject}.npz",
                    metadata_path=f"meta_{subject}.json",
                )
            (index / "recordings.json").write_text(json.dumps(recordings))
            (index / "index_metadata.json").write_text(
                json.dumps(dict(visual_count=10, recording_count=2, target_sfreq=200))
            )
            positions = index / "reve_positions"
            positions.mkdir()
            np.save(
                positions / "things_eeg2_63_positions.npy",
                np.arange(189, dtype=np.float32).reshape(63, 3),
            )
            fit_statistics(base, index, base / "stats")
            loader = ThingsEEG2ReveDataset(
                root=base,
                clip_cache=base / "nonexistent_cache",
                index_dir=index,
                stats_root=base / "stats",
                split="evaluation",
                subjects=[10],
                strict=False,
                return_clip_image=False,
            )
            sample = loader[0]
            self.assertTrue((sample["eeg"][:, :40] == 15).all())
            self.assertTrue((sample["eeg"][:, 40:] == 0).all())
            self.assertEqual(int(sample["valid_mask"].sum()), 40)
            self.assertNotIn("clip_image", sample)


if __name__ == "__main__":
    unittest.main()
