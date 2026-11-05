import unittest
import torch
from eval.eval_metrics import two_way_identification


class MetricTests(unittest.TestCase):
    def score(self, a, b):
        return two_way_identification(
            a, b, torch.nn.Identity(), lambda x: x, device="cpu", batch_size=2
        )

    def test_pairing_and_strict_ties(self):
        images = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]], [[[4.0, 1.0], [2.0, 3.0]]]])
        self.assertAlmostEqual(float(self.score(images, images)), 1.0)
        self.assertAlmostEqual(float(self.score(images.flip(0), images)), 0.0)
        self.assertAlmostEqual(
            float(self.score(torch.ones_like(images), torch.ones_like(images))), 0.0
        )
