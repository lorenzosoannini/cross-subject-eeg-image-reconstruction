import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from data_loader.images import load_image
from model.heads import BlurryReconDecoder


class ImageTests(unittest.TestCase):
    def test_center_crop_preserves_study_geometry(self):
        with tempfile.TemporaryDirectory() as d:
            values = np.zeros((224, 448, 3), dtype=np.uint8)
            values[:, :, 0] = 255
            values[:, 112:336] = [0, 0, 255]
            filename = Path(d) / "wide.png"
            Image.fromarray(values).save(filename)
            image = load_image(filename)
            self.assertEqual(tuple(image.shape), (3, 224, 224))
            self.assertTrue(torch.all(image[2] == 1))
            self.assertTrue(torch.all(image[0] == 0))

    def test_blurry_latent_matches_224_pixel_autoencoder_grid(self):
        with torch.no_grad():
            latent = BlurryReconDecoder().eval()(torch.randn(1, 256, 1664))
        self.assertEqual(tuple(latent.shape), (1, 4, 224 // 8, 224 // 8))
