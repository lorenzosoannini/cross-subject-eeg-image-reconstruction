"""Study image preprocessing: bicubic shortest-side resize and center crop."""

from pathlib import Path
import numpy as np
import torch
from PIL import Image


def load_image(path: Path, size: int = 224) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Missing THINGS image: {path}")
    with Image.open(path) as source:
        image = source.convert("RGB")
        width, height = image.size
        scale = float(size) / float(min(width, height))
        resized = (
            max(size, int(round(width * scale))),
            max(size, int(round(height * scale))),
        )
        resampling = getattr(Image, "Resampling", Image).BICUBIC
        image = image.resize(resized, resample=resampling)
        left = (image.width - size) // 2
        top = (image.height - size) // 2
        image = image.crop((left, top, left + size, top + size))
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1).copy())
