# SPDX-License-Identifier: Apache-2.0
"""Generated images in each framework representation become one PNG result."""

import base64
import io

import numpy as np
import pytest
import torch
from PIL import Image

from onemeld.backends.images import GeneratedImage, image_result

RED = (255, 0, 0)


def red_float_hwc() -> np.ndarray:
    array = np.zeros((4, 6, 3), dtype=np.float32)
    array[..., 0] = 1.0
    return array


@pytest.mark.parametrize(
    "image",
    [
        Image.new("RGB", (6, 4), RED),
        Image.new("RGBA", (6, 4), (*RED, 255)),
        red_float_hwc(),
        (red_float_hwc() * 255).astype(np.uint8),
        torch.from_numpy(red_float_hwc()).permute(2, 0, 1),
        torch.from_numpy(red_float_hwc()).permute(2, 0, 1).unsqueeze(0),
    ],
    ids=["pil", "pil-rgba", "float-hwc", "uint8-hwc", "tensor-chw", "tensor-batch"],
)
def test_each_representation_becomes_the_same_png(image: GeneratedImage) -> None:
    result = image_result(image)
    assert result["modality"] == "image"
    decoded = Image.open(io.BytesIO(base64.b64decode(result["image_b64"])))
    assert decoded.format == "PNG"
    assert decoded.size == (6, 4)
    assert decoded.convert("RGB").getpixel((0, 0)) == RED


def test_more_than_one_image_is_rejected() -> None:
    with pytest.raises(ValueError, match="one image"):
        image_result(np.zeros((2, 4, 6, 3), dtype=np.float32))
