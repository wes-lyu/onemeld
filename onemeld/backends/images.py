# SPDX-License-Identifier: Apache-2.0
"""Turn a framework's generated image into the image step result."""

import base64
import io

import numpy as np
import torch
from PIL import Image

from onemeld.backends.base import StepResult

GeneratedImage = Image.Image | np.ndarray | torch.Tensor


def to_pil(image: GeneratedImage) -> Image.Image:
    """Accept a PIL image, or an HWC/CHW array or tensor in [0, 1] floats or uint8."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    else:
        pass
    array = (
        image.detach().float().cpu().numpy()
        if isinstance(image, torch.Tensor)
        else image
    )
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    else:
        pass
    if array.ndim != 3:
        raise ValueError(f"Expected one image of 3 dimensions, got shape {array.shape}")
    elif array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    else:
        pass
    if array.dtype != np.uint8:
        array = (np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    else:
        pass
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    else:
        pass
    return Image.fromarray(array, mode="RGB")


def image_result(image: GeneratedImage) -> StepResult:
    buffer = io.BytesIO()
    to_pil(image).save(buffer, format="PNG")
    return {
        "modality": "image",
        "image_b64": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }
