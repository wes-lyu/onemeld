# SPDX-License-Identifier: Apache-2.0
"""Model-independent Images API registration, upload, and cancellation contracts."""

import asyncio
import io
from collections.abc import AsyncIterator
from typing import Literal
from unittest.mock import AsyncMock, create_autospec

import pytest
from fastapi import Request, UploadFile
from fastapi.testclient import TestClient
from PIL import Image

from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateChunk, GenerateRequest
from sglang_omni.serve.openai_api import (
    MAX_IMAGE_EDIT_PIXELS,
    MAX_IMAGE_EDIT_UPLOAD_BYTES,
    create_app,
    image_result_with_disconnect_watch,
    parse_image_size,
    read_image_edit_upload,
)
from tests.unit_test.fixtures.image_media import (
    png_with_bad_idat_crc,
    register_test_images,
)


def test_image_registrar_receives_initialized_application() -> None:
    backend = create_autospec(Client, instance=True)
    app = create_app(
        backend,
        model_name="registered-image-model",
        image_api_registrar=register_test_images,
    )
    with TestClient(app) as client:
        response = client.post("/v1/images/generations")
    assert response.status_code == 200
    assert response.json() == {"model": "registered-image-model"}
    backend.generate.assert_not_called()


def test_image_registrars_cannot_register_duplicate_routes() -> None:
    with pytest.raises(
        ValueError, match="either supports_image_api or image_api_registrar"
    ):
        create_app(
            create_autospec(Client, instance=True),
            supports_image_api=True,
            image_api_registrar=register_test_images,
        )


@pytest.mark.parametrize("size", ["32", "32x", "32x64x96", "0x32", "32x-1", "ax32"])
def test_image_size_rejects_malformed_dimensions(size: str) -> None:
    with pytest.raises(ValueError, match="size must be WxH"):
        parse_image_size(size)


def test_image_size_preserves_width_height_order() -> None:
    assert parse_image_size("64x32") == (64, 32)


@pytest.mark.asyncio
@pytest.mark.parametrize("image_format", ["PNG", "JPEG", "WEBP"])
async def test_image_upload_preserves_content_and_closes_file(
    image_format: Literal["PNG", "JPEG", "WEBP"],
) -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 4), "red").save(buffer, format=image_format)
    image_bytes = buffer.getvalue()
    buffer.seek(0)
    upload = UploadFile(file=buffer, filename="reference")
    assert await read_image_edit_upload(upload) == image_bytes
    assert buffer.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["empty", "bytes", "pixels", "format", "corrupt"])
async def test_image_upload_rejects_invalid_content_and_closes_file(
    failure: Literal["empty", "bytes", "pixels", "format", "corrupt"],
) -> None:
    if failure == "empty":
        image_bytes = b""
        error = "must not be empty"
    elif failure == "bytes":
        image_bytes = b"x" * (MAX_IMAGE_EDIT_UPLOAD_BYTES + 1)
        error = "bytes"
    elif failure == "corrupt":
        image_bytes = png_with_bad_idat_crc()
        error = "not a valid image"
    else:
        with io.BytesIO() as buffer:
            if failure == "pixels":
                Image.new("L", (MAX_IMAGE_EDIT_PIXELS // 4096 + 1, 4096)).save(
                    buffer, format="PNG"
                )
                error = "pixels"
            else:
                Image.new("RGB", (8, 4)).save(buffer, format="BMP")
                error = "format"
            image_bytes = buffer.getvalue()
    buffer = io.BytesIO(image_bytes)
    with pytest.raises(ValueError, match=error):
        await read_image_edit_upload(UploadFile(file=buffer, filename="reference"))
    assert buffer.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("disconnect", [True, False])
async def test_image_result_cancellation_aborts_backend_and_closes_stream(
    disconnect: bool,
) -> None:
    entered = asyncio.Event()
    closed = asyncio.Event()
    backend = create_autospec(Client, instance=True)
    request = create_autospec(Request, instance=True)
    request.is_disconnected = AsyncMock(
        side_effect=lambda: disconnect and entered.is_set()
    )

    async def stream() -> AsyncIterator[GenerateChunk]:
        try:
            entered.set()
            await asyncio.Event().wait()
            yield GenerateChunk(request_id="request")
        finally:
            closed.set()

    backend.generate.return_value = stream()
    task = asyncio.create_task(
        image_result_with_disconnect_watch(
            request, backend, GenerateRequest(prompt="draw", stream=False), "request"
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=1)
    if not disconnect:
        task.cancel()
    else:
        pass
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)
    backend.abort.assert_awaited_once_with("request")
    assert closed.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("empty", [True, False])
async def test_image_result_completion_closes_stream_without_aborting(
    empty: bool,
) -> None:
    closed = asyncio.Event()
    backend = create_autospec(Client, instance=True)
    request = create_autospec(Request, instance=True)
    request.is_disconnected = AsyncMock(return_value=False)
    expected = GenerateChunk(request_id="request", image="encoded-image")

    async def stream() -> AsyncIterator[GenerateChunk]:
        try:
            if not empty:
                yield expected
            else:
                pass
        finally:
            closed.set()

    backend.generate.return_value = stream()
    result = image_result_with_disconnect_watch(
        request, backend, GenerateRequest(prompt="draw", stream=False), "request"
    )
    if empty:
        with pytest.raises(RuntimeError, match="Pipeline returned no result"):
            await result
    else:
        assert await result is expected
    backend.abort.assert_not_awaited()
    assert closed.is_set()
