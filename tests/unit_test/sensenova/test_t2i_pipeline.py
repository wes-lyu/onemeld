# SPDX-License-Identifier: Apache-2.0
import asyncio
import base64
import io
from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock, Mock, create_autospec, patch

import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

import sglang_omni.models.sensenova_u1.generation as generation_module
import sglang_omni.models.sensenova_u1.stages as stages_module
from sglang_omni.models.sensenova_u1.cache_dit import SenseNovaCacheDit
from sglang_omni.models.sensenova_u1.image_processing import (
    decode_reference_images,
    prepare_reference_images,
    resolve_edit_output_size,
    resolve_reference_output_size,
)
from sglang_omni.models.sensenova_u1.neo_unify.generation_inputs import (
    build_it2i_inputs,
    build_t2i_text_inputs,
)
from sglang_omni.models.sensenova_u1.neo_unify.modeling_neo_chat import NEOChatModel
from sglang_omni.models.sensenova_u1.sampling import SenseNovaU1ImageEditSampling
from sglang_omni.models.sensenova_u1.tensor_parallel import (
    SenseNovaTensorParallelScheduler,
)
from sglang_omni.pipeline.stage.runtime import Stage
from sglang_omni.pipeline.umm import UMMRequestError
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.tensor_parallel_scheduler import TensorParallelScheduler
from sglang_omni.serve.openai_errors import generation_error_status_code
from tests.unit_test.fixtures.pipeline_fakes import (
    FakeRelay,
    RecordingStageControlPlane,
)

pytest_plugins = ["tests.unit_test.fixtures.sensenova_generation"]


@pytest.fixture
def tokenizer() -> PreTrainedTokenizerBase:
    tokenizer = create_autospec(
        PreTrainedTokenizerBase,
        instance=True,
        return_value={"input_ids": torch.tensor([[1, 2, 3]])},
    )
    tokenizer.model_max_length = 12288
    return tokenizer


@pytest.fixture
def prefix_model() -> Mock:
    return Mock(
        spec=NEOChatModel,
        language_model=Mock(),
        template="neo1_0",
        system_message="",
        patch_size=16,
        downsample_ratio=0.5,
        device=torch.device("cpu"),
        config=SimpleNamespace(
            llm_config=SimpleNamespace(max_position_embeddings=16384)
        ),
    )


@pytest.fixture
def prefix_tokenizer() -> PreTrainedTokenizerFast:
    backend = Tokenizer(WordLevel({"[UNK]": 0, "cat": 1}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        model_max_length=128,
        additional_special_tokens=["<IMG_CONTEXT>"],
    )


def test_t2i_stage_encodes_real_png_and_passes_source_arguments(
    cache_dit: SenseNovaCacheDit,
    t2i_generate_mock: Mock,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    model = Mock()
    t2i_generate_mock.return_value = torch.zeros((1, 3, 32, 64))
    payload = StagePayload(
        request_id="test",
        request=OmniRequest(
            inputs="a blue cat",
            params={"width": 64, "height": 32, "seed": 7},
        ),
        data=None,
    )

    result = generation_module.generate_image(payload, model, tokenizer, cache_dit)

    kwargs = t2i_generate_mock.call_args.kwargs
    assert (kwargs["sampling"].width, kwargs["sampling"].height) == (64, 32)
    assert kwargs["sampling"].seed == 7
    image = Image.open(io.BytesIO(base64.b64decode(result.data["image_b64"])))
    assert image.format == "PNG"
    assert image.size == (64, 32)


def test_t2i_stage_rejects_invalid_output(
    cache_dit: SenseNovaCacheDit,
    t2i_generate_mock: Mock,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    model = Mock()
    t2i_generate_mock.return_value = torch.zeros((1, 3, 16, 16))
    payload = StagePayload(
        request_id="test",
        request=OmniRequest(inputs="a cat", params={"width": 32, "height": 32}),
        data=None,
    )
    try:
        generation_module.generate_image(payload, model, tokenizer, cache_dit)
    except ValueError as exc:
        assert "invalid image tensor" in str(exc)
    else:
        raise AssertionError("Invalid model output was accepted")


def decode_result_pixel(payload):
    image = Image.open(io.BytesIO(base64.b64decode(payload.data["image_b64"])))
    return image.getpixel((0, 0))


def test_t2i_stage_batches_compatible_requests_with_per_request_seeds(
    cache_dit: SenseNovaCacheDit,
    t2i_generate_mock: Mock,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    tokenizer.side_effect = [{"input_ids": [1, 2, 3]}, {"input_ids": [1, 2, 3, 4, 5]}]
    model = Mock()
    t2i_generate_mock.return_value = torch.stack(
        [torch.full((3, 32, 64), -1.0), torch.full((3, 32, 64), 1.0)]
    )
    payloads = [
        StagePayload(
            request_id=f"test-{index}",
            request=OmniRequest(
                inputs=prompt,
                params={
                    "width": 64,
                    "height": 32,
                    "num_inference_steps": 12,
                    "guidance_scale": 2.5,
                    "seed": seed,
                },
            ),
            data=None,
        )
        for index, (prompt, seed) in enumerate(
            [("a blue cat", 7), ("a red dog on grass", 19)]
        )
    ]

    results = generation_module.generate_images(payloads, model, tokenizer, cache_dit)

    args = t2i_generate_mock.call_args.args
    kwargs = t2i_generate_mock.call_args.kwargs
    assert args[0] is model
    assert args[1] is tokenizer
    assert args[2] == ["a blue cat", "a red dog on grass"]
    assert kwargs["seeds"] == [7, 19]
    assert (kwargs["sampling"].width, kwargs["sampling"].height) == (64, 32)
    assert kwargs["sampling"].num_inference_steps == 12
    assert [decode_result_pixel(result) for result in results] == [
        (0, 0, 0),
        (255, 255, 255),
    ]


def test_image_generation_request_cost_counts_pixels_steps_and_cfg():
    payload = StagePayload(
        request_id="test",
        request=OmniRequest(
            inputs="a cat",
            params={
                "width": 64,
                "height": 32,
                "num_inference_steps": 12,
                "guidance_scale": 4.0,
            },
        ),
        data=None,
    )

    assert stages_module.image_generation_request_cost(payload) == 64 * 32 * 12 * 2


def test_image_generation_batch_key_separates_parameters_and_image_edits():
    def payload(request_id, inputs, **params):
        return StagePayload(
            request_id=request_id,
            request=OmniRequest(inputs=inputs, params=params),
            data=None,
        )

    first = payload("first", "a cat", width=64, height=32, seed=7)
    same = payload("same", "a dog", width=64, height=32, seed=19)
    different = payload("different", "a dog", width=32, height=32, seed=19)
    edit = payload("edit", {"task": "image_edit"})

    assert stages_module.image_generation_batch_key(
        first
    ) == stages_module.image_generation_batch_key(same)
    assert stages_module.image_generation_batch_key(
        first
    ) != stages_module.image_generation_batch_key(different)
    assert stages_module.image_generation_batch_key(edit) == ("image_edit", "edit")


def test_i2i_stage_decodes_reference_and_passes_source_arguments(
    cache_dit: SenseNovaCacheDit,
    it2i_generate_mock: Mock,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    reference = io.BytesIO()
    Image.new("RGB", (64, 32), (255, 0, 0)).save(reference, format="PNG")
    model = Mock()

    def edit_image(
        generation_model: NEOChatModel,
        model_tokenizer: PreTrainedTokenizerBase,
        prompt: str,
        references: list[Image.Image],
        *,
        sampling: SenseNovaU1ImageEditSampling,
    ) -> torch.Tensor:
        return torch.zeros((1, 3, 32, 64))

    it2i_generate_mock.side_effect = edit_image
    payload = StagePayload(
        request_id="test",
        request=OmniRequest(
            inputs={
                "task": "image_edit",
                "prompt": "make the background blue",
                "image_b64": base64.b64encode(reference.getvalue()).decode("ascii"),
            },
            params={
                "width": 64,
                "height": 32,
                "num_inference_steps": 12,
                "guidance_scale": 2.5,
                "img_cfg_scale": 1.5,
                "seed": 7,
                "do_resize": False,
                "size_explicit": True,
            },
        ),
        data=None,
    )

    result = generation_module.generate_image(payload, model, tokenizer, cache_dit)

    args = it2i_generate_mock.call_args.args
    kwargs = it2i_generate_mock.call_args.kwargs
    assert args[0] is model
    assert args[1] is tokenizer
    assert args[2] == "make the background blue"
    assert len(args[3]) == 1
    assert args[3][0].size == (64, 32)
    assert (kwargs["sampling"].width, kwargs["sampling"].height) == (64, 32)
    assert kwargs["sampling"].num_inference_steps == 12
    assert kwargs["sampling"].guidance_scale == 2.5
    assert kwargs["sampling"].img_cfg_scale == 1.5
    assert kwargs["sampling"].seed == 7
    image = Image.open(io.BytesIO(base64.b64decode(result.data["image_b64"])))
    assert image.format == "PNG"
    assert image.size == (64, 32)


def test_i2i_stage_rejects_invalid_reference_image(
    cache_dit: SenseNovaCacheDit,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    payload = StagePayload(
        request_id="test",
        request=OmniRequest(
            inputs={
                "task": "image_edit",
                "prompt": "edit this",
                "image_b64": "not-base64",
            }
        ),
        data=None,
    )

    try:
        generation_module.generate_image(payload, Mock(), tokenizer, cache_dit)
    except ValueError as exc:
        assert "invalid reference image" in str(exc)
    else:
        raise AssertionError("Invalid reference image was accepted")


def test_i2i_stage_preserves_reference_aspect_ratio_without_explicit_size(
    cache_dit: SenseNovaCacheDit,
    it2i_generate_mock: Mock,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    reference = io.BytesIO()
    Image.new("RGB", (64, 32), (255, 0, 0)).save(reference, format="PNG")
    model = Mock()

    def generate(*_args, **kwargs):
        width, height = (kwargs["sampling"].width, kwargs["sampling"].height)
        return torch.zeros((1, 3, height, width))

    it2i_generate_mock.side_effect = generate
    payload = StagePayload(
        request_id="test",
        request=OmniRequest(
            inputs={
                "task": "image_edit",
                "prompt": "edit this",
                "image_b64": [base64.b64encode(reference.getvalue()).decode("ascii")],
            },
            params={
                "width": 256,
                "height": 256,
                "do_resize": False,
                "size_explicit": False,
            },
        ),
        data=None,
    )

    result = generation_module.generate_image(payload, model, tokenizer, cache_dit)

    assert (
        it2i_generate_mock.call_args.kwargs["sampling"].width,
        it2i_generate_mock.call_args.kwargs["sampling"].height,
    ) == (384, 192)
    image = Image.open(io.BytesIO(base64.b64decode(result.data["image_b64"])))
    assert image.size == (384, 192)


@pytest.mark.parametrize("batched", [False, True])
def test_generation_runs_no_checkpoint_after_the_model_returns(
    monkeypatch: pytest.MonkeyPatch,
    batched: bool,
    prefix_model: Mock,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    """A cancellation check after the model returned discarded finished images and cost every TP request one more collective."""
    events: list[str] = []

    def generate(
        payload: StagePayload,
        model: NEOChatModel,
        tokenizer: PreTrainedTokenizerBase,
        cache_dit: SenseNovaCacheDit,
    ) -> StagePayload:
        events.append("model")
        return payload

    def generate_batch(
        payloads: list[StagePayload],
        model: NEOChatModel,
        tokenizer: PreTrainedTokenizerBase,
        cache_dit: SenseNovaCacheDit,
    ) -> list[StagePayload]:
        events.append("model")
        return payloads

    model = prefix_model
    model.eval.return_value = model
    model.to.return_value = model
    monkeypatch.setattr(
        stages_module, "generation_checkpoint", lambda: events.append("checkpoint")
    )
    monkeypatch.setattr(stages_module, "generate_image", generate)
    monkeypatch.setattr(stages_module, "generate_images", generate_batch)
    monkeypatch.setattr(stages_module, "register", lambda: None)
    monkeypatch.setattr(
        stages_module,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda model_path: tokenizer),
    )
    monkeypatch.setattr(
        stages_module,
        "AutoModel",
        SimpleNamespace(from_pretrained=lambda model_path, torch_dtype: model),
    )
    monkeypatch.setattr(
        stages_module,
        "resolve_concrete_device",
        lambda device, gpu_id: torch.device("cpu"),
    )
    scheduler = stages_module.create_generation_executor(
        "/model/sensenova",
        dtype="bfloat16",
        cooperative_cancellation=True,
        max_batch_size=2,
        enable_cache_dit=False,
        dit_layerwise_offload=False,
        dit_offload_prefetch_size=1,
        dit_layerwise_resident_layers=0,
        dit_layerwise_residency_policy="leading",
        pin_cpu_memory=False,
    )
    payloads = [
        StagePayload(f"request-{index}", OmniRequest("a cat", {}), None)
        for index in range(2)
    ]

    if batched:
        assert scheduler.batch_fn(payloads) == payloads
    else:
        assert scheduler.fn(payloads[0]) is payloads[0]

    assert events == ["checkpoint", "model"]


def test_multi_output_failure_cleans_cache_controller(
    cache_dit: SenseNovaCacheDit,
    t2i_generate_mock: Mock,
    tokenizer: PreTrainedTokenizerBase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The n>1 branch bypassed cache cleanup after a failed generation."""
    cleanup = Mock(wraps=generation_module.unmount_cache_dit_after_failure)
    monkeypatch.setattr(generation_module, "unmount_cache_dit_after_failure", cleanup)
    model = Mock()
    t2i_generate_mock.side_effect = RuntimeError("generation failed")
    payload = StagePayload(
        "multi", OmniRequest("a cat", {"n": 2, "width": 32, "height": 32}), None
    )
    with pytest.raises(RuntimeError, match="generation failed"):
        generation_module.generate_images([payload], model, tokenizer, cache_dit)
    cleanup.assert_called_once_with(cache_dit)


@pytest.mark.parametrize("size_explicit", [False, True])
@pytest.mark.parametrize(
    "guidance_scale, img_cfg_scale, branch_count",
    [(1.0, 1.0, 1), (4.0, 4.0, 2), (0.5, 1.0, 2)],
)
def test_i2i_stage_overrides_match_cost_and_tp_validation(
    prefix_model: Mock,
    it2i_generate_mock: Mock,
    tokenizer: PreTrainedTokenizerBase,
    guidance_scale: float,
    img_cfg_scale: float,
    branch_count: int,
    size_explicit: bool,
) -> None:
    """I2I cost and TP validation used options different from model execution."""
    reference = io.BytesIO()
    Image.new("RGB", (32, 32)).save(reference, format="PNG")
    payload = StagePayload(
        "edit",
        OmniRequest(
            {
                "task": "image_edit",
                "prompt": "edit this",
                "image_b64": base64.b64encode(reference.getvalue()).decode("ascii"),
            },
            {
                "width": 64,
                "height": 32,
                "size_explicit": size_explicit,
                "do_resize": False,
                "stage_params": {
                    "generate": {
                        "width": 96,
                        "height": 64,
                        "num_inference_steps": 4,
                        "guidance_scale": guidance_scale,
                        "img_cfg_scale": img_cfg_scale,
                    }
                },
            },
        ),
        None,
    )
    model = prefix_model
    cache_dit = create_autospec(SenseNovaCacheDit, instance=True)
    cache_dit.enabled_by_default = True
    output_width_pixels, output_height_pixels = (96, 64) if size_explicit else (96, 96)
    it2i_generate_mock.return_value = torch.zeros(
        (1, 3, output_height_pixels, output_width_pixels)
    )
    with patch.object(Image, "open", wraps=Image.open) as open_image:
        assert (
            stages_module.image_generation_request_cost(payload)
            == 96 * 64 * 4 * branch_count
        )
    open_image.assert_not_called()
    stages_module.validate_generation_request(payload, model, cache_dit, tokenizer)
    assert cache_dit.validate_request.call_args.kwargs["steps"] == 4
    result = generation_module.generate_image(payload, model, tokenizer, cache_dit)
    assert cache_dit.prepare.call_args.kwargs["steps"] == 4
    assert it2i_generate_mock.call_args.kwargs["sampling"].num_inference_steps == 4
    assert (
        it2i_generate_mock.call_args.kwargs["sampling"].width,
        it2i_generate_mock.call_args.kwargs["sampling"].height,
    ) == (
        output_width_pixels,
        output_height_pixels,
    )
    assert (
        it2i_generate_mock.call_args.kwargs["sampling"].guidance_scale == guidance_scale
    )
    assert (
        it2i_generate_mock.call_args.kwargs["sampling"].img_cfg_scale == img_cfg_scale
    )
    with Image.open(io.BytesIO(base64.b64decode(result.data["image_b64"]))) as image:
        decoded_image_size = image.size
    assert decoded_image_size == (
        output_width_pixels,
        output_height_pixels,
    )


@pytest.mark.parametrize("reference_count", [1, 3])
@pytest.mark.parametrize("do_resize", [False, True])
@pytest.mark.parametrize("reference_size_pixels", [(64, 32), (96, 64)])
def test_i2i_validation_decodes_references_without_resizing(
    cache_dit: SenseNovaCacheDit,
    prefix_model: Mock,
    tokenizer: PreTrainedTokenizerBase,
    reference_count: int,
    do_resize: bool,
    reference_size_pixels: tuple[int, int],
) -> None:
    """Validation must decode pixels but leave resizing to model preprocessing."""
    image_buffer = io.BytesIO()
    Image.new("RGB", reference_size_pixels).save(image_buffer, format="PNG")
    references = [
        base64.b64encode(image_buffer.getvalue()).decode("ascii")
    ] * reference_count
    parameters = {"width": 96, "height": 64, "do_resize": do_resize}
    options = SenseNovaU1ImageEditSampling.from_params(parameters)
    prepared_references = prepare_reference_images(references, options)
    output_width_pixels, output_height_pixels = resolve_edit_output_size(
        prepared_references[0].size, options
    )
    payload = StagePayload(
        "edit",
        OmniRequest(
            {"task": "image_edit", "prompt": "edit", "image_b64": references},
            parameters,
        ),
        None,
    )
    model = prefix_model
    with (
        patch.object(
            Image.Image, "load", autospec=True, side_effect=Image.Image.load
        ) as load,
        patch.object(
            Image.Image, "resize", autospec=True, side_effect=Image.Image.resize
        ) as resize,
    ):
        assert resolve_reference_output_size(references, options) == (
            output_width_pixels,
            output_height_pixels,
        )
        stages_module.validate_generation_request(payload, model, cache_dit, tokenizer)
    load.assert_called()
    resize.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_reference_index", [0, 1])
@pytest.mark.parametrize("image_format", ["JPEG", "PNG"])
@pytest.mark.parametrize("cooperative_cancellation", [False, True])
async def test_truncated_image_is_rejected_before_tp_fanout(
    cache_dit: SenseNovaCacheDit,
    prefix_model: Mock,
    it2i_generate_mock: Mock,
    tokenizer: PreTrainedTokenizerBase,
    invalid_reference_index: int,
    image_format: Literal["JPEG", "PNG"],
    cooperative_cancellation: bool,
) -> None:
    """A truncated reference killed TP execution instead of failing its request."""
    image_buffer = io.BytesIO()
    Image.new("RGB", (64, 32), (80, 100, 130)).save(image_buffer, format=image_format)
    image_bytes = image_buffer.getvalue()
    truncated_image_bytes = (
        image_bytes[:-2] if image_format == "JPEG" else image_bytes[:45]
    )
    with Image.open(io.BytesIO(truncated_image_bytes)) as image:
        if image_format == "JPEG":
            image.verify()
        else:
            pass
    reference = base64.b64encode(image_bytes).decode("ascii")
    references = [reference, reference]
    references[invalid_reference_index] = base64.b64encode(
        truncated_image_bytes
    ).decode("ascii")
    invalid = StagePayload(
        "invalid",
        OmniRequest(
            {"task": "image_edit", "prompt": "edit", "image_b64": references},
            {"width": 64, "height": 32, "size_explicit": True, "do_resize": False},
        ),
        None,
    )
    valid = StagePayload(
        "valid",
        OmniRequest(
            {"task": "image_edit", "prompt": "edit", "image_b64": reference},
            invalid.request.params,
        ),
        None,
    )
    model = prefix_model
    it2i_generate_mock.return_value = torch.zeros(1, 3, 32, 64)

    def compute(payload: StagePayload) -> StagePayload:
        return generation_module.generate_image(payload, model, tokenizer, cache_dit)

    def validate(payload: StagePayload) -> None:
        stages_module.validate_generation_request(payload, model, cache_dit, tokenizer)

    scheduler = (
        SenseNovaTensorParallelScheduler(
            compute,
            report_cancellation=True,
            shutdown_join_timeout_s=5.0,
            validate_request=validate,
        )
        if cooperative_cancellation
        else TensorParallelScheduler(
            compute, shutdown_join_timeout_s=5.0, validate_request=validate
        )
    )
    fanout = Mock()
    leader = Stage(
        name="generate",
        role="leader",
        get_next=lambda request_id, output: None,
        gpu_id=None,
        endpoints={},
        control_plane=RecordingStageControlPlane(),
        relay=FakeRelay(),
        scheduler=scheduler,
        tp_fanout=fanout,
    )
    leader.send_failure = AsyncMock()

    await leader.execute(invalid)
    leader.send_failure.assert_awaited_once_with(
        "invalid",
        "Invalid UMM request: SenseNova-U1 received an invalid reference image",
    )
    assert (
        generation_error_status_code(
            RuntimeError(leader.send_failure.call_args.args[1])
        )
        == 400
    )
    fanout.fanout_work.assert_not_called()
    assert scheduler.inbox.empty()
    it2i_generate_mock.assert_not_called()

    await leader.execute(valid)
    fanout.fanout_work.assert_called_once_with(valid)
    scheduler.run_single(scheduler.inbox.get_nowait(), asyncio.get_running_loop())
    result = scheduler.outbox.get_nowait()
    assert result.type == "result"
    assert result.request_id == "valid"
    assert scheduler.compute_error is None
    it2i_generate_mock.assert_called_once()
    with Image.open(
        io.BytesIO(base64.b64decode(result.data.data["image_b64"]))
    ) as image:
        assert image.size == (64, 32)


def test_invalid_edit_seed_is_rejected_before_tp_execution(
    cache_dit: SenseNovaCacheDit,
    it2i_generate_mock: Mock,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    """An invalid seed must fail validation before TP ranks execute model work."""
    reference = io.BytesIO()
    Image.new("RGB", (32, 32)).save(reference, format="PNG")
    payload = StagePayload(
        "invalid",
        OmniRequest(
            {
                "task": "image_edit",
                "prompt": "edit",
                "image_b64": base64.b64encode(reference.getvalue()).decode("ascii"),
            },
            {"seed": 2**64, "do_resize": False},
        ),
        None,
    )
    model = Mock()
    with pytest.raises(ValueError, match="seed"):
        stages_module.validate_generation_request(payload, model, cache_dit, tokenizer)
    with pytest.raises(ValueError, match="seed"):
        generation_module.generate_image(payload, model, tokenizer, cache_dit)
    it2i_generate_mock.assert_not_called()


@pytest.mark.parametrize("failure", [None, "later_reference", "aspect_ratio"])
def test_reference_images_are_closed_after_validation(
    failure: Literal["later_reference", "aspect_ratio"] | None,
) -> None:
    """Validation must close every decoded reference on success or failure."""
    image_buffer = io.BytesIO()
    reference_size_pixels = (32, 7000) if failure == "aspect_ratio" else (64, 32)
    Image.new("RGB", reference_size_pixels).save(image_buffer, format="PNG")
    encoded_reference = base64.b64encode(image_buffer.getvalue()).decode("ascii")
    references = [
        encoded_reference,
        "invalid" if failure == "later_reference" else encoded_reference,
    ]
    options = SenseNovaU1ImageEditSampling.from_params(
        {"width": 96, "height": 64, "input_max_pixels": 262144}
    )
    with patch.object(
        Image.Image, "close", autospec=True, side_effect=Image.Image.close
    ) as close_image:
        if failure is None:
            assert resolve_reference_output_size(references, options) == (96, 32)
        else:
            error_message = (
                "invalid reference image"
                if failure == "later_reference"
                else "aspect ratio"
            )
            with pytest.raises(ValueError, match=error_message):
                resolve_reference_output_size(references, options)
    assert close_image.call_count == (1 if failure == "later_reference" else 2)
    assert all(call.args[0].fp is None for call in close_image.call_args_list)


def test_reference_pixel_decode_failure_closes_open_images() -> None:
    """A truncated reference must close both earlier images and its own header."""
    image_buffer = io.BytesIO()
    Image.new("RGB", (64, 32)).save(image_buffer, format="PNG")
    encoded_reference = base64.b64encode(image_buffer.getvalue()).decode("ascii")
    truncated_reference = base64.b64encode(image_buffer.getvalue()[:45]).decode("ascii")
    with patch.object(
        Image.Image, "close", autospec=True, side_effect=Image.Image.close
    ) as close_image:
        with pytest.raises(ValueError, match="invalid reference image"):
            decode_reference_images([encoded_reference, truncated_reference])
    assert close_image.call_count == 2
    assert all(call.args[0].fp is None for call in close_image.call_args_list)


@pytest.mark.parametrize("references", [None, [], "", "invalid", ["invalid"]])
def test_reference_parsing_rejects_invalid_inputs(
    references: str | list[str] | None,
) -> None:
    """Validation and generation must both reject malformed references."""
    options = SenseNovaU1ImageEditSampling.from_params({})
    with pytest.raises(ValueError):
        decode_reference_images(references)
    with pytest.raises(ValueError):
        resolve_reference_output_size(references, options)


def test_image_edits_stay_single_item_without_decoding_cost() -> None:
    """I2I costs must not decode references that cannot affect batch admission."""
    scheduler = SimpleScheduler(
        Mock(),
        batch_compute_fn=Mock(),
        max_batch_size=2,
        max_batch_cost=1,
        batch_key_fn=stages_module.image_generation_batch_key,
        request_cost_fn=stages_module.image_generation_request_cost,
    )
    messages = [
        IncomingMessage(
            request_id=request_id,
            type="new_request",
            data=StagePayload(
                request_id,
                OmniRequest(
                    {"task": "image_edit", "prompt": "edit", "image_b64": "unused"},
                    {"width": 96, "height": 64},
                ),
                None,
            ),
        )
        for request_id in ("first", "second")
    ]
    scheduler.inbox.put(messages[1])
    with patch.object(Image, "open", wraps=Image.open) as open_image:
        assert scheduler.collect_batch(messages[0]) == [messages[0]]
        assert scheduler.next_message() is messages[1]
        assert scheduler.collect_batch(messages[1]) == [messages[1]]
    open_image.assert_not_called()


def test_generate_stage_params_layer_onto_t2i_options():
    payload = StagePayload(
        "request",
        OmniRequest(
            "A blue cat",
            {
                "stage_params": {
                    "generate": {
                        "width": 512,
                        "height": 384,
                        "num_inference_steps": 2,
                        "guidance_scale": 3.0,
                        "seed": 7,
                        "n": 1,
                    }
                }
            },
        ),
        None,
    )
    options = generation_module.text_to_image_options(payload)
    assert (options.width, options.height) == (512, 384)
    assert options.num_inference_steps == 2
    assert options.guidance_scale == 3.0
    assert options.seed == 7


@pytest.mark.parametrize("input_kind", ["text", "batch", "edit"])
@pytest.mark.parametrize("limit_source", ["tokenizer", "model"])
def test_overlong_prefix_is_rejected_before_device_allocation(
    prefix_model: Mock,
    prefix_tokenizer: PreTrainedTokenizerFast,
    input_kind: Literal["text", "batch", "edit"],
    limit_source: Literal["tokenizer", "model"],
) -> None:
    """Overlong prompts allocated quadratic attention masks instead of returning 400."""
    if limit_source == "model":
        prefix_model.config.llm_config.max_position_embeddings = 129
        prefix_tokenizer.model_max_length = 16384
    else:
        pass
    prefix_model.device = torch.device("meta")
    query = "cat " * 129
    with pytest.raises(UMMRequestError, match="maximum is 128 tokens") as raised:
        if input_kind == "edit":
            build_it2i_inputs(prefix_model, prefix_tokenizer, query)
        else:
            build_t2i_text_inputs(
                prefix_model,
                prefix_tokenizer,
                ["cat", query] if input_kind == "batch" else query,
            )
    assert generation_error_status_code(RuntimeError(str(raised.value))) == 400
    prefix_model.language_model.get_input_embeddings.assert_not_called()

    prefix_model.device = torch.device("cpu")
    token_ids, _, _, _, token_lengths = build_t2i_text_inputs(
        prefix_model, prefix_tokenizer, "cat " * 128
    )
    assert token_ids.shape == (1, 128)
    assert token_lengths.tolist() == [128]


@pytest.mark.parametrize("validation_path", ["preflight", "execution"])
def test_truncated_jpeg_is_a_bad_request(
    prefix_model: Mock,
    prefix_tokenizer: PreTrainedTokenizerFast,
    cache_dit: SenseNovaCacheDit,
    validation_path: Literal["preflight", "execution"],
) -> None:
    """A JPEG passing header verification returned 500 when pixel decoding failed."""
    image_buffer = io.BytesIO()
    Image.new("RGB", (64, 32), (80, 100, 130)).save(image_buffer, format="JPEG")
    truncated = image_buffer.getvalue()[:-2]
    with Image.open(io.BytesIO(truncated)) as image:
        image.verify()
    payload = StagePayload(
        "invalid",
        OmniRequest(
            {
                "task": "image_edit",
                "prompt": "cat",
                "image_b64": base64.b64encode(truncated).decode("ascii"),
            },
            {"do_resize": False},
        ),
        None,
    )
    with pytest.raises(UMMRequestError, match="invalid reference image") as raised:
        if validation_path == "preflight":
            stages_module.validate_generation_request(
                payload, prefix_model, cache_dit, prefix_tokenizer
            )
        else:
            generation_module.generate_image(
                payload, prefix_model, prefix_tokenizer, cache_dit
            )
    assert generation_error_status_code(RuntimeError(str(raised.value))) == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("request_kind", ["text", "edit_text", "edit_references"])
@pytest.mark.parametrize("cooperative_cancellation", [False, True])
async def test_overlong_prompt_never_fans_out_to_tp_workers(
    prefix_model: Mock,
    prefix_tokenizer: PreTrainedTokenizerFast,
    cache_dit: SenseNovaCacheDit,
    request_kind: Literal["text", "edit_text", "edit_references"],
    cooperative_cancellation: bool,
) -> None:
    """Unbounded text and expanded image tokens reached TP compute and stopped its workers."""
    parameters = {"do_resize": False, "width": 32, "height": 32}
    prefix_tokenizer.model_max_length = 512
    prompt = "cat " * 512
    if request_kind == "text":
        inputs = prompt
    else:
        image_buffer = io.BytesIO()
        size = (1024, 1024) if request_kind == "edit_references" else (32, 32)
        Image.new("RGB", size).save(image_buffer, format="PNG")
        inputs = {
            "task": "image_edit",
            "prompt": "cat" if request_kind == "edit_references" else prompt,
            "image_b64": base64.b64encode(image_buffer.getvalue()).decode("ascii"),
        }
    invalid = StagePayload("invalid", OmniRequest(inputs, parameters), None)
    valid = StagePayload("valid", OmniRequest("cat", {"width": 32, "height": 32}), None)

    def compute(payload: StagePayload) -> StagePayload:
        return payload

    def validate(payload: StagePayload) -> None:
        stages_module.validate_generation_request(
            payload, prefix_model, cache_dit, prefix_tokenizer
        )

    scheduler = (
        SenseNovaTensorParallelScheduler(
            compute,
            report_cancellation=True,
            validate_request=validate,
            shutdown_join_timeout_s=5.0,
        )
        if cooperative_cancellation
        else TensorParallelScheduler(
            compute, validate_request=validate, shutdown_join_timeout_s=5.0
        )
    )
    fanout = Mock()
    leader = Stage(
        name="generate",
        role="leader",
        get_next=lambda request_id, output: None,
        gpu_id=None,
        endpoints={},
        control_plane=RecordingStageControlPlane(),
        relay=FakeRelay(),
        scheduler=scheduler,
        tp_fanout=fanout,
    )
    leader.send_failure = AsyncMock()
    await leader.execute(invalid)
    leader.send_failure.assert_awaited_once()
    error = RuntimeError(leader.send_failure.call_args.args[1])
    assert generation_error_status_code(error) == 400
    assert "maximum is 512 tokens" in str(error)
    fanout.fanout_work.assert_not_called()
    assert scheduler.inbox.empty()
    assert scheduler.compute_error is None

    await leader.execute(valid)
    fanout.fanout_work.assert_called_once_with(valid)
    scheduler.run_single(scheduler.inbox.get_nowait(), asyncio.get_running_loop())
    result = scheduler.outbox.get_nowait()
    assert result.type == "result"
    assert result.request_id == "valid"
    assert scheduler.compute_error is None


@pytest.mark.parametrize("invalid_indices", [(0,), (1,), (0, 2), (0, 1, 2)])
def test_batch_rejects_only_overlong_requests(
    monkeypatch: pytest.MonkeyPatch,
    prefix_model: Mock,
    prefix_tokenizer: PreTrainedTokenizerFast,
    t2i_generate_mock: Mock,
    invalid_indices: tuple[int, ...],
) -> None:
    """An overlong prompt made compatible valid requests fail with the same 400."""
    prefix_tokenizer.model_max_length = 512
    prefix_model.eval.return_value = prefix_model
    prefix_model.to.return_value = prefix_model
    monkeypatch.setattr(stages_module, "register", lambda: None)
    monkeypatch.setattr(
        stages_module.AutoTokenizer,
        "from_pretrained",
        lambda model_path: prefix_tokenizer,
    )
    monkeypatch.setattr(
        stages_module.AutoModel,
        "from_pretrained",
        lambda model_path, torch_dtype: prefix_model,
    )
    monkeypatch.setattr(
        stages_module,
        "resolve_concrete_device",
        lambda device, gpu_id: torch.device("cpu"),
    )
    scheduler = stages_module.create_generation_executor(
        "unused",
        device="cpu",
        dtype="bfloat16",
        cooperative_cancellation=True,
        max_batch_size=3,
        enable_cache_dit=False,
        dit_layerwise_offload=False,
        dit_offload_prefetch_size=1,
        dit_layerwise_resident_layers=0,
        dit_layerwise_residency_policy="leading",
        pin_cpu_memory=False,
    )
    scheduler.running = True
    payloads = [
        StagePayload(
            str(index),
            OmniRequest(
                "cat " * 512 if index in invalid_indices else "cat",
                {"width": 32, "height": 32, "seed": index},
            ),
            None,
        )
        for index in range(3)
    ]
    valid_indices = [index for index in range(3) if index not in invalid_indices]
    t2i_generate_mock.return_value = torch.zeros(len(valid_indices), 3, 32, 32)
    loop = asyncio.new_event_loop()
    try:
        scheduler.run_batch(
            [
                IncomingMessage(
                    request_id=payload.request_id, type="new_request", data=payload
                )
                for payload in payloads
            ],
            loop,
        )
    finally:
        loop.close()
    messages = [scheduler.outbox.get_nowait() for _ in payloads]
    assert [message.request_id for message in messages] == ["0", "1", "2"]
    for index, message in enumerate(messages):
        if index in invalid_indices:
            assert message.type == "error"
            assert generation_error_status_code(RuntimeError(str(message.data))) == 400
            assert "maximum is 512 tokens" in str(message.data)
        else:
            assert message.type == "result"
            with Image.open(
                io.BytesIO(base64.b64decode(message.data.data["image_b64"]))
            ) as image:
                image.load()
                assert image.size == (32, 32)
    if valid_indices:
        t2i_generate_mock.assert_called_once()
        assert t2i_generate_mock.call_args.kwargs["seeds"] == valid_indices
        assert t2i_generate_mock.call_args.args[2] == ["cat"] * len(valid_indices)
    else:
        t2i_generate_mock.assert_not_called()
    assert scheduler.outbox.empty()
