# SPDX-License-Identifier: Apache-2.0
"""Batched attention, prefix masking, and per-request random state."""

from types import SimpleNamespace

import pytest
import torch
from transformers.cache_utils import DynamicCache

from sglang_omni.models.sensenova_u1.neo_unify import (
    attention_backend as attention_backend_module,
)
from sglang_omni.models.sensenova_u1.neo_unify.attention_backend import (
    create_block_causal_mask,
    position_ids_from_indexes,
    sdpa_attn_func,
)
from sglang_omni.models.sensenova_u1.neo_unify.configuration_neo_chat import (
    NEOLLMConfig,
)
from sglang_omni.models.sensenova_u1.neo_unify.flash_cache import prepare_flash_kv_cache
from sglang_omni.models.sensenova_u1.neo_unify.generation_inputs import (
    build_t2i_image_indexes,
    build_t2i_text_inputs,
)
from sglang_omni.models.sensenova_u1.neo_unify.generation_math import randn_with_seed
from sglang_omni.models.sensenova_u1.neo_unify.qwen_attention import Qwen3Attention
from sglang_omni.platforms import current_platform as model_platform


@pytest.mark.parametrize("prefix_length", [2, 5, 9])
def test_full_length_prefixes_keep_the_unmasked_attention_path(prefix_length):
    batch_size = 2
    cache = DynamicCache()
    cache.update(
        torch.zeros(batch_size, 2, prefix_length, 4),
        torch.zeros(batch_size, 2, prefix_length, 4),
        layer_idx=0,
    )

    prepare_flash_kv_cache(
        cache,
        current_len=3,
        batch_size=batch_size,
        prefix_lengths=torch.full((batch_size,), prefix_length),
    )

    layer = cache.layers[0]
    assert layer.flash_cache_layout == "BSND"
    assert layer.flash_key_padding_mask is None
    assert layer.flash_k_cache.shape == (batch_size, prefix_length + 3, 2, 4)
    assert layer.flash_v_cache.shape == (batch_size, prefix_length + 3, 2, 4)


def force_generator_fallback(monkeypatch, device_type):
    original_generator = torch.Generator

    def unsupported_device_generator(device="cpu"):
        if torch.device(device).type == device_type:
            raise RuntimeError(f"Generator is unsupported on {device_type}")
        return original_generator(device)

    monkeypatch.setattr(torch, "Generator", unsupported_device_generator)


def test_randn_fallback_preserves_cpu_rng(monkeypatch):
    force_generator_fallback(monkeypatch, "cpu")
    rng_state = torch.get_rng_state().clone()

    first = randn_with_seed((2, 3), device="cpu", dtype=torch.float32, seed=17)
    second = randn_with_seed((2, 3), device="cpu", dtype=torch.float32, seed=17)

    assert torch.equal(first, second)
    assert torch.equal(torch.get_rng_state(), rng_state)


def test_randn_supports_per_sample_seeds():
    actual = randn_with_seed(
        (2, 3, 4), device=torch.device("cpu"), dtype=torch.float32, seed=[7, 19]
    )
    expected = torch.cat(
        [
            randn_with_seed(
                (1, 3, 4),
                device=torch.device("cpu"),
                dtype=torch.float32,
                seed=seed,
            )
            for seed in (7, 19)
        ]
    )

    assert torch.equal(actual, expected)


class FakeTokenizer:
    model_max_length = 128
    pad_token_id = None
    eos_token_id = 2

    def __call__(self, text, return_tensors):
        del return_tensors
        token_count = len(text.split()) + 1
        return {"input_ids": torch.arange(1, token_count + 1).unsqueeze(0)}


def test_builds_padded_batched_text_inputs():
    model = SimpleNamespace(
        device=torch.device("cpu"),
        config=SimpleNamespace(llm_config=SimpleNamespace(max_position_embeddings=129)),
    )

    input_ids, indexes, attention_mask, valid_mask, prefix_lengths = (
        build_t2i_text_inputs(model, FakeTokenizer(), ["short", "a much longer prompt"])
    )

    assert input_ids.shape == (2, 5)
    assert indexes.shape == (2, 3, 5)
    assert prefix_lengths.tolist() == [2, 5]
    assert valid_mask.tolist() == [
        [True, True, False, False, False],
        [True, True, True, True, True],
    ]
    mask = attention_mask["full_attention"]
    assert mask.shape == (2, 1, 5, 5)
    assert torch.isneginf(mask[0, :, :, 2:]).all()
    assert torch.isfinite(mask[0, :, :, :2]).any()


def test_position_indexes_support_batched_inputs():
    indexes = torch.tensor(
        [
            [[0, 1], [0, 0], [0, 0]],
            [[4, 4], [0, 1], [0, 0]],
        ]
    )

    assert torch.equal(position_ids_from_indexes(indexes, 0), indexes[:, 0])
    assert torch.equal(
        position_ids_from_indexes(indexes[0], 1), indexes[0, 1].unsqueeze(0)
    )


def test_singleton_text_matches_valid_batched_tokens():
    model = SimpleNamespace(
        device=torch.device("cpu"),
        config=SimpleNamespace(llm_config=SimpleNamespace(max_position_embeddings=129)),
    )
    tokenizer = FakeTokenizer()
    batched = build_t2i_text_inputs(model, tokenizer, ["short", "a much longer prompt"])
    for index, prompt in enumerate(["short", "a much longer prompt"]):
        single = build_t2i_text_inputs(model, tokenizer, prompt)
        length = single[0].shape[1]
        assert torch.equal(batched[0][index, :length], single[0][0])
        assert torch.equal(batched[1][index, :, :length], single[1])
        assert torch.equal(
            batched[2]["full_attention"][index, :, :length, :length],
            single[2]["full_attention"][0],
        )


def test_block_causal_mask_rejects_padded_keys():
    indexes = torch.tensor([[0, 1, 2], [0, 1, 2]])
    valid = torch.tensor([[True, True, False], [True, True, True]])

    mask = create_block_causal_mask(indexes, valid)

    assert mask.shape == (2, 1, 3, 3)
    assert torch.isneginf(mask[0, :, :, 2]).all()
    assert mask[1, 0, 2, 2] == 0


def test_builds_per_sample_image_indexes():
    indexes = build_t2i_image_indexes(
        token_h=2,
        token_w=2,
        text_len=torch.tensor([2, 5]),
        device=torch.device("cpu"),
    )

    assert indexes.shape == (2, 3, 4)
    assert indexes[:, 0].tolist() == [[2, 2, 2, 2], [5, 5, 5, 5]]
    assert indexes[:, 1].tolist() == [[0, 0, 1, 1], [0, 0, 1, 1]]
    assert indexes[:, 2].tolist() == [[0, 1, 0, 1], [0, 1, 0, 1]]


def test_sdpa_masks_padded_prefix_keys():
    q = torch.tensor([[[[1.0, 0.0]]]])
    k = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]]])
    v = torch.tensor([[[[2.0, 0.0]], [[0.0, 4.0]], [[100.0, 100.0]]]])
    key_mask = torch.tensor([[[[True, True, False]]]])

    actual = sdpa_attn_func(q, k, v, attention_mask=key_mask)
    expected = sdpa_attn_func(q, k[:, :2], v[:, :2])

    torch.testing.assert_close(actual, expected)


@pytest.mark.accelerator
def test_randn_fallback_preserves_accelerator_rng(monkeypatch):
    device_type = model_platform.device_type
    if not device_type or device_type == "cpu":
        pytest.skip("No accelerator is available")

    device = torch.device(device_type, 0)
    device_module = torch.get_device_module(device)
    if not device_module.is_available():
        pytest.skip(f"{device_type} is not available")

    force_generator_fallback(monkeypatch, device_type)
    cpu_rng_state = torch.get_rng_state().clone()
    device_rng_state = device_module.get_rng_state(device).clone()

    first = randn_with_seed((2, 3), device=device, dtype=torch.float32, seed=17)
    second = randn_with_seed((2, 3), device=device, dtype=torch.float32, seed=17)

    assert torch.equal(first, second)
    assert torch.equal(torch.get_rng_state(), cpu_rng_state)
    assert torch.equal(device_module.get_rng_state(device), device_rng_state)


def test_batched_gqa_matches_unpadded_singletons():
    generator = torch.Generator().manual_seed(17)
    q = torch.randn(2, 3, 4, 8, generator=generator)
    k = torch.randn(2, 8, 2, 8, generator=generator)
    v = torch.randn(2, 8, 2, 8, generator=generator)
    valid = torch.ones(2, 8, dtype=torch.bool)
    valid[0, 2:5] = False
    attention_mask = valid[:, None, None, :].expand(-1, -1, q.shape[1], -1)

    actual = sdpa_attn_func(q, k, v, attention_mask=attention_mask)

    for index in range(2):
        expected = sdpa_attn_func(
            q[index : index + 1],
            k[index : index + 1, valid[index]],
            v[index : index + 1, valid[index]],
        )
        torch.testing.assert_close(actual[index : index + 1], expected)


@torch.no_grad()
def test_prefix_and_denoise_attention_match_singletons(monkeypatch):
    config = NEOLLMConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
    )
    config._attn_implementation = (  # noqa: leading-underscore -- Transformers attention backend field
        "eager"
    )
    with torch.random.fork_rng():
        torch.manual_seed(23)
        attention = Qwen3Attention(config, layer_idx=0).eval()
    generator = torch.Generator().manual_seed(31)
    text = torch.randn(2, 5, 64, generator=generator)
    image = torch.randn(2, 3, 64, generator=generator)
    attention_batch_sizes: list[int] = []
    sdpa = attention_backend_module.sdpa_attn_func

    def record_sdpa(q, k, v, *args, **kwargs):
        attention_batch_sizes.append(q.shape[0])
        return sdpa(q, k, v, *args, **kwargs)

    monkeypatch.setattr(attention_backend_module, "sdpa_attn_func", record_sdpa)

    def run(prefix, lengths, image_states):
        batch_size, width, _ = prefix.shape
        positions = torch.arange(width).expand(batch_size, -1)
        indexes = torch.stack(
            [positions, torch.zeros_like(positions), torch.zeros_like(positions)],
            dim=1,
        )
        valid = positions < torch.tensor(lengths)[:, None]
        cache = DynamicCache(config=config)
        attention.forward_und(
            prefix, indexes, create_block_causal_mask(positions, valid), cache
        )
        prefix_keys = cache.layers[0].keys.clone()
        prepare_flash_kv_cache(
            cache,
            current_len=3,
            batch_size=batch_size,
            prefix_lengths=torch.tensor(lengths),
        )
        image_indexes = build_t2i_image_indexes(
            1, 3, torch.tensor(lengths), torch.device("cpu")
        )
        outputs = []
        for _ in range(2):
            image_states, _ = attention.forward_gen(
                image_states,
                image_indexes,
                None,
                cache,
                update_cache=False,
            )
            outputs.append(image_states)
        torch.testing.assert_close(cache.layers[0].keys, prefix_keys)
        return prefix_keys, outputs

    keys, batched = run(text, [2, 5], image)
    # The padded batch attends in one call per denoise step, not once per row.
    assert attention_batch_sizes == [2, 2]
    for index, length in enumerate([2, 5]):
        single_keys, single = run(
            text[index : index + 1, :length],
            [length],
            image[index : index + 1],
        )
        torch.testing.assert_close(keys[index : index + 1, :, :length], single_keys)
        for step in range(2):
            torch.testing.assert_close(
                batched[step][index : index + 1],
                single[step],
                atol=1e-5,
                rtol=1e-4,
            )
