# Dynamic Qwen UMM GPU validation

On 2026-10-27, three natural-language tasks completed through the same `POST /v1/chat/completions` endpoint. Qwen-VL generated each execution plan on the GPU, and the existing UMM controller executed its actions across independent Qwen-VL and Qwen-Image model instances. All client controls were identical apart from the user message. No mode, action list, or plan was supplied by the client.

## Environment

- Branch: `wes/qwen-dynamic-planning`; base `53994012c9b050951db8a0452356a6c6f2e9c6ea` plus the uncommitted Qwen demo files.
- GPU: NVIDIA GeForce RTX 4090, 49,140 MiB. Conda environment: `sgl`; Torch 2.14.1+cu130, Transformers 5.19.0. The existing native SGLang source and dependencies were reused.
- Models: `/model/ModelScope/Qwen/Qwen2.5-VL-3B-Instruct` and `/model/ModelScope/Qwen/Qwen-Image-2512`. They have separate model weights and contexts; Qwen-Image owns its own condition encoder.
- The existing native chat and generation stage factories ran in `/workspace/qwen-umm-demo`. The installed Omni checkout was preserved. Shared UMM, native stage implementations, and HTTP routes were not modified.
- Both stages used GPU 0. SRT static memory fraction was 0.22 with a 4096-token context, Triton attention, and CUDA graphs disabled. Generation used native transformer layerwise offload, text encoder CPU offload, and VAE CPU offload.
- Images: 512×512, 20 steps, guidance 4.0, seeds starting at 42. Planner limit: 1536 tokens; response limit: 1024 tokens. Text sampling was greedy.

## Requests and actual execution

| User task | Model-selected actions after planning | Returned segments | HTTP | Client elapsed |
| --- | --- | --- | --- | --- |
| Explain cat purring in one sentence; no images | `respond` | text | 200 | 3.659 s |
| Generate one kitten photograph; no extra text | `generate_image` | image | 200 | 117.707 s |
| Generate two kitten photographs, inspect and score both, choose a winner | `generate_image`, `generate_image`, `respond` | image, image, text | 200 | 174.014 s |

Every request first ran a separate `plan` phase through Qwen-VL. The image-only request included native image warmup; the server recorded approximately 41 seconds for warmup. These single-run smoke timings include CPU offload and are not latency or throughput benchmarks.

The comparison plan named its generated images `generate_image_1` and `generate_image_2`. Its final `respond` action explicitly referenced both IDs and requested scores and a visible comparison. The accepted plan was model output, recorded before the action dispatches. The executor used each plan's actual instruction as the native generation prompt and resolved image references from the preceding results.

Qwen-VL scored the comparison's first image 9/10 and the second 8/10, preferring the first for perceived fur and window detail. The output was in Chinese. This verifies that the comparison received actual generated pixels; the three requests do not establish planner accuracy or calibrated image-quality scoring. The purring response also illustrates that successful routing does not establish factual answer accuracy.

## Execution evidence

The server's `Qwen UMM trace` JSON events record accepted plans, actual dispatches, and generated image hashes, keyed by the execution IDs returned in HTTP segments. The response artifacts and trace excerpts are retained separately; plans were not reconstructed from the requests.

| Output | SHA-256 |
| --- | --- |
| Image-only request | `697069d75714177694823815855efc16c5d65d8055e99317c54d3e91bdcd3ca9` |
| Comparison image 1 | `0048a239df6aaa615beb34df3c8c949f3c55c74017befb15d0ad841f671ad160` |
| Comparison image 2 | `0be82beb9aa249b4aa7a2c9d35bb1fc83bac3dc0ca63194ff3328e4026c6f667` |

Raw responses, request bodies, decoded images, and reports are saved under `/workspace/qwen-umm-demo/results/dynamic/{text,image,compare}`. The validation script saves `summary.json` after checking all three response shapes and distinct image hashes. The local delivery artifacts additionally match each accepted plan to its real dispatch events and image handoff hashes.

## Compatibility and reproduction

The 16 adapter tests passed locally. The new adapter tests, existing UMM controller tests, and existing UMM HTTP contract tests passed together in the complete GPU environment: 62 tests. Repository pre-commit checks passed. The shared UMM controller's file hash matches the base checkout; this demo extends behavior through its adapter and configuration.

Start the service with [README.md](README.md), then submit each complete task through `python -m examples.qwen_umm.compare`. The GPU validation used `dynamic_smoke.py` in the isolated workspace, invoking that client once for each task. Neither the client nor the validation script calls the image or vision stage directly.

This version generates one bounded plan before execution and runs its actions sequentially. Replanning after observations, arbitrary stage discovery, image editing, and cross-model tensor/KV transfer are not implemented. Existing UMM behavior remains available to the other model adapters.
