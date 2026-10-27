# Native Qwen UMM GPU smoke test

Tested on 2026-10-27 using the native SGLang chat and image stage factories in this checkout. The understanding model is Qwen2.5-VL-3B-Instruct. The image model is Qwen-Image-2512, with its approximately 20B image transformer and separate Qwen2.5-VL condition encoder. The two stages do not share VLM weights or KV caches.

## Environment

- GPU: NVIDIA GeForce RTX 4090, 49,140 MiB, compute capability 8.9; driver 595.80.
- Host: approximately 94 GiB RAM; native image worker reported an 88 GiB cgroup memory cap.
- Python environment: Conda `sgl`, Python 3.12; Torch 2.14.1+cu130, Transformers 5.19.0, Diffusers 0.37.0, SGLang 0.5.22.
- Native SGLang source: `baa9befa381511cd4d56f83980951e5a45fffc6d`, installed source under `/workspace/sgl-projects/sglang`.
- Omni source: local base `53994012c` plus this uncommitted demo, copied into `/workspace/qwen-umm-demo`. The existing installed Omni checkout was preserved.
- Both stages used GPU 0. Understanding: placement budget 0.25, SRT static fraction 0.22, context 4096, one running request, Triton attention, CUDA graphs disabled. Generation: placement budget 0.70, native transformer layerwise offload, text encoder and VAE CPU offload.
- Image sampling: 512×512, 20 steps, guidance 4.0, seed 42, one output. No quantization was enabled.

## Results

The complete cat comparison was subsequently verified with a single `POST /v1/chat/completions` request using `python -m examples.qwen_umm.compare`. The response was a Chat Completion containing two image segments followed by a text segment; the same text also appeared in `choices[0].message.content`. All three model calls ran inside the server's UMM controller. The client made one HTTP request and saved the returned images and text.

The request used the same hardware and offload configuration, 512×512 images, 20 steps, seeds 42 and 43, and up to 1024 evaluation tokens. The complete request took 205.153 seconds, including native image warmup. These are smoke-test timings, not a benchmark.

| Image | Prompt adherence | Naturalness | Visual quality | Total |
| --- | --- | --- | --- | --- |
| A | 8/10 | 9/10 | 7/10 | 24/30 |
| B | 10/10 | 9/10 | 8/10 | 27/30 |

Qwen-VL supplied the three scores and Chinese explanations. The adapter calculated the totals and selected B. The explanations mentioned fur, eyes, window lighting, background blur, and windowsill texture, but gave only a coarse distinction between the images. This run verifies image delivery and the complete workflow; it does not establish the evaluator's scoring accuracy.

The images had SHA-256 `6cf57c6253773d097fab4b974ee4071a97287f42d9db01911c420ca57078927a` (A) and `ad6ca95c14528c31e31fde64262805ecf200ac6c5a33568766d148a3df2867d2` (B). They were byte-identical to the preceding `/generate` comparison run with the same prompt and seeds. That earlier run took 167.972 seconds after warmup. Its free-form evaluator produced inconsistent totals; the current evaluator uses a JSON schema and the adapter sums its validated component scores.

Raw Chat Completions output, PNGs, and the comparison report are saved under `/workspace/qwen-umm-demo/results/cats-chat`. The demo adapter tests and existing UMM HTTP contract tests passed in the GPU environment: 30 tests. The 12 adapter tests also passed locally. The shared HTTP suite could not collect in the local test environment because native SGLang was absent, so it was run in the complete remote environment instead. Repository pre-commit checks passed. The test service was stopped after verification; the instance was left running.

The earlier single-image and text smoke results are retained below:

| Request | HTTP | Result | Client elapsed time |
| --- | --- | --- | --- |
| Text | 200 | `2 + 3` returned `5` | 0.133 s |
| Text followed by image | 200 | Ordered text/image segments and a valid 512×512 PNG | 115.595 s |
| Image only, same prompt and seed | 200 | PNG byte-identical to the combined request | 69.508 s |
| Image understanding / OCR | 200 | Correctly returned `HELLO 2026` from a generated test card | 2.537 s |
| Explicit scene description followed by image | 200 | Useful scene description followed by a valid 512×512 PNG | 72.808 s |

Both image responses contained 490,734 bytes with SHA-256 `b361c347b2baf3992b622fda8b60f28e9ab7069df8d8e4f2bdf395b03e26a97b`. PNG decoding, dimensions, byte counts, and the returned hashes were checked. Visual inspection confirmed a red panda reading a book beneath cherry blossoms. This compares the combined and image-only modes of the same native stage; it is not a comparison against Diffusers or a separate standalone SGLang server.

The first image request included native warmup; logs recorded about 47 seconds for warmup and about 3.46 seconds per denoising step. These are single-run smoke timings with CPU offload, not throughput or latency benchmarks. A spot GPU memory reading during generation was 30,768 MiB; this was not a peak-memory measurement.

The initial pure scene prompt caused the independent chat model to explain that it could not draw, while the image stage still generated the requested image. With the explicit scene-description prompt now shown in the README, the text became: “A red panda perches on a branch, engrossed in reading an open book under a blooming cherry tree bathed in soft daylight, rendered in a detailed watercolor illustration.” The image response had SHA-256 `dc8ff53a8675c779f00220fe2059ec70c24c197545f0ae59007348ec375f8205`. This demonstrates the semantic limitation of explicit two-model orchestration rather than an autonomous unified decoder.

## Reproduce

The mounted checkpoints used in this run were `/model/ModelScope/Qwen/Qwen2.5-VL-3B-Instruct` and `/model/ModelScope/Qwen/Qwen-Image-2512`. Activate the environment before launch so native JIT compilation can find `ninja`:

```bash
conda activate sgl
cd /workspace/qwen-umm-demo
python -m sglang_omni.cli serve \
  --config examples/qwen_umm/launch.yaml \
  --model-path /model/ModelScope/Qwen/Qwen-Image-2512 \
  --text-model-path /model/ModelScope/Qwen/Qwen2.5-VL-3B-Instruct \
  --host 127.0.0.1 --port 8897
```

Use the requests in [README.md](README.md). The GPU-side `smoke.py` records raw responses, decoded PNGs, the OCR input, and `summary.json` in `/workspace/qwen-umm-demo/results`. `describe.py` tests the clarified scene-description prompt. Neither script invokes an external image API.

An initial launch was rejected because colocated stage budgets were undeclared; the launch profile now declares them. A subsequent image request failed because the launch command selected the environment's Python without activating its executable PATH, hiding the installed `ninja`. Launching through `conda run --no-capture-output -n sgl` resolved that failure. No model implementation or installed dependency was patched to obtain the successful requests.
