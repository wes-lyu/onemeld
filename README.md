# OneMeld

OneMeld serves several specialist models behind one API and one model name. A request can ask for images and a comparison in the same turn. A planner model turns the request into steps, and OneMeld runs each step on the model that offers that capability, passing the actual results between them.

The name combines "one" with "meld", which means to fuse into a whole. Each model keeps its own strengths, and clients see a single service.

OneMeld hosts every model itself through a serving framework it imports as a library. This release supports [SGLang](https://github.com/sgl-project/sglang) and [SGLang-Omni](https://github.com/sgl-project/sglang-omni). Each model runs in its own worker process, which can use its own Python environment, so models served by different frameworks can share one deployment.

Send text, inline images or short speech instructions to `/v1/chat/completions`. Each request gets a plan. The response contains the selected text and images, with optional streaming through server-sent events. Use `/v1/models` to discover the model name and `/health` to check every backend.

## Run the first example

The [example profile](examples/onemeld/serve.yaml) runs three models with SGLang on one 48 GB Linux CUDA GPU, and has been run on an RTX 4090: Qwen3-VL-4B-Instruct plans and answers text and visual tasks, SenseNova-U1.5-8B-MoT generates images, and Qwen3-ASR-0.6B transcribes spoken instructions. SenseNova keeps its DiT on the GPU and offloads its encoders and VAE to CPU on its own, which leaves no room for a separate planner model on that card. To fit, the vision model loads with FP8 weights and the speech model transcribes 30 seconds of audio at startup to reserve its memory before SenseNova starts; the profile's comments list the measured memory per process. With a second GPU, add a dedicated planner backend there and point `composition.planner_stage` at it.

From the repository root, install into a fresh Python 3.12 environment:

```bash
uv venv --python 3.12
uv pip install --prerelease=allow -e ".[sglang]"
.venv/bin/onemeld serve --config examples/onemeld/serve.yaml --host 127.0.0.1 --port 8897
```

Each backend's `model` is a Hugging Face repository ID. To use weights you already have, set it to the checkpoint directory.

Send the [example request](examples/onemeld/compare-cats.json):

```bash
curl -fsS http://127.0.0.1:8897/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @examples/onemeld/compare-cats.json \
  -o response.json
jq -r '.choices[0].message.segments[].kind' response.json
```

The request asks for two cat photos and a recommendation based on clarity, subject completeness and composition. A successful response contains two image segments and a text segment; on the RTX 4090 it took about 23 seconds. The server log includes the accepted plan and each dispatch under `UMM composition trace`, keyed by the response segments' `execution_id`.

## Reproduce the validation cases

[examples/onemeld/cases](examples/onemeld/cases/cases.json) holds nine requests that exercise composition: writing a design before generating (F1), generating from a reference image's style (F2), checking a generated image (F3), scoring two generated images (F4), a Mandarin voice-only request for a designed poster (F5), describing an image without generating (F6), F4's request spoken in Mandarin (F7), and F5's and F7's requests spoken in English (F8, F9). The reference image and the recordings are in the same directory.

Start the server with per-request traces, then run the cases on the same host:

```bash
onemeld serve --config examples/onemeld/serve.yaml --port 8897 \
  --trace-dir /tmp/onemeld-trace
python scripts/run_cases.py --url http://127.0.0.1:8897 \
  --trace-dir /tmp/onemeld-trace --output results
```

The script needs `httpx`, which the `test` extra installs. Each case directory under `results` holds the request, the response, every returned text and image, the request's plan and trace, and `report.json`. A case passes when its plan is valid on the first attempt, every hash and reference in its trace is intact, its steps run as planned and feed each other, and it returns the expected outputs, including the text the request asked for, such as the scores and recommendation, from the step that produced it. For F4, F7 and F9 it also reads every photo's score on each criterion and the recommended photo from the answer, and requires a score for every photo on every criterion and a recommendation of one of them; `report.json` records what it read under `scored_answer`. Whether an image or answer is good beyond that still needs a person to look. Add `--repeat 3` to run every case three times on the same server, with each round under `results/round-N`.

## Mix frameworks

[sglang_omni.yaml](examples/onemeld/sglang_omni.yaml) is the same profile with speech transcription on an SGLang-Omni pipeline instead of SGLang. Unlike serve.yaml, it has not been run on a GPU yet. A backend whose framework needs its own environment names that environment's interpreter:

```yaml
backends:
  asr:
    framework: sglang_omni
    task: transcription
    model: Qwen/Qwen3-ASR-0.6B
    python: /opt/envs/sglang-omni/bin/python
```

The orchestrator itself needs only the core install. Install each framework extra in the environment its backends run in.

## Code map

- [Composition reference](docs/developer_reference/umm_composition.md) describes capabilities, plan validation, backends and how to add a framework.
- [Orchestrator](onemeld/orchestrator.py) runs a request's steps on its backends and publishes the results in plan order.
- [Composition](onemeld/composition/) holds the planner prompt, plan validation and the per-protocol step requests.
- [Backends](onemeld/backends/) hold the worker process protocol and the SGLang and SGLang-Omni adapters.

OneMeld inherits the [Apache 2.0 license](LICENSE) and preserves the upstream project history and notices.
