# OneMeld

OneMeld serves several specialist models behind one API and one model name. A request can ask for images and a comparison in the same turn. A planner model turns the request into steps, and OneMeld runs each step on the model that offers that capability, passing the actual results between them.

The name combines "one" with "meld", which means to fuse into a whole. Each model keeps its own strengths, and clients see a single service.

OneMeld hosts every model itself through a serving framework it imports as a library. This release supports [SGLang](https://github.com/sgl-project/sglang) and [SGLang-Omni](https://github.com/sgl-project/sglang-omni). Each model runs in its own worker process, which can use its own Python environment, so models served by different frameworks can share one deployment.

Send text, inline images or short speech instructions to `/v1/chat/completions`. Each request gets a plan. The response contains the selected text and images, with optional streaming through server-sent events. Use `/v1/models` to discover the model name and `/health` to check every backend.

## Run the first example

The [example profile](examples/onemeld/serve.yaml) runs four models with SGLang on one Linux CUDA GPU: Qwen3.5-4B plans, Qwen3-VL-4B-Instruct answers text and visual tasks, SenseNova-U1.5-8B-MoT generates images, and Qwen3-ASR-0.6B transcribes spoken instructions. SenseNova's pipeline comes from SGLang-Omni. The memory split in the profile has not been checked on hardware yet. If the image backend runs out of memory, give it a second GPU with `gpu: 1`.

From the repository root, install into a fresh Python 3.12 environment:

```bash
uv venv --python 3.12
uv pip install --prerelease=allow -e ".[sglang-omni]"
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

The request asks for two cat photos and a recommendation based on clarity, subject completeness and composition. A successful response contains two image segments and a text segment. The server log includes the accepted plan and each dispatch under `UMM composition trace`, keyed by the response segments' `execution_id`.

## Mix frameworks

[sglang_omni.yaml](examples/onemeld/sglang_omni.yaml) is the same profile with speech transcription on an SGLang-Omni pipeline instead of SGLang. A backend whose framework needs its own environment names that environment's interpreter:

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
