# OneMeld

OneMeld serves several specialist models behind one API and one model name. A request can ask for images and a comparison in the same turn. A planner describes the steps, and the serving runtime passes the actual results between the models that execute them.

OneMeld builds on [SGLang-Omni](https://github.com/sgl-project/sglang-omni). It reuses the UMM controller and native stage executors. The planner selects actions and dependencies from a deployment's configured capabilities. Stage bindings determine which model executes each action.

Send text, inline images or short speech instructions to `/v1/chat/completions`. Each request gets a plan. The response contains the selected text and images, with optional streaming through server-sent events. Use `/v1/models` to discover the model name and `/health` to check the worker group.

## Run the first example

The [example profile](examples/onemeld/serve.yaml) targets one Linux CUDA GPU with 48 GB of VRAM and uses host memory for image generation offload. It combines Qwen3.5-4B for planning, Qwen3-VL-4B-Instruct for text and visual tasks, SenseNova-U1.5-8B-MoT with Cache-DiT for image generation, and Qwen3-ASR-0.6B for spoken instructions. Use Python 3.12 and a CUDA environment compatible with the dependencies pinned in [pyproject.toml](pyproject.toml).

From the repository root, install into a fresh environment:

```bash
uv venv --python 3.12
uv pip install --prerelease=allow -e .
.venv/bin/onemeld serve --config examples/onemeld/serve.yaml \
  --host 127.0.0.1 --port 8897 --model-name onemeld
```

The profile uses model repository IDs. If weights are already available locally, set `model_path`, `text_model_path`, `asr_model_path`, and `planner_model_path` in the YAML to their checkpoint directories.

Send the [example request](examples/onemeld/compare-cats.json):

```bash
curl -fsS http://127.0.0.1:8897/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @examples/onemeld/compare-cats.json \
  -o response.json
jq -r '.choices[0].message.segments[].kind' response.json
```

The request asks for two cat photos and a recommendation based on clarity, subject completeness, and composition. A successful response contains two image segments and a text segment. The server log includes the accepted plan and dispatch trace under `UMM composition trace`, keyed by the response segments' `execution_id`. This plan is produced at request time. Its evaluation step consumes the two generated images through `input_refs`.

## Code map

- [UMM composition reference](docs/developer_reference/umm_composition.md) describes capabilities, plan validation, and artifact handoff.
- [Composition executor](sglang_omni/pipeline/umm_composition.py) schedules ready steps and publishes their results.
- [Model bindings](sglang_omni/models/qwen_umm/config.py) configure the four models. The Python namespace remains `sglang_omni` for compatibility with the inherited runtime.

OneMeld inherits the [Apache 2.0 license](LICENSE) and preserves the upstream project history and notices.
