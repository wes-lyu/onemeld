# OneMeld composition

A profile binds capabilities to backends. The planner picks actions from those capabilities and returns a complete plan before any step runs. The orchestrator validates the plan, starts steps whose inputs are ready, and publishes selected results in plan order.

## Request flow

1. `POST /v1/chat/completions` parses the conversation into text and named input artifacts. When the user sends a spoken instruction, the ASR backend transcribes it before planning, and the transcript becomes part of the planner input.
2. The planner backend returns JSON containing `steps`. A step specifies `id`, `action`, `instruction`, `input_refs` and `publish`. References must name client inputs or earlier steps and match the receiving capability's input kinds.
3. The orchestrator stores each completed result under its step ID. The next backend receives the actual text or media its step references. Independent steps can run on free backends within `max_concurrent_steps`.
4. The API returns selected results in `choices[0].message.segments`. Text is also available in `message.content`. Streaming requests receive the same segments as SSE deltas.

A malformed plan fails validation before anything runs. The example profile allows one repair attempt before reporting a planning error. The orchestrator bounds the active request count, retained bytes, number of dispatches and request duration. A client disconnect cancels the request's running steps.

## Backends

Each backend is one model in one worker process. Its `framework` and `task` select an adapter class, and the capability's protocol must match that task: `native_chat` needs a `chat` backend, `asr` a `transcription` backend and `image_base64` an `image` backend.

| Backend | Framework | Role in the example |
| --- | --- | --- |
| `understanding` | SGLang | Qwen3-VL-4B-Instruct produces the upfront plan, answers text tasks and compares referenced images. |
| `asr` | SGLang | Qwen3-ASR-0.6B transcribes short audio instructions. |
| `generation` | SGLang | SenseNova-U1.5-8B-MoT returns one image per step. |

A backend's `args` go to its framework unchanged, for example SGLang `Engine` arguments such as `mem_fraction_static`. `gpu` sets the worker's `CUDA_VISIBLE_DEVICES`, `env` adds environment variables, and `python` names the interpreter of the environment that has the framework installed. Backends on different GPUs start together, and backends sharing a GPU start one at a time in the order the profile lists them, because an SGLang engine sizes itself from the memory free when it starts.

With `trace: true` the log records the accepted plan, each dispatch and each produced artifact. Use the returned segments' `execution_id` to find these entries. A structurally valid plan can still misinterpret the user's request.

## Adding a framework backend

1. Write a class in a new module, for example `onemeld/backends/vllm.py`, that takes a `BackendSpec` and implements `load`, `start`, `run`, `cancel` and `close` as described by `Backend` in [base.py](../../onemeld/backends/base.py). `load` runs on the worker's main thread before its event loop starts, which is where a framework engine that installs signal handlers must be built.
2. Return the result shape of the backend's task from `run`: `{"text", "finish_reason"}` for chat, `{"text"}` for transcription and `{"modality": "image", "image_b64"}` for images. [chat_request.py](../../onemeld/backends/chat_request.py) and [images.py](../../onemeld/backends/images.py) already convert OpenAI-style chat requests and framework image outputs.
3. Add the class to `ADAPTERS` under its framework and task, or name it in a profile with `adapter: module:Class` without changing OneMeld.
4. Install the framework in the environment that runs its backends, and set `python` when that is not the orchestrator's environment.

[The test worker backend](../../tests/unit_test/fixtures/echo_backend.py) is a minimal adapter that runs on CPU.
