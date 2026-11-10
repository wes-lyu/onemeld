# OneMeld composition

A deployment binds capabilities to native stages. The planner picks actions from those capabilities and returns a complete plan before the execution stages run. The controller validates the plan, dispatches steps whose inputs are ready, and publishes selected artifacts in plan order.

## Request flow

1. `POST /v1/chat/completions` parses the conversation into text and named input artifacts. When the user sends a spoken instruction, the ASR stage transcribes it before planning. The transcript becomes part of the planner input.
2. The planning stage returns JSON containing `steps`. A step specifies `id`, `action`, `instruction`, `input_refs`, and `publish`. References must identify client inputs or earlier steps and match the receiving capability's input kinds.
3. The composition executor stores each completed result under its step ID. The next stage receives the actual text or media referenced by its step. Independent steps can run on free stages within the configured concurrency limit.
4. The API returns selected results in `choices[0].message.segments`. Text is also available in `message.content`. For streaming requests the same segments arrive as SSE deltas.

A malformed plan fails validation before execution. The profile allows one repair attempt before reporting a planning error. The controller bounds the active request count, retained bytes, number of dispatches, and request duration. Disconnect and cancellation propagate to the running stages.

## Native bindings

| Stage | Role in the example |
| --- | --- |
| `planning` | Qwen3.5-4B produces the upfront plan. |
| `understanding` | Qwen3-VL-4B-Instruct answers text tasks and compares referenced images. |
| `generation` | SenseNova-U1.5-8B-MoT returns one image per step. |
| `asr` | Qwen3-ASR-0.6B transcribes short audio instructions. |

[The example profile](../../examples/onemeld/serve.yaml) configures the model paths and placement. [The executor](../../sglang_omni/pipeline/umm_composition.py) is model-independent. [The stage bindings](../../sglang_omni/models/qwen_umm/config.py) declare model capabilities and map them to existing stage factories.

With tracing enabled, the log records the accepted plan, each dispatch, and each produced artifact. Use the returned segments' `execution_id` to find these entries. A structurally valid plan can still misinterpret the user's request.
