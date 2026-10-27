# Planned multi-model composition on UMM

The public UMM composition mode combines independently configured native model stages behind one request. A planner model generates a bounded action list from declared capabilities. The shared executor validates the list, resolves typed text/image references, dispatches its actions, and publishes only the requested outputs through the existing UMM segment protocol.

## Relation to existing UMM

The original UMM controller already supports adapter-selected phases, result-dependent transitions, multiple stage bindings, bounded execution, and parallel dispatches to different stages. Independent models can already be connected by an adapter. Composition mode standardizes the planning and artifact handling that such an adapter would otherwise implement; it does not introduce dynamic routing or multi-stage execution for the first time.

| Concern | Existing adapter mode | Composition mode |
| --- | --- | --- |
| Next action | Model-specific adapter callbacks | Model-generated plan interpreted by a shared adapter |
| Model bindings | Phase-to-stage configuration | Capability-to-stage configuration plus declared native protocols |
| Intermediate results | Adapter-owned state and optional engine-context references | Shared named text/image artifacts with validated dependencies |
| Output visibility | Adapter constructs outputs | Plan declares which artifacts to publish |
| Planning errors | Adapter implements its own handling | Shared structural validation and optional bounded model repair |
| Scheduling and serving | Existing controller, budgets, cancellation, continuations and segments | Reuses the same mechanisms |
| Execution strategy | Adapter can branch after results or dispatch to different stages in parallel | One accepted plan, executed sequentially |

This implementation extends the available configuration modes while preserving adapter mode. It does not merge checkpoints, train a joint model, or add observation-driven replanning to the composition executor.

## Public interfaces

- `UMMCapability` in `sglang_omni.pipeline.umm_plan` binds a logical action name to a stage and a native request/response protocol. It declares the artifact kinds that action accepts, a description for the planner, and native parameter defaults.
- `UMMCompositionSpec` declares the planner stage, capabilities, plan/image limits, token limits, inline-image size limit, and opt-in execution tracing. Its `phases` property constructs the required UMM phase map.
- `UMMExecutionPlan` and `UMMPlanStep` are the shared model-output format. Steps name capabilities rather than model packages or Python factories.
- `UMMPlanExample` supplies an optional prompt/plan example through `composition.plan_examples`. Examples are validated against the same capability bindings and bounds as real plans.
- `UMMCompositionAdapter` in `sglang_omni.pipeline.umm_composition` implements the existing adapter protocol for every such configuration. Model packages do not supply their own planning or execution adapter.

`create_umm_controller` accepts an optional `composition` specification, either a validated object or a JSON-compatible mapping. With it, the factory selects the public executor. Without it, `adapter` retains its original import-path and one-argument constructor contract. A model-specific adapter and composition mode cannot be selected together.

## Declare capabilities

For example, an existing pipeline can construct its controller arguments as follows:

```python
from sglang_omni.config.schema import FactoryArgs
from sglang_omni.pipeline.umm_plan import UMMCapability, UMMCompositionSpec

composition = UMMCompositionSpec(
    planner_stage="planner_model",
    capabilities={
        "paint": UMMCapability(
            stage="image_model",
            protocol="native_image",
            description="Create exactly one image from a visual description.",
            input_kinds=["text"],
            parameters={"width": 512, "height": 512, "num_inference_steps": 20},
        ),
        "inspect": UMMCapability(
            stage="vision_model",
            protocol="native_chat",
            description="Answer questions or inspect the supplied images.",
            input_kinds=["text", "image"],
            parameters={"temperature": 0.0},
        ),
    },
)
controller_factory = FactoryArgs(
    composition=composition.model_dump(mode="json"),
    phases=composition.phases,
    max_dispatches=composition.max_steps + composition.max_plan_attempts,
    max_outputs=composition.max_steps,
    max_active=1,
    max_retained_bytes=128 * 1024 * 1024,
    timeout_s=1200.0,
)
```

Use these arguments on the ordinary `create_umm_controller` stage of a `UMMPipelineConfig`. Declare each bound stage on that pipeline, with its own model path, native factory and placement, and route its result back to the controller. Planner and execution capabilities can share a stage or use different stages. The existing UMM topology, admission and process-placement validation still applies. Configuration loading also rejects inconsistent composition/phase bindings and a dispatch or output budget smaller than the declared plan bound.

Capability names are configuration values. A configuration can use `paint` and `inspect`, or different names, without changing the executor. The planner's JSON schema enumerates exactly the declared names. Supported protocols are currently `native_chat`, using the native SRT chat stage and producing text, and `native_image`, using the native generation stage and producing one inline image. Image editing, audio and video need additional protocol implementations; declaring a name alone does not add those model capabilities.

## Native stage contracts

| Protocol | Dispatched request | Returned stage data |
| --- | --- | --- |
| `native_chat` | `OmniRequest.inputs` contains chat messages with text and referenced image URLs; sampling values go in `params` | `text`, optionally `finish_reason` |
| `native_image` | `OmniRequest.inputs` contains `prompt`, native generation parameters and `num_outputs_per_prompt=1` | `media` containing exactly one validated inline image |

The stage remains responsible for its model architecture, weights, tokenizer or processor, and native inference. Model families can differ as long as their stages implement these contracts. A compatible replacement mainly changes model/stage bindings and capability configuration. An unsupported architecture still needs backend implementation; a different wire format needs stage or protocol adaptation. The composition executor does not translate arbitrary tensors, latent representations or model-specific responses automatically.

## Plans and typed artifacts

```json
{
  "steps": [
    {"id": "description", "action": "inspect", "instruction": "Write a concise visual description of the requested scene.", "input_refs": [], "publish": false},
    {"id": "photo", "action": "paint", "instruction": "Generate a photograph matching the supplied description.", "input_refs": ["description"], "publish": true},
    {"id": "answer", "action": "inspect", "instruction": "Check the image against the description.", "input_refs": ["description", "photo"], "publish": true}
  ]
}
```

Each completed step produces one artifact named by its unique ID. Artifact kinds follow the chosen capability's protocol. References must point to available artifacts from earlier steps and their kinds must be accepted by the consuming capability. Plans with unknown capabilities, duplicate IDs, forward references, incompatible kinds, no published outputs, or excessive step/image counts are rejected before action execution. A truncated planner response is also rejected; the runtime never substitutes a fixed workflow.

`max_plan_attempts` defaults to one. Set it to two or three to allow bounded model-driven repair: the planner receives its rejected response and the validation error, then generates a corrected complete plan. Each attempt consumes a controller dispatch. Exhausting the attempt or dispatch budget fails the request. No action runs until a plan passes validation.

Optional `plan_examples` become labeled example data in the planner's system instructions, separate from the actual user message. Their actions are never dispatched and their artifacts are never created. They let a deployment demonstrate capability selection and private dependencies without implementing a custom adapter. They do not establish semantic correctness: a structurally valid plan can still misunderstand the user's task. Evaluate planning quality separately from protocol and execution correctness.

`publish` controls delivery to the user, not availability to later actions. Private text can condition an image prompt without appearing as a text segment. Chat actions receive the explicitly referenced text and/or actual inline images. The native image protocol appends referenced text to the planned visual instruction. The shared executor retains all needed artifacts, counts their bytes through `usage`, and releases them through the controller's existing close path on completion, cancellation, failure or expiry. Action execution failures are not retried automatically.

The first implementation plans once and executes sequentially. It retains the controller's one-in-flight-phase-per-stage-per-request rule. It does not add iterative replanning, cross-request memory, arbitrary worker creation, or cross-model tensor/KV transfer.

## Requests and outputs

Existing HTTP and SDK entrypoints are reused. The initial composition request accepts a string task, or a single user chat message with string content. Top-level sampling seed and completion limit are forwarded; image dimensions and steps can be supplied under `stage_params.<controller_stage>`. With `/generate`, use `metadata.composition` for these controls. An optional base64 PNG there is available as the `input_image` artifact to the planner and consuming actions.

The controller returns the existing ordered segments. A Chat Completion includes them in `choices[0].message.segments` and joins published text in `message.content`; image display requires Omni's existing segment extension. Planner output and private artifacts are not response segments. Streaming retains the existing segment behavior and does not stream individual planner or response tokens.

Enable `composition.trace` only when execution inspection is desired. `UMM composition trace` JSON log entries contain accepted plans, rejected planner responses and validation errors, and each actual action's stage, input references and artifact hashes. Private text is represented by its hash in artifact events. Match `execution_id` to returned segments to inspect the execution without exposing private artifacts as user output.

The [Qwen example](../../examples/qwen_umm/README.md) supplies model paths, native stages and a capability configuration. Its model package contains configuration only; the planner schema and execution logic are public UMM components.
