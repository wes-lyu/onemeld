# Multi-model composition on UMM

Composition mode serves one request with several independently configured models. A deployment declares capabilities, each bound to native stages and a protocol. A planner model decides which capabilities to run, with which instructions and inputs, and which results the user sees. One shared executor validates the planner's replies, runs the steps on the bound stages, passes typed text, image, video and audio artifacts between them, and publishes the chosen outputs through the existing UMM segment protocol.

## Relation to existing UMM

Composition is an adapter on the shared UMM controller. Deadlines, load shedding, admission, dispatch and output budgets, retained-byte accounting, cancellation, continuations and ordered segments are the controller's, unchanged. Model-specific adapters, such as those of SenseNova, LLaDA2-Uni and Cosmos3, keep their constructor and execution contract and remain the right choice for a single unified checkpoint, whose phases share engine state through context references.

| Concern | Model adapter | Composition |
| --- | --- | --- |
| Next action | Adapter code | Planner model reply, validated against declared capabilities |
| Model bindings | Phase-to-stage map | Capability-to-stages map with a native protocol |
| Intermediate results | Adapter state, optional engine context references | Named text, image, video and audio artifacts |
| Parallel work | Adapter dispatches to distinct stages | Steps whose inputs are ready run on free capability stages |
| Output visibility | Adapter builds outputs | The planner's final publish list, in order |

## Execution modes

A deployment sets `execution_mode` and the `allowed_modes` a request may select with the `mode` option. Both modes share the step format, validation, protocols, artifact store, scheduling, budgets and tracing. They differ only in what the planner returns and whether it is consulted again.

- `plan`: the planner returns every step before any runs, as `{"steps": [{"id", "action", "instruction", "input_refs", "publish"}]}`. Steps with `publish: true` become the output, in plan order. The planner is not consulted again.
- `iterative`: each planner turn returns `{"steps": [...], "final": null | {"publish": [...], "answer": "..."}}`. A turn either runs steps with `final: null`, and the planner sees their results in the next turn, or finishes with `final` and no steps, publishing results it has observed. When a turn runs steps and also returns `final`, the server drops the `final` and runs the steps, so a result is never published before the planner has seen it. On the last turn allowed by `max_planner_turns`, or when no steps remain, the JSON schema itself requires `final`; that turn may still run steps, whose outputs it publishes unseen, and an answer written before their results is dropped. Plan examples are shown to an iterative planner as a run turn followed by a finishing turn.

Steps reference client inputs, earlier results, and earlier steps of the same reply through `input_refs`. A step runs once every artifact it references exists and one of its capability's stages is free, so steps that do not reference each other can run in parallel. Each stage runs one step of a request at a time; a capability that lists several stages spreads ready steps over them, and the remaining steps wait. Set `max_concurrent_steps=1` to serialize business steps across stages on a shared GPU; its default of `None` preserves the existing parallel behavior.

The executor publishes the longest completed prefix of the final publish list after each result. In plan mode a parallel plan therefore still returns its outputs in plan order. In iterative mode nothing is published until the planner returns `final`, so a retried result the planner rejected is never shown. A text `answer` follows the published artifacts.

## Task contracts

`planning_temperature` controls proposal sampling and defaults to zero. Contract repairs and optional reviews use the same value; legacy plan and iterative repairs retain their existing repair temperature. A reasoning planner also needs its native stage's `engine.reasoning_parser` configured and enough `planning_max_new_tokens` for reasoning plus the final JSON. The native chat stage returns only the final content to the composition executor; reasoning token usage remains in the native result.

Task-contract planning forwards a supplied request `seed`, including repair and review calls. Media generation retains its existing per-step seed derivation. Setting a seed helps reproduce sampling but does not guarantee identical results across engine configurations or batch schedules.

Set `task_operations` on a plan-only composition to enable the task-contract planner. The default empty mapping keeps the existing plan and iterative protocols unchanged. Each `UMMTaskBinding` names an existing capability, describes the operation, and declares its semantic input and output types. For example, a recommendation consumes an evaluation, while an image inspection consumes an image. `text_image_task_bindings(text_capability, image_capability)` provides this vocabulary without model names; a deployment chooses the capability bindings.

The model returns an interpretation and operations with `source_text`, per-operation `constraints`, `count`, `depends_on`, `input_refs`. A separate `publish` list selects operation indices to return, in operation order. The decoding schema restricts source quotes and constraints to clauses from the request. The compiler allocates step IDs, expands media counts into one output per call, resolves references, checks semantic types and existing UMM budgets, and builds the publish list. Constraints stay with their operation so later scoring instructions do not become image content. Text operations produce one result even when they inspect several images. A hidden result remains available to dependent operations.

Invalid proposals use the existing bounded repair path before any business step runs. Set `review_plan=True` to add a model review of the contract and its compiled execution before dispatch; the default disables it because model review can reject valid plans and accept incorrect ones. With review enabled, each attempt reserves a proposal and review dispatch and a rejecting review triggers repair. Traces add `contract` and optional `contract_review` alongside the compiled `plan`. This is still upfront planning: execution results do not trigger replanning.

The planner receives text and input IDs with their types. It does not need a vision encoder; the capability stage reads the actual image when dispatched. `planner_stage` can name a separate native text-model stage, independently of the stage bound to image understanding. Spoken instructions still pass through the configured ASR capability first.

Compilation guarantees reference, type, budget and execution-order checks. It cannot prove that a model understood a natural-language request. Model review can both miss mistakes and reject valid plans, so evaluate actions, quantities, constraints, dependencies and output visibility against independently labeled requests. An accepted review is not evidence of semantic correctness by itself.

## Observations

In iterative mode the planner's history holds its accepted replies, each followed by an observation of the results that reply produced: a JSON summary of each artifact with its ID, kind, action, SHA-256, size and text cut to `observation_text_bytes`, then each media artifact whose kind is in `planner_media_kinds` attached once as a content part. Media of other kinds is described only; a planner that cannot hear asks for a transcription step. The observation of the latest turn ends with the remaining steps, turns and media budget.

## Capabilities and protocols

```python
from sglang_omni.pipeline.umm_plan import UMMCapability, UMMCompositionSpec

composition = UMMCompositionSpec(
    planner_stage="understanding",
    capabilities={
        "respond": UMMCapability(stages=["understanding"], protocol="native_chat", description="Answer questions or inspect referenced media.", input_kinds=["text", "image", "video"]),
        "paint": UMMCapability(stages=["image_a", "image_b"], protocol="native_image", description="Generate exactly one image.", input_kinds=["text"], parameters={"width": 512, "height": 512}),
        "transcribe": UMMCapability(stages=["asr"], protocol="asr", description="Transcribe one recording.", input_kinds=["audio"]),
        "speak": UMMCapability(stages=["tts_preprocessing"], protocol="tts", description="Read the referenced text aloud.", input_kinds=["text"], speech={"voice": "Vivian"}),
    },
    execution_mode="iterative",
    allowed_modes=["iterative", "plan"],
)
```

| Protocol | Accepts | Produces | Native request | Native result |
| --- | --- | --- | --- | --- |
| `native_chat` | text, image, video, audio | text | Chat messages; media references as `image_url`, `video_url` and `audio_url` data URL parts | `text`, `finish_reason` |
| `native_image` | text, up to 3 images | image | Generation inputs with `prompt`; image references in `image_path`, which edits them | One inline image in `media` |
| `native_video` | text, up to 1 image | video | Generation inputs with `prompt`; an image reference in `image_path` | One inline video in `media` |
| `image_base64` | text | image | String prompt in `inputs`; sampling settings in `params`, with `n=1` | `modality: image` and one base64 PNG in `image_b64` |
| `asr` | exactly 1 audio | text | `{"audio_bytes": ...}` | `text` |
| `tts` | text | audio | The referenced text, or the instruction; `metadata.tts_params` from `speech` | A float32 waveform, returned as WAV |

`parameters` become native request fields or sampling parameters, except the fields each protocol owns, such as `prompt`, `image_path` and `messages`. Referenced text is appended to generation prompts. Client `width` and `height` apply to text-to-image and text-to-video steps; an edit or image-to-video step keeps its source size unless the capability sets one. Image and video steps take the client seed plus the step's position among accepted generation steps, so seeds do not depend on completion order. The stage remains responsible for its model, weights and processor; a different wire format needs a stage or protocol change.

## Pipeline configuration

Each capability stage maps to a controller phase of the same name, and the planner stage to the `plan` phase; `UMMCompositionSpec.phases` builds this map. A phase stage returns its result to the controller directly, or is the entry of a chain whose last stage does. A chain lets a multi-stage pipeline, such as the speech pipeline `tts_preprocessing → tts_engine → vocoder`, be one capability. Configuration loading checks that each chain is linear and acyclic, ends at the controller, contains no terminal, routed or shared stage, and streams only to later stages of the same chain, and that the controller's `next` names chain entries but no stage inside a chain.

Runtime `num_replicas` binds a request to one replica of a stage, so parallel steps of one request need distinctly named stages listed in one capability. Several stages on one GPU share it, which shows the scheduling but does not speed it up.

The controller's `max_dispatches` must cover the most expensive allowed mode, `max_plan_attempts` per planner turn (`max_planner_turns` turns in iterative mode, one in plan mode) plus `max_steps`, and one input transcription when enabled; `required_dispatches()` computes it. `max_outputs` must cover `max_steps`, plus the answer when iterative mode is allowed; `required_outputs()` computes it.

Before validating a reply, the server fixes what it can without the planner: a step ID that repeats an earlier one or starts with `user_` is renamed, for example a second `generate` to `generate_2`, and each reference names the latest earlier step of that name in the reply, or else an existing artifact. `max_plan_attempts` above one lets the planner repair a rejected reply: it receives the reply and the validation error and returns a corrected one, sampled at temperature 0.7 because a greedy repair tends to repeat the rejected reply. No step runs from a rejected reply. When the attempts run out, the request fails with HTTP 422, whose message names the error, the attempt count and the `execution_id`; a failed step fails the request.

## Requests and outputs

The chat and `/generate` entrypoints are reused. A chat request may hold several messages with `system`, `user` and `assistant` roles, whose content is a string or parts of type `text`, `image_url`, `video_url`, `audio_url` and `input_audio`. Media must be inline data URLs, given as `{"url": ...}` or as a string, or base64 WAV or MP3 data for `input_audio`. Each media part becomes an input artifact `user_image_1`, `user_video_1`, `user_audio_1` and so on, within `max_input_media` and `max_inline_media_bytes`. Options such as `mode`, `width`, `height`, `steps`, `seed` and `planning_max_new_tokens` go under `stage_params.<controller stage>`; a `/generate` string task takes them in `metadata.composition`, where a base64 PNG in `image_base64` becomes the `input_image` artifact.

Set `audio_instruction_capability` to the name of an `asr` capability to accept spoken instructions before planning. If the latest user message has one audio clip and no nonempty text, the executor runs that capability first. Its nonempty transcription replaces the consumed audio in the planner's task and becomes the `user_transcript` text artifact; the original clip is no longer a plan input. Messages with text and an audio attachment still plan first and may transcribe the attachment as a business step. Multiple clips in a voice instruction and empty transcriptions fail explicitly. The default of `None` leaves input handling unchanged.

The result is the existing ordered segment list in `choices[0].message.segments`, with published text also joined in `message.content`. Image, video and audio segments carry inline media with `kind`, `mime_type`, `size_bytes`, `sha256` and a data URL; `InlineMedia` in `sglang_omni.client.image_media` validates them. Audio arrives as a segment, not in `delta.audio`.

With `trace`, `UMM composition trace` log entries record each accepted reply (`plan`), rejected reply (`plan_rejected`), server fix of a reply, with renamed IDs and dropped fields (`normalized`), dispatch with its stage and input hashes (`dispatch`), artifact (`artifact`) and replanning point (`observe`), keyed by the `execution_id` of the returned segments. A spoken instruction adds `input_transcription` before the first plan; its artifact event includes the transcribed text and its hash.

## Limits

The planner decides the work, so a request's length is not known when it is admitted; `max_steps`, `max_planner_turns`, the per-kind media limits and the controller deadline bound it. Steps exchange text and inline media rather than engine state, so a step re-encodes the media it reads. Phases do not stream: a step's result is complete before later steps see it. ASR steps take one clip; long recordings need splitting before the request.

Planner template options such as `enable_thinking` can be passed through `planning_chat_template_kwargs`. The default empty mapping leaves the native model template unchanged. These options affect planning requests only, including repairs and optional reviews.
