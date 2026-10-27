# Dynamic Qwen-VL and Qwen-Image composition on UMM

Send a natural-language task through one Chat Completions request. The public UMM composition executor runs a model-generated plan across two independent models: Qwen2.5-VL-3B-Instruct for planning and text/vision answers, and Qwen-Image-2512 for image generation. Both use native SGLang stages. The Qwen package declares capability bindings and model paths only; planning and execution live in `sglang_omni.pipeline.umm_plan` and `sglang_omni.pipeline.umm_composition`. The client submits one request and saves the response.

```text
POST /v1/chat/completions
  → Qwen-VL produces a JSON execution plan
  → UMM dispatches each planned action:
       generate_image → native SGLang multimodal_gen → image segment
       respond        → native SGLang SRT, with referenced images → text segment
  → one Chat Completion with ordered images and text
```

All requests use the same entrypoint and stage configuration. The model chooses the actions, their number, generation prompts, and image dependencies from the user's request. Request keywords and output modalities do not select a predefined workflow. For example, a text question can produce one `respond` action, a one-image request one `generate_image` action, and a two-image comparison two generation actions followed by a response referencing both images.

This is composition of independent model instances inside the serving runtime. The image model owns its own condition encoder; it does not share Qwen-VL weights or KV caches with the understanding stage. The public controller factory and configuration validator now accept an opt-in `composition` specification. Existing adapters retain their original path; native stage implementations and HTTP routes are reused. The [public composition reference](../../docs/developer_reference/umm_composition.md) describes the reusable configuration and protocols.

## Launch

Use a Linux CUDA environment with this checkout's dependencies installed. The supplied profile colocates both stages on one GPU: the understanding stage has a 4096-token context and a 22% SRT memory fraction, while the image stage uses native layerwise transformer offload plus text encoder and VAE CPU offload. This requires substantial host memory and trades latency for GPU residency. The previous fixed-flow tests used a 48 GB RTX 4090 with approximately 94 GiB host RAM; [GPU_RESULTS.md](GPU_RESULTS.md) records that baseline.

```bash
python -m sglang_omni.cli serve \
  --config examples/qwen_umm/launch.yaml \
  --model-path /path/to/Qwen-Image-2512 \
  --text-model-path /path/to/Qwen2.5-VL-3B-Instruct \
  --host 127.0.0.1 --port 8897
```

Activate the environment before launch so native JIT compilation can find its build tools. To place the models on different GPUs, change `understanding.gpu` and `generation.gpu`, and adjust their memory budgets.

## Submit a task

```bash
python -m examples.qwen_umm.compare \
  --base-url http://127.0.0.1:8897 \
  --output-dir results/dynamic-cats \
  --prompt 'Generate two photographs of an orange kitten on a wooden windowsill. Inspect both images, score each out of 10, and explain which is better.'
```

Use another prompt to request a different action sequence, for example `Explain why cats purr in one sentence. Do not generate images.` or `Generate only one photograph of an orange kitten. Do not add text.` The client uses identical controls for these tasks; it does not send a mode or plan.

The request shape is:

```json
{
  "model": "qwen_umm_demo",
  "messages": [{"role": "user", "content": "Generate two cat photographs, inspect them, and choose the better result."}],
  "modalities": ["text", "image"],
  "stream": false,
  "max_completion_tokens": 1024,
  "seed": 42,
  "stage_params": {"orchestrator": {"steps": 20, "width": 512, "height": 512}}
}
```

The planner's own token limit is separate from the answer limit so a short answer request does not truncate the plan. Set `stage_params.orchestrator.planning_max_new_tokens` to override the composition's default of 1536, within the native model's configured context capacity. Text sampling is greedy in this example. Image seeds increment for each generated image within the request.

The configuration supplies one planning example about describing, generating and checking a red bicycle. The public `plan_examples` interface places it in the planner's chat context to demonstrate private text dependencies. Example actions are not executed; each actual request still requires its own model-generated, validated plan. The small planner can misunderstand a task even when its output is structurally valid, so inspect plans as well as returned segments when evaluating a deployment.

## Plans and execution evidence

A model-generated plan has this structure:

```json
{
  "steps": [
    {"id": "cat_a", "action": "generate_image", "instruction": "An orange kitten on a wooden windowsill, natural morning light, photography", "input_refs": [], "publish": true},
    {"id": "cat_b", "action": "generate_image", "instruction": "An orange kitten on a wooden windowsill, natural morning light, photography", "input_refs": [], "publish": true},
    {"id": "review", "action": "respond", "instruction": "Score both images out of 10 and explain the visible difference supporting the winner.", "input_refs": ["cat_a", "cat_b"], "publish": true}
  ]
}
```

The server validates unique IDs, configured actions, typed references to earlier text/image artifacts, and plan limits before execution. This configuration allows six steps and four generated images, with at most two planning attempts. A rejected plan is returned to the model with its validation error for one repair attempt; a second rejection fails the request. No fixed workflow is substituted. A `respond` action receives the actual artifacts named by `input_refs`. Only actions with `publish: true` produce user output, so a private text description can feed a later image prompt without being returned to the user.

This example enables the public executor's optional tracing. `UMM composition trace` log entries contain JSON events for accepted plans, actual capability/stage dispatches, and text/image artifact hashes. Match `execution_id` to the returned segments to verify actual routing and artifact handoff. Other deployments can disable tracing in their composition specification.

The client saves `request.json`, `response.json`, `report.json`, `answer.txt`, and any returned `image_1.png`, `image_2.png`, and subsequent images. Its report includes the endpoint, HTTP status, execution ID, output order, image hashes, and elapsed time. It validates image content against the returned hashes.

## Output and limits

The Chat Completion places joined text in `choices[0].message.content` and ordered image/text segments in `choices[0].message.segments`. The `segments` field and `stage_params` are existing Omni extensions; clients must understand the extension to display generated images. With the OpenAI Python client, supply `stage_params` through `extra_body`.

This version generates one plan before execution and runs it sequentially. It does not revise the plan after inspecting a result. The existing UMM contract permits one in-flight phase per stage per request, so image actions are dispatched sequentially. Planning and response phases share the same Qwen-VL engine.

Chat input is one user message with string content. Multi-turn messages, multipart chat input, image editing, video, external tools, and cross-model tensor/KV transfer are outside the current protocols. The `/generate` endpoint also accepts string tasks with sampling options under `metadata.composition`; an optional PNG in `image_base64` is available as `input_image`. The old demo-specific `mode` options and Qwen adapter have been replaced by the public composition path.

## Validation

```bash
python -m pytest tests/unit_test/qwen_umm \
  tests/unit_test/pipeline/test_umm_composition.py \
  tests/unit_test/pipeline/test_umm_planning.py \
  tests/unit_test/pipeline/test_umm.py \
  tests/unit_test/serve/test_umm_serving.py -q
```

Public composition tests use arbitrary capability and stage names, including separate planner/vision stages, and exercise exact image handoff, private text-to-image dependencies, output visibility, invalid plans and cancellation cleanup. Existing controller, serving and model-adapter tests check compatibility. [Earlier dynamic GPU results](DYNAMIC_GPU_RESULTS.md) record the preceding Qwen-specific implementation before public extraction.
