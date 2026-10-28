# Qwen any-to-any composition on UMM

Send a natural-language task, with optional images and speech, through one Chat Completions request. The public UMM composition executor lets Qwen2.5-VL plan over independent Qwen checkpoints, each on its own native SGLang stage, and returns ordered text, image and audio segments. The Qwen package declares model paths, stages and capabilities only; planning and execution live in `sglang_omni.pipeline`, and the [composition reference](../../docs/developer_reference/umm_composition.md) describes them.

| Configuration | Capabilities | Models |
| --- | --- | --- |
| `QwenUMMPipelineConfig` ([launch.yaml](launch.yaml)) | `respond`, `generate_image` | Qwen2.5-VL, Qwen-Image-2512 |
| `QwenAnyToAnyPipelineConfig` ([launch_any_to_any.yaml](launch_any_to_any.yaml)) | adds `transcribe`, `speak` | adds Qwen3-ASR, Qwen3-TTS |
| `QwenAnyToAnyEditPipelineConfig` ([launch_any_to_any_edit.yaml](launch_any_to_any_edit.yaml)) | adds `edit_image` | adds Qwen-Image-Edit |

```text
POST /v1/chat/completions (text, image_url, input_audio parts)
  → Qwen-VL replies with steps, or with the final outputs
  → UMM runs each ready step on its stage, in parallel where steps are independent:
       respond        → native SGLang chat, with referenced images and videos
       generate_image → native SGLang diffusion
       edit_image     → native SGLang diffusion with image_path
       transcribe     → Qwen3-ASR
       speak          → Qwen3-TTS preprocessing → engine → vocoder, one phase
  → iterative mode: Qwen-VL sees the results and replies again
  → one Chat Completion with the published segments in order
```

The any-to-any configurations default to iterative mode and also allow plan mode; a request picks one with `stage_params.orchestrator.mode`. The text-to-image configuration plans upfront.

## Launch

Use a Linux CUDA environment with this checkout's dependencies installed, and activate it before launch so native JIT compilation can find its build tools. The any-to-any profile places every stage on one 48 GB GPU: the image stage offloads its transformer layers, text encoder and VAE to the CPU, which needs substantial host memory and trades latency for GPU residency. The memory fractions are starting points to measure, not tuned values.

```bash
python -m sglang_omni.cli serve \
  --config examples/qwen_umm/launch_any_to_any.yaml \
  --model-path /path/to/Qwen-Image-2512 \
  --text-model-path /path/to/Qwen2.5-VL-3B-Instruct \
  --asr-model-path /path/to/Qwen3-ASR-1.7B \
  --tts-model-path /path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --host 127.0.0.1 --port 8897
```

A CustomVoice checkpoint may need a voice: set `speech` on the `speak` capability, for example `{"voice": "Vivian"}`, in the configuration's composition. The edit profile also takes `--edit-model-path`; it holds two offloaded diffusion transformers, so it needs more host memory than the base profile. To place models on different GPUs, change each stage's `gpu` and memory fraction.

## Submit a task

```bash
python -m examples.qwen_umm.compare \
  --base-url http://127.0.0.1:8897 \
  --model qwen_umm_any_to_any \
  --output-dir results/fox \
  --audio question.wav \
  --prompt 'Transcribe my recording, draw what it describes, describe the drawing in one sentence and read that sentence aloud.'
```

`--image` attaches images, `--audio` attaches WAV or MP3 recordings, and `--mode` selects `plan` or `iterative`. The client saves `request.json`, `response.json`, `report.json` and each segment as `segment_<n>.txt`, `.png` or `.wav`, after checking each media item against its SHA-256.

Media must be inline: the client sends images as data URLs and audio as `input_audio` parts. The planner sees attached and generated images; it learns what a recording says only through a `transcribe` step. A Qwen3-ASR step takes one clip of at most about 30 seconds.

## Replies and evidence

An iterative turn names the steps to run now and, when the task is done, the outputs to publish:

```json
{"steps": [{"id": "said", "action": "transcribe", "instruction": "Transcribe the recording", "input_refs": ["user_audio_1"]}, {"id": "fox", "action": "generate_image", "instruction": "Illustrate the transcript", "input_refs": ["said"]}], "final": null}
```

```json
{"steps": [{"id": "caption", "action": "respond", "instruction": "Describe the drawing in one sentence", "input_refs": ["fox"]}, {"id": "voice", "action": "speak", "instruction": "Read the caption aloud", "input_refs": ["caption"]}], "final": {"publish": ["fox", "voice"], "answer": ""}}
```

The server checks every reply before any step runs: configured actions, unique IDs, references to existing artifacts of accepted kinds, the step, turn and media budgets, and a final output that publishes only step results. A rejected reply goes back to the planner with its error for one repair; a second rejection fails the request. The configurations enable tracing: `UMM composition trace` log entries record accepted and rejected replies, each dispatch with its stage and input hashes, each artifact and each replanning point, keyed by the `execution_id` of the returned segments.

A 3B planner can misread a task even when its reply is valid, so inspect the traced replies as well as the segments when evaluating a deployment. A larger Qwen2.5-VL planner needs a larger `understanding` budget.

## Validation

```bash
python -m pytest tests/unit_test/qwen_umm tests/unit_test/client/test_image_media.py \
  tests/unit_test/pipeline/test_umm.py tests/unit_test/pipeline/test_umm_composition.py \
  tests/unit_test/pipeline/test_umm_planning.py tests/unit_test/pipeline/test_umm_protocols.py \
  tests/unit_test/serve/test_umm_serving.py -q
```

These CPU tests run both modes on the same tasks with fake stages, including speech to image to speech, parallel replicas, retries, multipart input, chains on real stage runtimes and cancellation. The any-to-any configurations have not run on a GPU yet. [GPU_RESULTS.md](GPU_RESULTS.md) and [DYNAMIC_GPU_RESULTS.md](DYNAMIC_GPU_RESULTS.md) record smoke runs of earlier text-to-image versions.
