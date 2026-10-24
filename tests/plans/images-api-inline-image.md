# Images API and InlineImage test plan

Status: 112 related CPU cases passed on the remote Linux host on 2026-10-24 (America/Los_Angeles), with no skips. This PR has no accelerator-specific implementation. Real-model HTTP smoke remains an integration check that can share the later SenseNova allocation.

## Versions and isolation

Run this branch in its own checkout and environment with the full test dependencies installed. Verify the imported sglang_omni path. Do not share source PYTHONPATH entries, editable installations, servers, pytest caches or output directories with other branches. The preceding shared-runtime commit is the regression baseline; the Images API changes can also be checked against main independently.

## Unit and HTTP contract cases

```bash
python -m pytest \
  tests/unit_test/client/test_image_media.py \
  tests/unit_test/client/test_image_completion.py \
  tests/unit_test/serve/test_image_api_extensions.py \
  tests/unit_test/serve/test_image_generation_api.py \
  tests/unit_test/serve/test_umm_serving.py \
  tests/unit_test/serve/test_launch_services.py \
  tests/unit_test/pipeline/test_ipc.py::test_launcher_resolves_image_registrar_from_config \
  tests/unit_test/serve/test_openai_api.py::test_chat_api_client_preserves_ordered_media_and_turns \
  tests/unit_test/serve/test_openai_api.py::test_health_reports_starting_until_the_launcher_marks_the_server_up \
  tests/unit_test/serve/test_openai_api.py::test_chat_answer_excludes_the_matched_stop_string \
  -m 'not benchmark and not accelerator' -q
```

Check configured and absent route registrars through the launcher, registration after application initialization, and rejection of conflicting generic/custom image registration. Preserve existing image and non-image routes. Check InlineImage round trips for PNG/JPEG/WebP, MIME/data-URL consistency, byte size, content hash and malformed base64 through the validation consumer.

Check positive dimensions and width/height order, empty and oversized uploads, pixel limits, unsupported formats, corrupt images and file closure on success and failure. Check result completion, empty results, client disconnect and task cancellation, including backend abort and iterator closure. Limit regression to image/UMM routes, launcher registration and the selected existing chat/health consumers. Report collection failures rather than bypassing fixtures; unrelated audio/model suites are outside this run.

## Serving regression

For HTTP plumbing, use the model-independent fixtures in this commit. For end-to-end regression, use an existing baseline image pipeline and the later SenseNova image pipeline; the existing chat/health contract is covered by the selected CPU cases. Send the same valid and invalid requests to parent and candidate servers, including concurrent requests and disconnects. Check response status, schema, generated output, continued service after invalid requests and complete worker cleanup.

Reuse a hardware allocation sequentially for baseline and candidate, but start fresh processes for each version. Keep requests, seeds, checkpoints, dependencies and warmup identical. Record resource usage during the functional pass. Use repeated paired runs only when investigating a difference or making a performance claim.

## Acceptance and evidence

All focused tests and supported regression suites must pass without unexpected skips. Existing endpoints must preserve their successful and error responses. Save commit IDs, environment details, pytest reports, HTTP request/response records, output comparisons and resource measurements. Public API tests must remain runnable before SenseNova is added.

## CPU versus device boundary

InlineImage hashing, strict validation, MIME/data-URL handling, upload limits, image decoding, route registration, response/error schemas and disconnect/abort/iterator cleanup all run on CPU. No GPU or NPU is needed for the 112 passing cases. The remaining check is that real model output crosses these same HTTP boundaries correctly; reuse PR3's T2I/I2I requests for that evidence instead of allocating a separate machine for PR2. Keep an existing image-pipeline smoke for the generic route, which differs from the custom registrar route.

The branch was tested at e687ebaf3 in its own remote directory with the original SGLang 0.5.22 environment, CPUOmniPlatform and CUDA hidden. Raw evidence is in /workspace/omni-cpu-shared-stage-20261024/results/four-pr/pr2-api.log and pr2-api.xml. No SenseNova source or test fixture was present in this branch's snapshot.
