# SenseNova CUDA test plan

Status: 562 related CPU cases passed on the remote Linux host on 2026-10-24 (America/Los_Angeles), with no skips. Eight accelerator cases and real-checkpoint CUDA validation remain pending.

## Versions and isolation

Use this branch's own checkout and installed source. Its prerequisites are the shared Stage/scheduler and Images API commits. NPU code and NPU test fixtures are not prerequisites. Keep the server, source path, pytest cache and output directory separate from other revisions.

The parent has no SenseNova implementation: use it for existing-model regressions, not as the SenseNova output reference. Compare SenseNova behavior with the preserved unsplit implementation at 026469802ec65e6b677e27491315d7f1602a8614 on the same CUDA hardware and checkpoint. Keep a separate environment if the two revisions require different dependencies and record the difference.

## Automated cases

```bash
python -m pytest tests/unit_test/sensenova -m 'not accelerator' -q
python -m pytest tests/unit_test/config/test_example_configs.py -k sensenova -q
python -m pytest tests/unit_test/sensenova -m accelerator -q
```

Run the first two commands in the normal test environment, with Gloo available for CPU TP checks. Run the accelerator selection on CUDA and inspect skips; an all-skipped hardware suite is not a pass. Check model registration, configuration/factory precedence, checkpoint-compatible modules, attention and normalization, RoPE, sharding, sampling, admission, output ordering, UMM lifecycle and API errors. Restrict discovery/config regression to the shipped SenseNova examples and the changed model registration paths; unrelated model suites are outside this validation.

## CUDA serving matrix

Use the launch and request examples in [the SenseNova guide](../../docs/cookbook/sensenova_u1.md). Record the exact command and model revision for every row.

| Workload | Required cases | Expected result |
| --- | --- | --- |
| T2I | Fixed prompts/seeds; n=1 and n=2; square and portrait/landscape sizes | Correct image count, dimensions, ordered UMM segments and valid InlineImage content |
| I2I | One and multiple references; do_resize on/off; supported image formats | Accepted edits match the reference implementation's behavior and configured limits |
| Admission | Empty/oversized prompt, invalid seed/count/size, corrupt later reference, pixel and retained-byte limits | Stable client errors before model work where required; next valid request succeeds |
| Dynamic batching | Concurrent compatible requests and mixed incompatible sizes/settings | Correct per-request results, batching only compatible requests, no cross-request state leakage |
| Cache-DiT | Disabled/enabled; consecutive different requests; cancellation followed by reuse | Correct outputs and reset state; quantify approximation against the disabled reference |
| Offload | Disabled/enabled; supported residency and prefetch settings | Output parity within the predeclared numerical tolerance; memory and latency recorded |
| Lifecycle | Queued/active cancellation, disconnect, shutdown during compute, controlled failure | Correct terminal status and cleanup, no stuck requests or surviving workers |
| Parallelism | TP1, TP2, DP2, DP2xTP2 with the shipped configs | Correct placement, outputs, request routing and cancellation; no collective hangs |

TP2 requires two GPUs, DP2 requires two, and DP2xTP2 requires four. A TP1 run does not qualify another topology. Fix the request corpus before running and reuse it for the unsplit reference and candidate. Compare exact outputs when deterministic; otherwise retain the images and numerical/image-quality measurements with tolerances chosen before the candidate run. Record failures and recovery separately from successful requests.

## Regression and acceptance

Run representative existing image/UMM pipelines against this commit's parent to catch model registration or serving interference. Use fresh processes on the same hardware and identical warmup. Complete one functional pass first; use three paired repeats only for an investigated difference or a performance claim. Capture latency, throughput, peak device memory, startup/shutdown time and paired outputs. No unaccounted functional regression, unexpected skip, state leakage or worker leak is acceptable. Performance differences beyond the predeclared baseline variability need investigation.

Save commit IDs, dependency/driver versions, checkpoint revision, hardware topology, launch commands, request corpus, pytest reports, server logs, generated images and measurement tables. All hardware rows remain pending until actually run.

## CPU completion and economical hardware order

The 562 passing cases comprise 559 SenseNova cases and three example-config cases. They include actual CPU tensors, small native-model forwards, Gloo TP2/TP4/TP8 parity, two-rank native Qwen3 prefix/denoise parity, asymmetric cancellation, API/admission errors, UMM lifecycle, output ordering, Cache-DiT and offload contracts. Gloo tests are CPU tests even when their names include TP; passing them does not validate NCCL or real device kernels. The offload contract tests do not measure actual host/device transfers.

The branch was tested at f0f7e5599 in its own directory with the original SGLang 0.5.22 environment. Tests were split into serial groups for a 4 GiB container; no case was dropped for memory and all Gloo sizes passed. Raw logs and XML are under /workspace/omni-cpu-shared-stage-20261024/results/four-pr/pr3-*. The hardware selection contains three CUDA RoPE cases, four normalization cases and one accelerator RNG case.

After weights, environments and request inputs are ready, use one CUDA allocation sequentially: operator tests first, then TP1 T2I/I2I and HTTP smoke, then batching/cache/offload/cancellation, then TP2 and DP2. PR2's HTTP evidence and the final PR4 CUDA regression can share this allocation, with fresh processes and separate result directories for each revision. PR4 must still be compared with this CUDA parent rather than inferred from NPU success.

TP2 and DP2 each require two devices. DP2xTP2 requires four devices, as does actual TP4 validation. TP8 numerical behavior is CPU-tested, but an eight-device NCCL claim requires an eight-device run; otherwise retain that topology as hardware-unverified. Do not rent four or eight devices for tests that only need one or two. Device memory must fit the checkpoint, replicated modules, target image size and concurrent workload; choose the device after checking those inputs.

Before allocation, freeze the request corpus, seeds, baseline revision, output comparisons, checkpoint cache, launch commands and the latest shared-runtime patch. Download and import checks belong in the CPU preparation phase. Do not spend device time collecting CPU unit tests or resolving Python dependencies.
