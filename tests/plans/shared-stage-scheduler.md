# Shared Stage and scheduler test plan

Status: 327 related CPU cases passed on the remote Linux host on 2026-10-24 (America/Los_Angeles), including the failure-notification fix and real Gloo lifecycle checks. Eleven accelerator cases were deselected. Hardware validation remains pending.

## Versions and isolation

Run from this branch's checkout with its own environment, output directory and pytest cache. Install the repository's test dependencies, including SGLang. Do not add another checkout to PYTHONPATH or use an editable install pointing at another branch. Confirm the imported sglang_omni path before running. No SenseNova code or fixtures are required.

Use this commit's parent as the regression baseline. Run baseline and candidate sequentially on the same hardware with identical checkpoints, requests, seeds and runtime settings. Start a fresh server for each revision and wait for all workers to exit before switching. Record both commit IDs and dependency versions.

## Unit and integration cases

```bash
python -m pytest \
  tests/unit_test/pipeline/test_runtime_schema.py \
  tests/unit_test/pipeline/test_runtime_adapter.py \
  tests/unit_test/pipeline/test_replica_factory_kwargs.py \
  tests/unit_test/pipeline/test_simple_scheduler_batch_wait.py \
  tests/unit_test/pipeline/test_simple_scheduler_shutdown.py \
  tests/unit_test/pipeline/test_simple_scheduler_concurrent.py \
  tests/unit_test/pipeline/test_tensor_parallel_scheduler.py \
  tests/unit_test/pipeline/test_process_failure.py \
  tests/unit_test/pipeline/test_native_child_processes.py \
  tests/unit_test/pipeline/test_weight_share_runner_wiring.py \
  tests/unit_test/mps/test_mps_runner_wiring.py \
  tests/unit_test/pipeline/test_scheduler.py \
  tests/unit_test/pipeline/test_stage.py \
  tests/unit_test/ming_omni/test_tp.py -m 'not accelerator and not benchmark' -q
```

Check fallback precedence through the factory consumer, including omitted values, explicit None, False and zero, serialization round trips and replica expansion. Check compatible batching, incompatible batch keys, cost limits and concurrency. Check cancellation during validation, follower cancellation after dispatch, failed collective sequences, stop before start, draining queued work, cleanup once on the compute thread and bounded shutdown after a peer failure.

The process-failure suite starts two real CPU workers through StageGroup, blocks cleanup after a forward error, and verifies that the parent starts recovery while the failed worker is still alive, rejects pending and later requests, and reaps both workers. It also checks leader/follower notification before cleanup, outer process cleanup, startup supervision, healthy workers, unreported process death, request validation, cancellation, and normal Stage shutdown. The MPS cases use test doubles and verify the existing launcher-owned cleanup order; they do not run GPU workloads.

The agreed scope is this PR and its affected consumers. Do not expand this run to the whole repository. Keep normal pytest conftest loading enabled and report collection failures or unexpected skips as incomplete validation.

## Hardware regression

First run the existing Stage accelerator cases with python -m pytest tests/unit_test/pipeline/test_stage.py -m accelerator -q. They cover real device payloads, CUDA IPC and relay behavior; the CPU cases cannot substitute for them.

The new TensorParallelScheduler has no production model consumer in this branch. Its two-rank probe uses real all-reduces: one rank enters compute while the other has not started, then both receive cancellation and stop. Both dispatched requests must finish before cleanup. The failure cases use StageGroup and the production Stage/scheduler path, with either rank failing while its peer has entered a collective; the parent must fail pending/new requests and reap both workers even when cleanup blocks. CPU/Gloo variants passed. The three NCCL variants are collected but remain unexecuted:

```bash
python -m pytest tests/unit_test/pipeline/test_tensor_parallel_scheduler.py tests/unit_test/pipeline/test_process_failure.py -m accelerator -q
```

Run this command with two visible CUDA devices. An unrelated model's TP success does not exercise this scheduler. The cancellation/drain probe checks scheduler output; the Stage cancellation tests separately check that cancelled results are suppressed.

For the changed shared Stage paths, run one representative existing pipeline with cached weights against the parent and candidate, using the same requests and fresh processes. Select that consumer before allocation instead of loading several unrelated models. The SenseNova runs in later PRs provide additional integration coverage, not a replacement for this branch's own checks.

One device covers the Stage accelerator cases; two devices are required for the NCCL probe. Record outputs, startup/shutdown time and peak memory during functional runs. Repeat paired measurements only to investigate an unexpected difference or substantiate a performance claim; do not multiply every functional case by an unconditional benchmark repeat count.

## Acceptance and evidence

Every selected test must pass, with skips justified by the selected hardware. Requests must complete or fail according to the cancellation/error contract, with no missing results, collective hangs or surviving workers. Preserve raw logs, exit codes, workload settings, timing/memory observations and paired output comparisons. A downstream model's successful run does not replace this branch's own result.

## Local validation record

The failure-notification changes on top of de864ede9 were exercised on macOS arm64 with Python 3.12.8, PyTorch 2.14.1, Transformers 5.19.0 and pytest 9.1.1. SGLang 0.5.22 was unavailable from the configured package index, so this run used a wheel built from the local SGLang source snapshot f274184f2f. Omni was imported from this branch's own checkout. No repository dependencies or production APIs were patched to make tests collect.

The eleven test_process_failure.py cases passed, including real spawned-worker recovery. The wider CPU selection listed above, excluding test_scheduler.py, test_stage.py and ming_omni/test_tp.py, produced 115 passed and one dependency failure in test_replica_factory_kwargs.py: the SGLang import chain requires mlx. A separate attempt to collect test_stage.py failed because the available SGLang snapshot lacks get_dp_tp_group. These results are partial local validation, not a passing full suite or pinned-environment CI result. Repository pre-commit checks passed on the changed files. Normal pytest conftest loading remained enabled. A short pytest basetemp was needed for macOS Unix socket path limits.

As a regression control, the original de864ede9 death-only monitor was temporarily substituted in the test process without editing repository files. The live-worker recovery test timed out; the new monitor passed the same case. Parent notification does not depend on worker cleanup returning. This does not establish an overall deadline for all teardown paths or validate NCCL/HCCL behavior. No accelerator inference or hardware tests were run.

## Remote CPU validation and allocation gate

The final selection contains the fourteen focused modules above, seven startup/stop/launcher nodes from test_ipc.py, and test_stage_process_exits_without_interpreter_teardown from test_stage_process_env.py. It passed 324 cases with no skips and eight accelerator cases deselected. The existing factory-traceback test initially exposed a regression: an early failure event hid the child exception. The fix sends exception details before cleanup and includes them in the parent's startup error; the original assertion and the early-notification tests pass together.

The host used Python 3.12.15, SGLang 0.5.22, PyTorch 2.14.1+cu130, Transformers 5.19.0 and pytest 9.1.1, with CPUOmniPlatform and no visible CUDA devices. The original host environment was reused without dependency or production API shims. Raw evidence is in /workspace/omni-cpu-shared-stage-20261024/results/four-pr/pr1-final.log and pr1-final.xml. This result covers the working-tree fix on top of de864ede9, not that commit alone.

Before hardware allocation, preserve the final patch/commit identity, prepare the NCCL probe and request corpus, and verify checkpoint availability without loading it on a device. Keep the later branches based on the finalized shared-runtime commit and verify their source identities before publishing.

The temporary composition of this PR stack plus the latest PR1 working-tree fix passed 235 selected API/UMM/scheduler/failure cases with no skips. It was assembled in /workspace/omni-cpu-shared-stage-20261024/stack-latest without rewriting any branch. Evidence is in results/four-pr/stack-latest.log and stack-latest.xml; the tested patch is preserved as pr1-latest-code.patch.

## Final CPU lifecycle recheck

The full focused selection passed 325 cases with no skips and eleven accelerator cases deselected. It includes the real Gloo drain test and both failed-rank variants. Evidence is in results/four-pr/pr1-recheck.log and pr1-recheck.xml; the focused lifecycle run is in collectives-cpu.log and collectives-cpu.xml. NCCL cases remain pending hardware execution.

The subsequent review found two additional paths that entered cleanup before notifying the parent: a fatal Stage receive-loop error and partial startup failure. Both now notify before cleanup. The focused selection passed 327 cases with no failures or skips and eleven accelerator cases deselected. Evidence is in results/review-final/pr1-final.log and pr1-final.xml.
