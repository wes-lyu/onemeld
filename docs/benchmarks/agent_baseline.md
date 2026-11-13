# Agent baseline log

This log records every change to the iterative-agent baseline and its results, so a reader can see how much the comparison depends on the harness. All runs used one 48 GB RTX 4090, the same three model workers as OneMeld (Qwen3-VL-4B-Instruct with FP8 weights as controller, SenseNova-U1.5-8B-MoT, Qwen3-ASR-0.6B), temperature 0 and cases F1, F2, F3, F4, F6 and F7, three rounds each.

## Results by baseline version

"Trace" is the original pass rule, which also requires the reference edges OneMeld's plans use. "Outcome" judges only what the user receives; see [scripts/summarize_runs.py](../../scripts/summarize_runs.py). Outcome was recomputed only for the last run, whose segments are archived locally.

| Run (2026-11-11) | Baseline | Change from the previous run | Agent, trace | Agent, outcome | OneMeld, trace |
|---|---|---|---:|---:|---:|
| `onemeld-agent-benchmark-sh2c` | In-tree agent loop: JSON turns with constrained decoding and the `AGENT_RULES` prompt | First version | 1/18 | not recomputed | 18/18 |
| `onemeld-qwen-agent-retest-sh2c` | Qwen-Agent 0.0.34 `FnCallAgent` through [scripts/run_qwen_agent.py](../../scripts/run_qwen_agent.py) | Replaced the in-tree loop with Qwen-Agent's own tool loop and prompt format | 6/18 | not recomputed | 18/18 |
| `onemeld-qwen-agent-optional-refs` | Same | `input_refs` defaults to an empty list when a tool call omits it | 8/18 | 11/18 | 18/18 |

## 2026-11-12: nine cases, three modes

Run directory on the instance: `/workspace/paper-exp-20261112/runs`. Cases F1 to F9, three rounds, one server. The bridge is unchanged apart from a new request mode, `agent_constrained`, which constrains each tool call's arguments to the tool's JSON schema and leaves other text free.

| Mode | Outcome | Trace | Control calls per request | Rejected calls or plans |
|---|---:|---:|---:|---:|
| OneMeld | 27/27 | 27/27 | 1.00 | 0 |
| Qwen-Agent | 12/27 | 9/27 | 2.56 | 0 |
| Qwen-Agent, constrained calls | 12/27 | 9/27 | 2.56 | 0 |

The two agent modes produced identical token counts. The agent never emitted a malformed call, so constraining its calls changes nothing. Its failures are behavioral:

- **F1, F5, F8:** it generates the poster directly and never returns the written design.
- **F6:** the controller, which cannot see images, describes the image itself instead of calling the vision tool, and the description is invented.
- **F9:** it scores each photo in a separate call and never recommends one.

OneMeld planner ablations, one round of nine cases each on a fresh server:

| Planner variant | Outcome | Trace | Rejected plans |
|---|---:|---:|---:|
| Full | 9/9 | 9/9 | 0 |
| No constrained decoding | 8/9 | 7/9 | 0 |
| Minimal planning rules | 8/9 | 7/9 | 0 |
| One plan attempt, no repair | 9/9 | 9/9 | 0 |
| Minimal rules and no constrained decoding | 7/9 | 7/9 | 0 |
| Two concurrent steps | 9/9 | 9/9 | 0 |

Read by hand, two of these outcome failures are the score reader's, not the system's: the F9 answer without constrained decoding says a photo "is preferred" instead of "recommended", and the F7 answer with both removed gives scores and a recommendation but labels both photos "Image 1". The real failures are F5 under minimal rules, which skips the written design as the agent does, and F9 with both removed, which never recommends a photo.

What this shows on the development cases:

- The planner never produced an invalid plan in any variant, so constrained decoding, validation and repair explain none of the gap here.
- The planning rules explain part of it: without them OneMeld makes one of the agent's own mistakes.
- The agent had not been given equivalent rules in this run. The next section reports that run.
- Two concurrent steps on one GPU did not change latency (F4 27.7 s against 28.5 s), since both images queue on the same generation worker.

## 2026-11-12: the agent with OneMeld's planning rules

Run directory on the instance: `/workspace/paper-exp-20261112/runs/agent-rules`. Cases F1 to F9, three rounds, one server, modes `plan`, `agent` and `agent_rules`. The new mode appends `PLANNING_RULES` to the agent's system message. These are the five numbered rules of OneMeld's `PLAN_RULES`, reworded so that a plan step is a tool call and a reference is `input_refs`. The text was written once and run once, with no revision against the cases.

| Mode | Outcome | Trace | Control calls | Control tokens in / out | Total tokens | Tokens per success |
|---|---:|---:|---:|---:|---:|---:|
| OneMeld | 27/27 | 27/27 | 1.00 | 765 / 165 | 1,725 | 1,725 |
| Agent | 12/27 | 9/27 | 2.56 | 2,501 / 179 | 3,134 | 7,052 |
| Agent, planning rules | 24/27 | 20/27 | 3.04 | 4,347 / 271 | 5,351 | 6,019 |

- The rules fix F1, F5, F6 and F8 in all three rounds. The agent now writes the design first and calls the vision tool on the reference photo.
- The three remaining failures are F4 round 1 and F9 rounds 2 and 3. Read by hand, all three are real: the agent scores each photo in its own call and never asks for a recommendation. F9 round 1 passes only because one scoring call saw both photos and volunteered a recommendation.
- On the 11 requests every mode completed, totals are 1,780 (OneMeld), 3,937 (agent) and 4,921 (agent with rules) tokens, and median latency is 16.8 s, 17.1 s and 17.9 s.
- The agent with rules is not deterministic across rounds on F4 and F9, although the seed is fixed.

What this shows: most of the reliability gap on these cases came from the prompt, not from planning once. With the same rules the remaining gap is 3 of 27, all on the two-photo comparison, and the remaining cost gap is about 3 times the tokens with equal latency.

## 2026-11-12: held-out batch 1, first run

Run directory on the instance: `/workspace/paper-exp-20261112/runs/heldout-b1`. Task file `benchmarks/heldout/tasks.json`, the 54 cases (18 tasks, three paraphrases each) whose inputs exist. One pass per paraphrase, modes `plan`, `agent` and `agent_rules`, one server. Prompts were frozen before the run; SHA-256 of the planner, protocols, bridge, task file and profile are in `runs/heldout-b1.frozen.sha256`. Prompt revisions at freeze: the agent's planning rules had one; OneMeld's rules were revised over the F1 to F7 runs and the count was not tracked.

Automatic outcome: OneMeld 41/54, agent 35/54, agent with rules 42/54. Three corrections, all read by hand:

- **J01 is impossible under the profile.** It asks for three images and `max_images` is 2, so every system returned two. It is a task-design error and is excluded.
- **A01 and A04 can exceed the image budget.** Their worst case needs three or four images. OneMeld plans the retry unconditionally and hits the limit (HTTP 422); the agents passed by never retrying. Both are excluded until revised.
- **OneMeld's J04 answers are complete.** All three state both scores per cover and pick one, but label covers "Cover 1" and say "Winner", which the score reader misses. The agents' J04 verdicts match a hand read.

| Tasks | OneMeld | Agent | Agent, planning rules |
|---|---:|---:|---:|
| Valid tasks (excluding J01, A01, A04) | 43/45 | 30/45 | 37/45 |
| Valid, not adaptive | 42/42 | 28/42 | 35/42 |
| Adaptive (A02 only) | 1/3 | 2/3 | 2/3 |

| Cost and latency | OneMeld | Agent | Agent, planning rules |
|---|---:|---:|---:|
| Control calls per request | 1.15 | 2.76 | 3.06 |
| Tokens per request, all | 1,638 | 3,192 | 4,749 |
| Tokens per request, where all three succeeded (25) | 1,307 | 2,720 | 3,831 |
| Median end to end where all three succeeded (s) | 10.5 | 9.3 | 10.3 |

Failure patterns, read by hand:

- **The agent with rules drops the joining text step on fan-out tasks.** J02, J04 and J06 (two of three paraphrases each) return the images without the comparison, scores or consistency check. This is the same failure as F4 and F9 on the development set. R01-p3 described the reference image instead of generating a sailboat.
- **The plain agent also skips written steps** (C01, C03, C06, J06) and on S02 generated an image of the poem instead of writing it, in all three paraphrases.
- **Plan-once loses on adaptive tasks, as expected.** On A01 and A02 OneMeld always plans the retry, returns both images and both checks, and on A02-p1 the last text says it "will now generate" an image it cannot.
- **OneMeld's planner steers comparisons.** On J04-p2 and p3 it asked for one "moody" cover and one "clean, readable" cover, which predetermines the scores.

Limits of this result: 14 valid tasks, one writer who knew both prompts, one pass per paraphrase. The non-adaptive gap against the agent with rules (42/42 against 35/42) comes from four tasks. Paraphrases of one task are not independent, so a paired test should treat the task as the unit.

## 2026-11-12: scorer and task fixes after held-out batch 1

- **Score reader.** It now reads labels such as "Cover 1", "Cover Concept 2", "Design 1", "Option 2" and "Icon 1", and recommendations worded as "Winner", "better choice", "best choice" or "is preferred". It looks for the label after the word within the sentence, then before it ("Image 2 is preferred"), and falls back to a heading's title ("the Cracked Key cover is the winner").
- **Rescored with the new reader:** held-out batch 1 gives OneMeld 44/54 automatically (J04 now 3/3, matching the hand read), the agents unchanged. The development runs `main` and `agent-rules` are unchanged. The ablation without constrained decoding is now 9/9, matching its hand read.
- **Known remaining miss:** per-image reviews in separate segments with no or wrong labels, as in `ablate-minimal-unconstrained` F7, where both reviews say "Image 1". A blind judge is still needed.
- **Tasks J01, A01 and A04 rewritten to fit `max_images: 2`** (two teapots; one retry; one icon). Version 1 of the three is excluded from results; `revisions` in the task file records why. The task file's `limits.max_images` is now 2, and a test checks every task against it.

## 2026-11-12: rewritten J01, A01 and A04

Run directory on the instance: `/workspace/paper-exp-20261112/runs/heldout-b1-revised`. The three rewritten tasks, three paraphrases each, modes `plan`, `agent` and `agent_rules`, one pass. Hashes in `runs/heldout-b1-revised.frozen.sha256`.

| Task | OneMeld | Agent | Agent, planning rules |
|---|---:|---:|---:|
| J01 (two teapots, then pick) | 3/3 | 1/3 | 2/3 automatic, 1/3 by hand |
| A01 (retry once if not yellow) | 0/3 | 2/3 | 2/3 |
| A04 (retry the icon once if under 3) | 0/3 | 3/3 | 3/3 |

- **J01:** both agents skipped the comparison on some paraphrases, as on J02, J04 and J06. The agent with rules passed J01-p3 automatically, but its comparison step referenced no images and judged from the instruction text alone, and its answer names `step_1`. Counted as a failure by hand.
- **A01 and A04:** OneMeld always planned the second image, and returned both, in all six requests. On A04-p2 it wrote the condition into the generator's instruction ("If the previous icon scored less than 3"), which a generator cannot act on. The agents passed A04 because every first icon scored 4 or 5, so no retry was needed. On A01-p3 every system generated two images without a check between them.

Batch 1 with the rewritten tasks, 54 requests, hand-corrected:

| Tasks | OneMeld | Agent | Agent, planning rules |
|---|---:|---:|---:|
| All | 46/54 | 36/54 | 42/54 |
| Not adaptive | 45/45 | 29/45 | 35/45 |
| Adaptive (A01, A02, A04) | 1/9 | 7/9 | 7/9 |

## 2026-11-13: the agent's final reply is returned

Baseline change: the bridge used to drop the agent's last message and tell the agent to end with an acknowledgement. Traces of the earlier runs showed real answers in that message: S02's poems, C06's names, the F9 and J04 scores and picks, J01's teapot choice. The bridge now returns the final reply as a text segment after the selected tool results (artifact `final_reply`), and the prompt says so and adds that the agent sees only text. A request with no selected result but a reply no longer fails. All agent modes get the change.

Runs: `/workspace/paper-exp-20261112/runs/final-reply-heldout` (54 held-out cases, rewritten tasks) and `final-reply-dev` (F1 to F9), modes `plan`, `agent` and `agent_rules`, one pass, hashes in `runs/final-reply.frozen.sha256`.

Counting text segments is now too lenient, since every agent request returns a reply. Every agent pass that needed the reply was read by hand and classed as:

- **Grounded:** the reply summarizes tool results the agent received (F4, F9, J04-p3). Counted.
- **Written by the controller:** text it can write without seeing media (C03's ingredients, C06's names). Counted.
- **After the fact:** a design that was asked for before generation and written after it (F1, F5, F8, C01-p3). Counted only in the lenient figure.
- **Ungrounded:** a caption, comparison or check of images the agent never sent to the vision tool (C05, J01, J02, J06, A01-p3, F6), or a reply saying the comparison failed. Not counted.

The score reader also missed complete answers on J04 and F4 (scores written as "Mood Rating: 5" or without image labels); these were read by hand and counted.

| Held-out batch 1, 54 requests | OneMeld | Agent | Agent, planning rules |
|---|---:|---:|---:|
| All | 46/54 | 37/54 (38 lenient) | 45/54 |
| Not adaptive | 45/45 | 31/45 (32 lenient) | 38/45 |
| Adaptive | 1/9 | 6/9 | 7/9 |
| Before the fix, not adaptive | 45/45 | 29/45 | 35/45 |

| Development, F1 to F9, one pass | OneMeld | Agent | Agent, planning rules |
|---|---:|---:|---:|
| Outcome, hand-read | 9/9 | 5/9 (8 lenient) | 9/9 |

| Cost, held-out | OneMeld | Agent | Agent, planning rules |
|---|---:|---:|---:|
| Tokens per request | 1,468 | 3,024 | 4,296 |
| Tokens per request, the 36 all three passed automatically | 1,393 | 2,598 | 4,065 |
| Median end to end on those 36 (s) | 10.9 | 9.4 | 12.8 |
| Tokens per hand-read success | 1,723 | 4,413 | 5,155 |

What changed and what did not:

- The fix closes the development gap for the agent with rules (24/27 before, 9/9 now): F4 and F9 put the recommendation in the final reply.
- On held-out non-adaptive tasks the agent with rules goes from 35/45 to 38/45. Six of its seven remaining failures share one cause: it writes a comparison or caption of generated images without sending them to the vision tool. In J02, J06 and J01-p2 the vision call was rejected first because the agent guessed IDs such as `image_1` while issuing all calls in one turn; the rejection named the real IDs, and the agent answered from its own instructions instead of retrying.
- Final replies often name internal IDs (`step_1`), which the trace checks flag.

## 2026-11-13: hand-written plans as an upper bound (side run)

Not part of the paper's results: on this task set the hand-written plans and OneMeld both reach the ceiling, so the comparison adds no evidence. The `oracle` mode stays in the code for a harder batch.

New request mode `oracle`: the request carries a hand-written plan (`oracle_plan`), which goes through the same validation and execution as a planner reply with no planner call. Every held-out text task has one in `benchmarks/heldout/tasks.json`; a voice task uses the plan of the task it speaks. For adaptive tasks the hand-written plan has no retry, so it is correct only when the first attempt passes. The plans were written after the earlier held-out runs had been read, so they are an upper bound, not a blind baseline.

Run: `/workspace/paper-exp-20261112/runs/oracle-heldout`, modes `plan` and `oracle`, 54 cases, one pass, hashes in `runs/oracle.frozen.sha256`.

| Held-out, 54 requests, hand-read | OneMeld | Hand-written plan |
|---|---:|---:|
| Not adaptive | 45/45 | 45/45 |
| Adaptive | 1/9 | 9/9 |
| Control calls per request | 1.00 | 0 |
| Tokens per request, the 45 both passed | 1,356 | 462 |
| Median end to end on those 45 (s) | 10.9 | 5.4 |

- **On non-adaptive tasks OneMeld matches the hand-written plans.** Pass/fail shows no loss from planning there, so this benchmark cannot separate OneMeld from the ceiling; the agent gap is the agent's own failures.
- **The adaptive tasks never needed a retry.** Every first umbrella, clock and icon passed its check, so the plan without a retry was correct on all nine. OneMeld failed them only by always planning the retry. A useful adaptive task needs a first attempt that reliably fails.
- **Instruction quality differs where pass/fail cannot see it.** The hand-written A02 plan says where the clock hands go and gets 3:00 (checked by eye); OneMeld's earlier A02 plans got 12:10 and 3:50.
- **Planning costs about half the latency on these tasks.** The single planner call accounts for 5.5 s of the 10.9 s median and about two thirds of OneMeld's tokens.
- The three paraphrases of a task share one hand-written plan, so they produce identical outputs; the oracle has 18 independent tasks here, not 54.

## What the current bridge gives the agent

- **Tools:** one per capability (`respond`, `generate_image`, `transcribe`) plus `select_results`. Every tool takes `instruction`, `input_refs` and `publish`.
- **Prompt:** the `SYSTEM_MESSAGE` in the bridge, about 150 words, written once and not tuned per case.
- **Decoding:** free text parsed by Qwen-Agent. There is no constrained decoding, and a rejected tool call is returned to the agent as a tool error.
- **Budget:** `(max_steps + 1) * max_plan_attempts` control calls, 14 with the benchmark profile, and the same step, image and time limits as OneMeld.
- **Inputs:** the controller sees text only, as OneMeld's planner does (`planner_media_kinds: []`).

## Known asymmetries

- OneMeld's planner prompt (`PLAN_RULES`) was developed on F1 to F7. Its rules name style transfer and tie-breaking, which are F2 and F4.
- OneMeld's planner has JSON-schema constrained decoding, plan validation and one repair attempt.
- The agent must return results through tools. A direct answer with no tool call fails with HTTP 422, which is how F6 fails.
- The trace pass rule requires a reference edge from a description to the generation that uses it. The agent passes text inline instead, which is why F2 fails on trace and passes on outcome.

## Rules from here on

- The bridge and both prompts are frozen before any held-out run. Each later change gets a row in the table above.
- Each system gets the same number of prompt revisions on the development cases, and the count is recorded here. Counts so far were not tracked.
- Success is reported on outcome. Trace checks are diagnostics.
