# SPDX-License-Identifier: Apache-2.0
"""Summarize a paired plan and agent run as Markdown tables.

    python scripts/summarize_runs.py results --cases examples/onemeld/cases/cases.json

Reads the summary.json that run_cases.py --compare-agent writes. Success is the
outcome verdict, recomputed from the saved segments, beside the trace verdict the
run recorded. Tokens are split into input and output for control and task calls.
Every cost and latency figure is given over all requests and over the pairs where
both modes succeeded, since each view alone is biased.
"""

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from run_cases import (
    DEFAULT_CASES,
    Case,
    load_cases,
    outcome_failures,
    read_scored_answer,
)

MODE_NAMES = {
    "plan": "OneMeld",
    "oracle": "Hand-written plan",
    "agent": "Agent",
    "agent_constrained": "Agent, constrained calls",
    "agent_rules": "Agent, planning rules",
}


@dataclass(frozen=True, kw_only=True)
class Attempt:
    round_number: int
    mode: str
    case: str
    category: str | None
    trace_passed: bool
    outcome_passed: bool
    outcome_failures: list[str]
    end_to_end_seconds: float
    control_calls: int
    control_input_tokens: int
    control_output_tokens: int
    task_input_tokens: int
    task_output_tokens: int
    stage_seconds: dict[str, float]

    @property
    def total_tokens(self) -> int:
        return (
            self.control_input_tokens
            + self.control_output_tokens
            + self.task_input_tokens
            + self.task_output_tokens
        )


def load_attempt(entry: dict, results: Path, case: Case) -> Attempt:
    directory = results / f"round-{entry['round']}" / entry["mode"] / entry["case"]
    texts = [path.read_text() for path in sorted(directory.glob("segment_*.txt"))]
    scored_answer = (
        read_scored_answer("\n\n".join(texts), case.scoring)
        if case.scoring is not None
        else None
    )
    failures = (
        outcome_failures(case, entry["segment_kinds"], scored_answer)
        if entry["http_status"] == 200
        else [f"HTTP {entry['http_status']}"]
    )
    tokens: dict[tuple[str, str], int] = defaultdict(int)
    stage_seconds: dict[str, float] = defaultdict(float)
    for call in entry["model_calls"]:
        role = "control" if call["role"] == "control" else "task"
        stage = "control" if role == "control" else call["backend"]
        stage_seconds[stage] += call["elapsed_ns"] / 1e9
        usage = call["usage"]
        if usage is not None:
            tokens[role, "input"] += usage["prompt_tokens"]
            tokens[role, "output"] += usage["completion_tokens"]
        else:
            pass
    return Attempt(
        round_number=entry["round"],
        mode=entry["mode"],
        case=entry["case"],
        category=case.category,
        trace_passed=entry["passed"],
        outcome_passed=not failures,
        outcome_failures=failures,
        end_to_end_seconds=entry["elapsed_seconds"],
        control_calls=sum(call["role"] == "control" for call in entry["model_calls"]),
        control_input_tokens=tokens["control", "input"],
        control_output_tokens=tokens["control", "output"],
        task_input_tokens=tokens["task", "input"],
        task_output_tokens=tokens["task", "output"],
        stage_seconds=dict(stage_seconds),
    )


def mean(values: list[float]) -> str:
    return f"{statistics.fmean(values):,.0f}" if values else "—"


def median_seconds(values: list[float]) -> str:
    return f"{statistics.median(values):.1f}" if values else "—"


def pair(*cells: str) -> str:
    return " / ".join(cells)


def cost_rows(attempts: list[Attempt], title: str, modes: list[str]) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| Mode | Requests | Control calls | Control in | Control out | Task in | Task out | Total | End to end (s, median) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in modes:
        chosen = [attempt for attempt in attempts if attempt.mode == mode]
        calls = (
            f"{statistics.fmean(attempt.control_calls for attempt in chosen):.2f}"
            if chosen
            else "—"
        )
        lines.append(
            f"| {MODE_NAMES[mode]} | {len(chosen)} | {calls} "
            f"| {mean([attempt.control_input_tokens for attempt in chosen])} "
            f"| {mean([attempt.control_output_tokens for attempt in chosen])} "
            f"| {mean([attempt.task_input_tokens for attempt in chosen])} "
            f"| {mean([attempt.task_output_tokens for attempt in chosen])} "
            f"| {mean([attempt.total_tokens for attempt in chosen])} "
            f"| {median_seconds([attempt.end_to_end_seconds for attempt in chosen])} |"
        )
    return [*lines, ""]


def summarize(attempts: list[Attempt]) -> str:
    cases = sorted({attempt.case for attempt in attempts})
    modes = [
        mode for mode in MODE_NAMES if any(attempt.mode == mode for attempt in attempts)
    ]
    order = pair(*(MODE_NAMES[mode] for mode in modes))
    lines = [
        "## Success",
        "",
        f"Each cell is {order}. Outcome judges the returned segments only; trace also requires the reference edges the checks expect.",
        "",
        "| Case | Outcome | Trace |",
        "|---|---:|---:|",
    ]
    for case in [*cases, None]:
        cells = []
        for verdict in ("outcome_passed", "trace_passed"):
            counts = []
            for mode in modes:
                chosen = [
                    attempt
                    for attempt in attempts
                    if attempt.mode == mode and case in (None, attempt.case)
                ]
                passed = sum(getattr(attempt, verdict) for attempt in chosen)
                counts.append(f"{passed}/{len(chosen)}")
            cells.append(pair(*counts))
        lines.append(f"| {case or 'Overall'} | {cells[0]} | {cells[1]} |")
    categories = sorted(
        {attempt.category for attempt in attempts if attempt.category is not None}
    )
    if categories:
        lines += [
            "",
            "## Success by category",
            "",
            f"Outcome successes, {order}.",
            "",
            "| Category | Outcome |",
            "|---|---:|",
        ]
        for category in categories:
            counts = []
            for mode in modes:
                chosen = [
                    attempt
                    for attempt in attempts
                    if attempt.mode == mode and attempt.category == category
                ]
                counts.append(
                    f"{sum(attempt.outcome_passed for attempt in chosen)}/{len(chosen)}"
                )
            lines.append(f"| {category} | {pair(*counts)} |")
    else:
        pass
    lines += ["", "## Outcome failures", ""]
    reasons: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for attempt in attempts:
        for failure in attempt.outcome_failures:
            reasons[attempt.mode, attempt.case, failure].append(attempt.round_number)
    for (mode, case, failure), rounds in sorted(reasons.items()):
        lines.append(f"- {MODE_NAMES[mode]} {case}, rounds {sorted(rounds)}: {failure}")
    if not reasons:
        lines.append("- None")
    else:
        pass
    succeeded = {
        (attempt.round_number, attempt.case, attempt.mode)
        for attempt in attempts
        if attempt.outcome_passed
    }
    both = [
        attempt
        for attempt in attempts
        if all(
            (attempt.round_number, attempt.case, mode) in succeeded for mode in modes
        )
    ]
    lines += [
        "",
        "## Cost and latency",
        "",
        "Token counts are per-request means. Image generation and transcription report no tokens.",
        "",
        *cost_rows(attempts, "All requests, including failures that stop early", modes),
        *cost_rows(both, "Requests where every mode succeeded on outcome", modes),
        "### Tokens per successful request",
        "",
        "All tokens spent, including on failed requests, divided by outcome successes.",
        "",
        "| Mode | Tokens spent | Successes | Tokens per success |",
        "|---|---:|---:|---:|",
    ]
    for mode in modes:
        chosen = [attempt for attempt in attempts if attempt.mode == mode]
        spent = sum(attempt.total_tokens for attempt in chosen)
        successes = sum(attempt.outcome_passed for attempt in chosen)
        per_success = f"{spent / successes:,.0f}" if successes else "—"
        lines.append(
            f"| {MODE_NAMES[mode]} | {spent:,} | {successes} | {per_success} |"
        )
    lines += [
        "",
        "## End-to-end latency by case",
        "",
        f"Median seconds, {order}.",
        "",
        "| Case | All requests | Where every mode succeeded | Matched sets |",
        "|---|---:|---:|---:|",
    ]
    for case in cases:
        cells = []
        for pool in (attempts, both):
            cells.append(
                pair(
                    *(
                        median_seconds(
                            [
                                attempt.end_to_end_seconds
                                for attempt in pool
                                if attempt.mode == mode and attempt.case == case
                            ]
                        )
                        for mode in modes
                    )
                )
            )
        pairs = sum(attempt.case == case for attempt in both) // len(modes)
        lines.append(f"| {case} | {cells[0]} | {cells[1]} | {pairs} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    options = parser.parse_args()
    cases = {case.id: case for case in load_cases(options.cases).cases}
    entries = json.loads((options.results / "summary.json").read_text())
    print(
        summarize(
            [
                load_attempt(entry, options.results, cases[entry["case"]])
                for entry in entries
            ]
        ),
        end="",
    )


if __name__ == "__main__":
    main()
