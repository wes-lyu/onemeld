# SPDX-License-Identifier: Apache-2.0
"""Run the composition validation cases against a OneMeld server and check each one.

The server must run on this host with composition.trace_dir set, so this script
can read each request's plan, trace and intermediate results:

    python scripts/run_cases.py --url http://127.0.0.1:8897 \\
        --trace-dir /tmp/onemeld-trace --output results

Each case directory under --output holds the request, the response, each
returned segment, a copy of the request's trace and report.json. summary.json
lists every case. Checks cover the request's structure: one valid plan, intact
hashes and references, steps that run as planned and feed each other, and the
expected outputs. A case with scoring rules also reads each image's scores and the
recommendation from the answer, and checks that both are there.
Whether an image or answer is good beyond that still needs a person.
"""

import argparse
import base64
import hashlib
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from pydantic import JsonValue

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = REPO_ROOT / "examples" / "onemeld" / "cases" / "cases.json"
REQUEST_TIMEOUT_S = 900.0
HEALTH_TIMEOUT_S = 600.0
HEALTH_POLL_INTERVAL_S = 3.0
Json = dict[str, JsonValue]
RECOMMENDATION_WORDS = ("recommend", "推荐")
IMAGE_LABEL = re.compile(
    r"(?:photo|image|picture|照片|图片)(?:\s*id)?[\s:：#*]*(step\s*\d+|\d+|[一二两])"
    r"|第\s*([一二两\d])\s*张",
    re.IGNORECASE,
)
CHINESE_DIGITS = {"一": "1", "二": "2", "两": "2"}


@dataclass(frozen=True, kw_only=True)
class Scoring:
    """What a scored comparison of generated images must state."""

    image_count: int
    # Each criterion maps to the names an answer may call it in any case language.
    criteria: dict[str, list[str]]


@dataclass(frozen=True, kw_only=True)
class ScoredAnswer:
    """The scores and recommendation read from an answer, keyed by image label."""

    scores: dict[str, dict[str, float]]
    recommended_image: str | None

    def is_complete(self, scoring: Scoring) -> bool:
        return (
            len(self.scores) == scoring.image_count
            and all(
                len(scores) == len(scoring.criteria) for scores in self.scores.values()
            )
            and self.recommended_image in self.scores
        )

    def as_json(self) -> Json:
        return {"scores": self.scores, "recommended_image": self.recommended_image}


@dataclass(frozen=True, kw_only=True)
class Case:
    id: str
    purpose: str
    checks: list[str]
    prompt: str | None = None
    image: str | None = None
    audio: str | None = None
    scoring: Scoring | None = None


@dataclass(frozen=True, kw_only=True)
class CaseFile:
    directory: Path
    options: Json
    seed: int
    max_completion_tokens: int
    cases: list[Case]


@dataclass(kw_only=True)
class CaseReport:
    case: str
    purpose: str
    http_status: int | None = None
    elapsed_seconds: float | None = None
    execution_id: str | None = None
    segment_kinds: list[str] = field(default_factory=list)
    error: str | None = None
    integrity_errors: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)
    scored_answer: ScoredAnswer | None = None

    @property
    def passed(self) -> bool:
        return (
            self.http_status == 200
            and not self.integrity_errors
            and all(self.checks.values())
        )

    def as_json(self) -> Json:
        return {
            "case": self.case,
            "purpose": self.purpose,
            "passed": self.passed,
            "http_status": self.http_status,
            "elapsed_seconds": self.elapsed_seconds,
            "execution_id": self.execution_id,
            "segment_kinds": self.segment_kinds,
            "error": self.error,
            "integrity_errors": self.integrity_errors,
            "checks": self.checks,
            "scored_answer": (
                self.scored_answer.as_json() if self.scored_answer is not None else None
            ),
        }


def load_scoring(values: Json | None) -> Scoring | None:
    if values is None:
        return None
    else:
        return Scoring(image_count=values["image_count"], criteria=values["criteria"])


def load_cases(path: Path) -> CaseFile:
    values = json.loads(path.read_text())
    return CaseFile(
        directory=path.parent,
        options=values["options"],
        seed=values["seed"],
        max_completion_tokens=values["max_completion_tokens"],
        cases=[
            Case(
                id=case["id"],
                purpose=case["purpose"],
                checks=case["checks"],
                prompt=case.get("prompt"),
                image=case.get("image"),
                audio=case.get("audio"),
                scoring=load_scoring(case.get("scoring")),
            )
            for case in values["cases"]
        ],
    )


def build_request(case: Case, case_file: CaseFile) -> Json:
    content: list[JsonValue] = []
    if case.prompt is not None:
        content.append({"type": "text", "text": case.prompt})
    else:
        pass
    if case.image is not None:
        encoded = base64.b64encode((case_file.directory / case.image).read_bytes())
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded.decode()}"},
            }
        )
    else:
        pass
    if case.audio is not None:
        encoded = base64.b64encode((case_file.directory / case.audio).read_bytes())
        content.append(
            {
                "type": "input_audio",
                "input_audio": {"data": encoded.decode(), "format": "wav"},
            }
        )
    else:
        pass
    return {
        "messages": [{"role": "user", "content": content}],
        "modalities": ["text", "image"],
        "stream": False,
        "max_completion_tokens": case_file.max_completion_tokens,
        "seed": case_file.seed,
        "stage_params": {"orchestrator": case_file.options},
    }


def sha256_of(segment_data: JsonValue, kind: str) -> str:
    """The SHA-256 of a text segment's UTF-8 bytes or a media segment's decoded bytes."""
    if kind == "text":
        assert isinstance(segment_data, str)
        return hashlib.sha256(segment_data.encode("utf-8")).hexdigest()
    else:
        assert isinstance(segment_data, dict) and isinstance(segment_data["url"], str)
        binary = base64.b64decode(segment_data["url"].split(",", 1)[1], validate=True)
        return hashlib.sha256(binary).hexdigest()


def save_segments(segments: list[Json], directory: Path) -> None:
    for index, segment in enumerate(segments, 1):
        data = segment["data"]
        if segment["kind"] == "text" and isinstance(data, str):
            (directory / f"segment_{index}.txt").write_text(data)
        elif isinstance(data, dict) and isinstance(data.get("url"), str):
            (directory / f"segment_{index}.png").write_bytes(
                base64.b64decode(data["url"].split(",", 1)[1], validate=True)
            )
        else:
            pass


def client_inputs(request: Json) -> dict[str, Json]:
    """The input artifacts OneMeld names after the client's media, with their hashes."""
    (message,) = request["messages"]
    inputs: dict[str, Json] = {}
    for kind, key, part_type in (
        ("image", "user_image", "image_url"),
        ("audio", "user_audio", "input_audio"),
    ):
        parts = [part for part in message["content"] if part["type"] == part_type]
        for index, part in enumerate(parts, 1):
            payload = (
                part[part_type]["url"].split(",", 1)[1]
                if kind == "image"
                else part[part_type]["data"]
            )
            inputs[f"{key}_{index}"] = {
                "kind": kind,
                "sha256": hashlib.sha256(base64.b64decode(payload)).hexdigest(),
            }
    return inputs


def image_labels(text: str) -> list[str]:
    """Image labels in order of appearance, such as 1 for Photo 1 or step2 for Image step2."""
    labels: list[str] = []
    for match in IMAGE_LABEL.finditer(text):
        label = re.sub(r"\s+", "", (match.group(1) or match.group(2)).lower())
        labels.append(CHINESE_DIGITS.get(label, label))
    return labels


def read_scored_answer(text: str, scoring: Scoring) -> ScoredAnswer:
    """Each image's score per criterion, read up to the recommendation, and the image it names."""
    lowered = text.lower()
    recommendation_start = min(
        (lowered.find(word) for word in RECOMMENDATION_WORDS if word in lowered),
        default=len(lowered),
    )
    scores: dict[str, dict[str, float]] = {}
    current_image: str | None = None
    for line in lowered[:recommendation_start].splitlines():
        positions = {
            criterion: match
            for criterion, names in scoring.criteria.items()
            for name in names
            if (
                match := re.search(
                    re.escape(name.lower()) + r"\W{0,12}(\d+(?:\.\d+)?)", line
                )
            )
        }
        first_criterion = min(
            (match.start() for match in positions.values()), default=len(line)
        )
        # A label after the first criterion is a comparison, such as "softer than Photo 1".
        labels = image_labels(line[:first_criterion])
        if labels:
            current_image = labels[0]
        else:
            pass
        if current_image is not None:
            for criterion, match in positions.items():
                scores.setdefault(current_image, {}).setdefault(
                    criterion, float(match.group(1))
                )
        else:
            pass
    recommended = image_labels(lowered[recommendation_start:])
    return ScoredAnswer(
        scores=scores, recommended_image=recommended[0] if recommended else None
    )


def check_integrity(
    trace: list[Json], plan: Json, artifacts: dict[str, Json], inputs: dict[str, Json]
) -> tuple[list[str], list[Json]]:
    """Errors in the trace's structure, and the steps it dispatched in order."""
    errors: list[str] = []
    plans = [event for event in trace if event["event"] == "plan"]
    if len(plans) != 1:
        errors.append(f"expected one accepted plan, found {len(plans)}")
    else:
        pass
    if any(event["event"] == "plan_rejected" for event in trace):
        errors.append("the planner needed a repair")
    else:
        pass
    known: dict[str, Json] = dict(inputs)
    running: set[str] = set()
    dispatched: list[Json] = []
    for event in trace:
        if event["event"] == "dispatch":
            running.add(event["step_id"])
            dispatched.append(event)
            for reference in event["input_artifacts"]:
                source = known.get(reference["id"])
                if source is None:
                    errors.append(
                        f"{event['step_id']} read {reference['id']} before it existed"
                    )
                elif (source["kind"], source["sha256"]) != (
                    reference["kind"],
                    reference["sha256"],
                ):
                    errors.append(
                        f"{event['step_id']} read a different {reference['id']}"
                    )
                else:
                    pass
        elif event["event"] == "artifact":
            artifact = artifacts.get(event["step_id"])
            if artifact is None or artifact["sha256"] != event["sha256"]:
                errors.append(
                    f"artifact {event['step_id']} does not match its trace event"
                )
            elif sha256_of(artifact["value"], artifact["kind"]) != artifact["sha256"]:
                errors.append(
                    f"artifact {event['step_id']} does not match its own hash"
                )
            else:
                known[event["step_id"]] = artifact
            running.discard(event["step_id"])
        else:
            pass
    if running:
        errors.append(f"steps never finished: {sorted(running)}")
    else:
        pass
    planned = [(step["id"], step["action"]) for step in plan.get("steps", [])]
    if [(event["step_id"], event["action"]) for event in dispatched] != planned:
        errors.append("dispatched steps differ from the plan")
    else:
        pass
    return errors, dispatched


def contract_checks(
    names: list[str],
    request: Json,
    trace: list[Json],
    dispatched: list[Json],
    kinds: list[str],
    published: list[str],
    scoring: Scoring | None,
    scored_answer: ScoredAnswer | None,
) -> dict[str, bool]:
    """Each named check on the dispatched steps and the published step IDs."""
    generations = [event for event in dispatched if event["action"] == "generate_image"]
    responses = [event for event in dispatched if event["action"] == "respond"]
    response_ids = {event["step_id"] for event in responses}
    generation_ids = {event["step_id"] for event in generations}

    def reads(event: Json) -> set[str]:
        return {reference["id"] for reference in event["input_artifacts"]}

    described = {
        event["step_id"] for event in responses if "user_image_1" in reads(event)
    }
    published_responses = [
        event for event in responses if event["step_id"] in published
    ]
    viewed = {
        reference["id"]
        for event in responses
        for reference in event["input_artifacts"]
        if reference["kind"] == "image"
    }
    (message,) = request["messages"]
    available = {
        "text_result_drives_generation": len(generations) == 1
        and bool(response_ids & reads(generations[0])),
        "publishes_text_and_image": "text" in kinds and "image" in kinds,
        "reference_seen_by_vl": bool(described),
        "description_drives_generation": len(generations) == 1
        and bool(described & reads(generations[0])),
        "generated_image_checked": len(generations) == 1
        and any(generations[0]["step_id"] in reads(event) for event in responses),
        "two_images_generated_and_viewed": len(generation_ids) == 2
        and generation_ids <= viewed,
        "publishes_two_images": kinds.count("image") == 2,
        "publishes_text_that_drove_generation": len(generations) == 1
        and any(
            event["step_id"] in reads(generations[0]) for event in published_responses
        ),
        "publishes_review_of_generated_images": bool(generation_ids)
        and any(generation_ids <= reads(event) for event in published_responses),
        "states_scores_and_recommendation": scoring is not None
        and scored_answer is not None
        and scored_answer.is_complete(scoring),
        "publishes_reference_description": any(
            "user_image_1" in reads(event) for event in published_responses
        ),
        "only_visual_response": not generations
        and bool(responses)
        and all(event["action"] == "respond" for event in dispatched),
        "text_only_output": set(kinds) == {"text"},
        "audio_only_client_input": all(
            part["type"] == "input_audio" for part in message["content"]
        ),
        "one_pre_transcription": sum(
            event["event"] == "input_transcription" for event in trace
        )
        == 1,
    }
    unknown = sorted(set(names) - available.keys())
    if unknown:
        raise ValueError(f"Unknown checks {unknown}; known checks: {sorted(available)}")
    else:
        return {name: available[name] for name in names}


def copy_trace(trace_root: Path, execution_id: str, directory: Path) -> Path:
    destination = directory / "trace"
    shutil.copytree(trace_root / execution_id, destination)
    return destination


def run_case(
    client: httpx.Client,
    case: Case,
    case_file: CaseFile,
    trace_root: Path,
    output: Path,
) -> CaseReport:
    directory = output / case.id
    directory.mkdir(parents=True)
    report = CaseReport(case=case.id, purpose=case.purpose)
    request = build_request(case, case_file)
    (directory / "request.json").write_text(
        json.dumps(request, ensure_ascii=False, indent=2)
    )
    before = {path.name for path in trace_root.glob("*")}
    started = time.monotonic()
    try:
        response = client.post("/v1/chat/completions", json=request)
    except httpx.TransportError as error:
        report.error = str(error)
        return report
    report.elapsed_seconds = round(time.monotonic() - started, 3)
    report.http_status = response.status_code
    (directory / "response.json").write_text(response.text)
    segments: list[Json] = []
    if response.status_code == 200:
        segments = response.json()["choices"][0]["message"]["segments"]
        report.segment_kinds = [segment["kind"] for segment in segments]
        report.execution_id = segments[0]["execution_id"] if segments else None
        save_segments(segments, directory)
    else:
        report.error = response.text
    if report.execution_id is None:
        # A failed request still traces up to its failure.
        started_traces = {path.name for path in trace_root.glob("*")} - before
        report.execution_id = started_traces.pop() if len(started_traces) == 1 else None
    else:
        pass
    if report.execution_id is None or not (trace_root / report.execution_id).is_dir():
        report.integrity_errors.append("no trace found for the request")
        return report
    else:
        trace_directory = copy_trace(trace_root, report.execution_id, directory)
    trace = [
        json.loads(line)
        for line in (trace_directory / "events.jsonl").read_text().splitlines()
    ]
    if response.status_code != 200:
        return report
    else:
        pass
    plan_path = trace_directory / "plan.json"
    plan: Json = json.loads(plan_path.read_text()) if plan_path.exists() else {}
    artifacts = {
        path.stem: json.loads(path.read_text())
        for path in (trace_directory / "artifacts").glob("*.json")
    }
    errors, dispatched = check_integrity(trace, plan, artifacts, client_inputs(request))
    publish = plan.get("final", {}).get("publish", [])
    if len(segments) != len(publish) or any(
        sha256_of(segment["data"], segment["kind"]) != artifacts[reference]["sha256"]
        for segment, reference in zip(segments, publish)
    ):
        errors.append("published segments differ from the planned outputs")
    else:
        pass
    report.integrity_errors = errors
    if case.scoring is not None:
        answer = "\n\n".join(
            segment["data"] for segment in segments if segment["kind"] == "text"
        )
        report.scored_answer = read_scored_answer(answer, case.scoring)
    else:
        pass
    report.checks = contract_checks(
        case.checks,
        request,
        trace,
        dispatched,
        report.segment_kinds,
        publish,
        case.scoring,
        report.scored_answer,
    )
    return report


def wait_until_healthy(client: httpx.Client) -> None:
    deadline = time.monotonic() + HEALTH_TIMEOUT_S
    while True:
        try:
            if client.get("/health", timeout=5).status_code == 200:
                return
            else:
                pass
        except httpx.TransportError:
            pass
        if time.monotonic() > deadline:
            raise RuntimeError(f"The server was not healthy within {HEALTH_TIMEOUT_S}s")
        else:
            time.sleep(HEALTH_POLL_INTERVAL_S)


def run_cases(
    client: httpx.Client,
    case_file: CaseFile,
    trace_root: Path,
    output: Path,
    only: set[str] | None = None,
) -> list[CaseReport]:
    reports: list[CaseReport] = []
    for case in case_file.cases:
        if only is not None and case.id not in only:
            continue
        else:
            pass
        report = run_case(client, case, case_file, trace_root, output)
        (output / case.id / "report.json").write_text(
            json.dumps(report.as_json(), ensure_ascii=False, indent=2)
        )
        reports.append(report)
        failed = [name for name, passed in report.checks.items() if not passed]
        problems = [*report.integrity_errors, *(f"check {name}" for name in failed)]
        print(
            f"{case.id}: {'PASS' if report.passed else 'FAIL'} "
            f"http={report.http_status} {report.elapsed_seconds}s "
            f"segments={report.segment_kinds} {'; '.join(problems)}",
            flush=True,
        )
    (output / "summary.json").write_text(
        json.dumps(
            [report.as_json() for report in reports], ensure_ascii=False, indent=2
        )
    )
    return reports


def run_rounds(
    client: httpx.Client,
    case_file: CaseFile,
    trace_root: Path,
    output: Path,
    only: set[str] | None,
    repeat: int,
) -> list[CaseReport]:
    """Run the cases repeat times on one server, each round under output/round-N."""
    reports: list[CaseReport] = []
    for round_number in range(1, repeat + 1):
        directory = output / f"round-{round_number}" if repeat > 1 else output
        directory.mkdir(parents=True, exist_ok=True)
        if repeat > 1:
            print(f"Round {round_number} of {repeat}", flush=True)
        else:
            pass
        reports.extend(run_cases(client, case_file, trace_root, directory, only))
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8897")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument(
        "--trace-dir",
        type=Path,
        required=True,
        help="The server's composition.trace_dir, on this host.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", help="Comma-separated case IDs, such as F1,F3.")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Run the cases this many times on the same server, to check reliability.",
    )
    options = parser.parse_args()
    if options.repeat < 1:
        parser.error("--repeat must be at least 1")
    else:
        pass
    if options.output.exists():
        parser.error(f"{options.output} already exists; choose a new output directory")
    else:
        options.output.mkdir(parents=True)
    with httpx.Client(
        base_url=options.url, timeout=REQUEST_TIMEOUT_S, trust_env=False
    ) as client:
        wait_until_healthy(client)
        reports = run_rounds(
            client,
            load_cases(options.cases),
            options.trace_dir,
            options.output,
            set(options.only.split(",")) if options.only else None,
            options.repeat,
        )
    passed = sum(report.passed for report in reports)
    print(f"{passed}/{len(reports)} case runs passed; results in {options.output}")
    sys.exit(0 if passed == len(reports) else 1)


if __name__ == "__main__":
    main()
else:
    pass
