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
expected outputs. Whether an image or answer is good still needs a person.
"""

import argparse
import base64
import hashlib
import json
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


@dataclass(frozen=True, kw_only=True)
class Case:
    id: str
    purpose: str
    checks: list[str]
    prompt: str | None = None
    image: str | None = None
    audio: str | None = None


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
        }


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
) -> dict[str, bool]:
    generations = [event for event in dispatched if event["action"] == "generate_image"]
    responses = [event for event in dispatched if event["action"] == "respond"]
    response_ids = {event["step_id"] for event in responses}
    generation_ids = {event["step_id"] for event in generations}

    def reads(event: Json) -> set[str]:
        return {reference["id"] for reference in event["input_artifacts"]}

    described = {
        event["step_id"] for event in responses if "user_image_1" in reads(event)
    }
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
    report.checks = contract_checks(
        case.checks, request, trace, dispatched, report.segment_kinds
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
    options = parser.parse_args()
    if options.output.exists():
        parser.error(f"{options.output} already exists; choose a new output directory")
    else:
        options.output.mkdir(parents=True)
    with httpx.Client(
        base_url=options.url, timeout=REQUEST_TIMEOUT_S, trust_env=False
    ) as client:
        wait_until_healthy(client)
        reports = run_cases(
            client,
            load_cases(options.cases),
            options.trace_dir,
            options.output,
            set(options.only.split(",")) if options.only else None,
        )
    passed = sum(report.passed for report in reports)
    print(f"{passed}/{len(reports)} cases passed; results in {options.output}")
    sys.exit(0 if passed == len(reports) else 1)


if __name__ == "__main__":
    main()
else:
    pass
