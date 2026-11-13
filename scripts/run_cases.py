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
from difflib import SequenceMatcher
from pathlib import Path

import httpx
from pydantic import JsonValue

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = REPO_ROOT / "examples" / "onemeld" / "cases" / "cases.json"
REQUEST_TIMEOUT_S = 900.0
HEALTH_TIMEOUT_S = 600.0
HEALTH_POLL_INTERVAL_S = 3.0
MIN_COPIED_WORDS = 12
Json = dict[str, JsonValue]
# Words that open a recommendation; the label may follow, or precede within the sentence.
RECOMMENDATION_WORDS = (
    "recommend",
    "winner",
    "better choice",
    "best choice",
    "is preferred",
    "推荐",
)
IMAGE_LABEL = re.compile(
    r"(?:(?:book\s+)?cover\s+)?"
    r"(?:photo|image|picture|cover|concept|design|option|version|icon|照片|图片|封面|方案)"
    r"(?:\s*id)?[\s:：#*]*(step\s*\d+|\d+|[一二两])"
    r"|第\s*([一二两\d])\s*张",
    re.IGNORECASE,
)
SENTENCE_END = re.compile(r"[.!?。！？\n]")
IMAGE_ORDINALS = {"一": "1", "二": "2", "两": "2", "first": "1", "second": "2"}
ORDINAL_IMAGE_LABEL = re.compile(
    r"\b(first|second)\s+(?:photo|image|picture)\b", re.IGNORECASE
)
IMAGE_MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


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
class Deliverables:
    """What the user must receive, whichever way a system produced it."""

    min_texts: int
    images: int


@dataclass(frozen=True, kw_only=True)
class Case:
    id: str
    purpose: str
    checks: list[str]
    deliverables: Deliverables
    prompt: str | None = None
    # What a voice-only request says, for a reader who cannot hear the recording.
    transcript: str | None = None
    # Input files, as paths relative to the case file's directory.
    images: list[str] = field(default_factory=list)
    audio: str | None = None
    scoring: Scoring | None = None
    # Held-out cases only: one case per paraphrase of a task.
    task: str | None = None
    category: str | None = None
    constraints: list[str] = field(default_factory=list)
    # The hand-written plan that mode oracle runs in place of the planner.
    oracle_plan: Json | None = None


@dataclass(frozen=True, kw_only=True)
class CaseFile:
    directory: Path
    options: Json
    seed: int
    max_completion_tokens: int
    cases: list[Case]
    # Cases left out because an input file does not exist yet, with the missing files.
    skipped: dict[str, list[str]] = field(default_factory=dict)


@dataclass(kw_only=True)
class CaseReport:
    case: str
    purpose: str
    task: str | None = None
    category: str | None = None
    http_status: int | None = None
    elapsed_seconds: float | None = None
    execution_id: str | None = None
    segment_kinds: list[str] = field(default_factory=list)
    error: str | None = None
    integrity_errors: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)
    scored_answer: ScoredAnswer | None = None
    model_calls: list[Json] = field(default_factory=list)
    control_rejections: int = 0
    outcome_failures: list[str] = field(default_factory=list)

    @property
    def outcome_passed(self) -> bool:
        """Whether the user received every deliverable; ignores how the steps were wired."""
        return self.http_status == 200 and not self.outcome_failures

    @property
    def passed(self) -> bool:
        return (
            self.outcome_passed
            and not self.integrity_errors
            and all(self.checks.values())
        )

    def as_json(self) -> Json:
        return {
            "case": self.case,
            "purpose": self.purpose,
            "task": self.task,
            "category": self.category,
            "passed": self.passed,
            "outcome_passed": self.outcome_passed,
            "outcome_failures": self.outcome_failures,
            "http_status": self.http_status,
            "elapsed_seconds": self.elapsed_seconds,
            "execution_id": self.execution_id,
            "segment_kinds": self.segment_kinds,
            "error": self.error,
            "integrity_errors": self.integrity_errors,
            "checks": self.checks,
            "model_calls": self.model_calls,
            "control_rejections": self.control_rejections,
            "scored_answer": (
                self.scored_answer.as_json() if self.scored_answer is not None else None
            ),
        }


def load_scoring(values: Json | None) -> Scoring | None:
    if values is None:
        return None
    else:
        return Scoring(image_count=values["image_count"], criteria=values["criteria"])


def heldout_cases(values: Json) -> list[Case]:
    """One case per paraphrase; a voice task takes its spec from the text task it speaks."""
    assets = values["assets"]
    tasks = {task["id"]: task for task in values["tasks"]}

    def asset_path(name: str) -> str:
        return assets.get(name, {}).get("path", name)

    cases: list[Case] = []
    for task in values["tasks"]:
        spec = tasks[task["spoken_from"]] if "spoken_from" in task else task
        images = task.get("images", [task["image"]] if "image" in task else [])
        if "spoken_from" in task:
            requests = [
                (None, text, f"{task['id']}-p{number}.wav")
                for number, text in enumerate(spec["paraphrases"], 1)
            ]
        else:
            requests = [(text, None, task.get("audio")) for text in task["paraphrases"]]
        for number, (prompt, transcript, audio) in enumerate(requests, 1):
            cases.append(
                Case(
                    id=f"{task['id']}-p{number}",
                    purpose=f"{task['category']}: {'; '.join(spec['constraints'])}",
                    checks=[],
                    deliverables=Deliverables(**task["deliverables"]),
                    prompt=prompt,
                    transcript=transcript,
                    images=[asset_path(name) for name in images],
                    audio=asset_path(audio) if audio is not None else None,
                    scoring=load_scoring(spec.get("scoring")),
                    task=task["id"],
                    category=task["category"],
                    constraints=spec["constraints"],
                    oracle_plan=spec.get("oracle_plan"),
                )
            )
    return cases


def missing_inputs(case: Case, directory: Path) -> list[str]:
    names = [*case.images, *([case.audio] if case.audio is not None else [])]
    return [name for name in names if not (directory / name).is_file()]


def load_cases(path: Path) -> CaseFile:
    """Read the development cases, or a held-out task file with a tasks list."""
    values = json.loads(path.read_text())
    if "tasks" in values:
        cases = heldout_cases(values)
        skipped = {
            case.id: missing_inputs(case, path.parent)
            for case in cases
            if missing_inputs(case, path.parent)
        }
        return CaseFile(
            directory=path.parent,
            options=values["options"],
            seed=values["seed"],
            max_completion_tokens=values["max_completion_tokens"],
            cases=[case for case in cases if case.id not in skipped],
            skipped=skipped,
        )
    else:
        pass
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
                deliverables=Deliverables(**case["deliverables"]),
                prompt=case.get("prompt"),
                transcript=case.get("transcript"),
                images=[case["image"]] if "image" in case else [],
                audio=case.get("audio"),
                scoring=load_scoring(case.get("scoring")),
                constraints=case.get("constraints", []),
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
    for image in case.images:
        encoded = base64.b64encode((case_file.directory / image).read_bytes())
        media_type = IMAGE_MEDIA_TYPES[Path(image).suffix.lower()]
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{encoded.decode()}"},
            }
        )
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
    labels: list[tuple[int, str]] = []
    for match in IMAGE_LABEL.finditer(text):
        label = re.sub(r"\s+", "", (match.group(1) or match.group(2)).lower())
        labels.append((match.start(), IMAGE_ORDINALS.get(label, label)))
    for match in ORDINAL_IMAGE_LABEL.finditer(text):
        labels.append((match.start(), IMAGE_ORDINALS[match.group(1).lower()]))
    return [label for _, label in sorted(labels)]


def read_scored_answer(
    text: str, scoring: Scoring, image_label: str | None = None
) -> ScoredAnswer:
    """Each image's score per criterion and the image named in the recommendation."""
    lowered = text.lower()
    recommendation_start = min(
        (lowered.find(word) for word in RECOMMENDATION_WORDS if word in lowered),
        default=len(lowered),
    )
    scores: dict[str, dict[str, float]] = {}
    # The words after a heading's label, such as "foggy harbor" in "Cover 1: Foggy Harbor".
    titles: dict[str, str] = {}
    current_image = image_label
    for line in lowered.splitlines():
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
        # Only a line that starts with a label, such as "**Image 2 Scores:**", switches
        # images; a label elsewhere is a comparison, such as "softer than Image 1".
        heading = IMAGE_LABEL.match(line.lstrip(" \t-*#>•|"))
        if heading is not None:
            current_image = image_labels(heading.group(0))[0]
            title = re.sub(
                r"\b(?:scores?|ratings?|evaluation)\b",
                "",
                line.lstrip(" \t-*#>•|")[heading.end() :],
            ).strip(" :：*-—()（）")
            if len(title) >= 4 and title not in titles.values():
                titles.setdefault(current_image, title)
            else:
                pass
        else:
            pass
        if current_image is not None:
            for criterion, match in positions.items():
                scores.setdefault(current_image, {}).setdefault(
                    criterion, float(match.group(1))
                )
        else:
            pass
    return ScoredAnswer(
        scores=scores,
        recommended_image=recommended_label(lowered, recommendation_start, titles),
    )


def read_published_scores(
    answer: ScoredAnswer,
    scoring: Scoring,
    dispatched: list[Json],
    published: list[str],
    artifacts: dict[str, Json],
) -> ScoredAnswer:
    """Label separately published reviews by the actual image each tool received."""
    if answer.is_complete(scoring) or not all(
        reference in artifacts for reference in published
    ):
        return answer
    else:
        pass
    images = [
        reference for reference in published if artifacts[reference]["kind"] == "image"
    ]
    labels = {reference: str(index) for index, reference in enumerate(images, 1)}
    scores = {label: dict(criteria) for label, criteria in answer.scores.items()}
    for event in dispatched:
        response_id = event["step_id"]
        if response_id not in published or artifacts[response_id]["kind"] != "text":
            continue
        else:
            pass
        viewed = {
            reference["id"]
            for reference in event["input_artifacts"]
            if reference["kind"] == "image"
        }
        if len(viewed) == 1 and viewed <= labels.keys():
            (image_id,) = viewed
            label = labels[image_id]
            review = read_scored_answer(artifacts[response_id]["value"], scoring, label)
            for criterion, score in review.scores.get(label, {}).items():
                scores.setdefault(label, {}).setdefault(criterion, score)
        else:
            pass
    return ScoredAnswer(scores=scores, recommended_image=answer.recommended_image)


def recommended_label(
    lowered: str, recommendation_start: int, titles: dict[str, str]
) -> str | None:
    """The image a recommendation names: after the word in its sentence, then before it.

    So both "Recommendation: Image 2" and "Image 2 is preferred" read as 2. With no
    label nearby, the first label later on counts, and then a heading's title.
    """
    if recommendation_start == len(lowered):
        return None
    else:
        pass
    rest = lowered[recommendation_start:]
    sentence_end = SENTENCE_END.search(rest, 1)
    after = image_labels(rest[: sentence_end.start()] if sentence_end else rest)
    sentence_start = max(
        (
            match.end()
            for match in SENTENCE_END.finditer(lowered[:recommendation_start])
        ),
        default=0,
    )
    before = image_labels(lowered[sentence_start:recommendation_start])
    later = image_labels(rest)
    if after:
        return after[0]
    elif before:
        return before[-1]
    elif later:
        return later[0]
    else:
        sentence = lowered[sentence_start:]
        named = [
            (sentence.find(title), label)
            for label, title in titles.items()
            if title in sentence
        ]
        return min(named)[1] if named else None


def leaked_step_ids(step_ids: set[str], texts: list[str]) -> list[str]:
    """Step IDs a returned text names, also as Step 1 for step1.

    Only IDs with a digit or underscore count, so a word ID such as verdict is not flagged.
    """
    patterns = {
        step_id: r"[\s_]*".join(
            re.escape(part) for part in re.split(r"(?<=[A-Za-z])(?=\d)", step_id)
        )
        for step_id in step_ids
        if re.search(r"[\d_]", step_id)
    }
    return sorted(
        {
            step_id
            for step_id, pattern in patterns.items()
            for text in texts
            if re.search(
                rf"(?<![A-Za-z0-9_]){pattern}(?![A-Za-z0-9_])", text, re.IGNORECASE
            )
        }
    )


def outcome_failures(
    case: Case, segment_kinds: list[str], scored_answer: ScoredAnswer | None
) -> list[str]:
    """Deliverables missing from a response, judged on the returned segments alone."""
    failures: list[str] = []
    texts = segment_kinds.count("text")
    images = segment_kinds.count("image")
    if texts < case.deliverables.min_texts:
        failures.append(
            f"returned {texts} text segments, needs at least {case.deliverables.min_texts}"
        )
    else:
        pass
    if images != case.deliverables.images:
        failures.append(f"returned {images} images, needs {case.deliverables.images}")
    else:
        pass
    if case.scoring is not None and (
        scored_answer is None or not scored_answer.is_complete(case.scoring)
    ):
        failures.append(
            "the text lacks a score per image and criterion, or a recommendation"
        )
    else:
        pass
    return failures


def check_integrity(
    trace: list[Json],
    plan: Json,
    artifacts: dict[str, Json],
    inputs: dict[str, Json],
    *,
    allow_repairs: bool = False,
) -> tuple[list[str], list[Json]]:
    """Errors in the trace's structure, and the steps it dispatched in order."""
    errors: list[str] = []
    plans = [event for event in trace if event["event"] == "plan"]
    if len(plans) != 1:
        errors.append(f"expected one accepted plan, found {len(plans)}")
    else:
        pass
    if not allow_repairs and any(event["event"] == "plan_rejected" for event in trace):
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
    artifacts: dict[str, Json] | None = None,
) -> dict[str, bool]:
    """Each named check on the dispatched steps and the published step IDs."""
    generations = [event for event in dispatched if event["action"] == "generate_image"]
    responses = [event for event in dispatched if event["action"] == "respond"]
    response_ids = {event["step_id"] for event in responses}
    generation_ids = {event["step_id"] for event in generations}

    def reads(event: Json) -> set[str]:
        return {reference["id"] for reference in event["input_artifacts"]}

    ancestors: dict[str, set[str]] = {}
    for event in dispatched:
        direct = reads(event)
        ancestors[event["step_id"]] = direct.union(
            *(ancestors[reference] for reference in direct if reference in ancestors)
        )
    described = {
        event["step_id"] for event in responses if "user_image_1" in reads(event)
    }
    published_responses = [
        event for event in responses if event["step_id"] in published
    ]
    published_review_inputs = set().union(
        *(ancestors[event["step_id"]] for event in published_responses)
    )
    viewed = {
        reference["id"]
        for event in responses
        for reference in event["input_artifacts"]
        if reference["kind"] == "image"
    }
    (message,) = request["messages"]
    generation_inputs = reads(generations[0]) if len(generations) == 1 else set()
    if len(generations) == 1 and artifacts:
        generation = generations[0]
        generation_position = trace.index(generation)
        accepted_plan = next(
            (event["reply"] for event in trace if event["event"] == "plan"), {}
        )
        generation_instruction = next(
            (
                step["instruction"]
                for step in accepted_plan.get("steps", [])
                if step["id"] == generation["step_id"]
            ),
            "",
        )
        instruction_words = re.findall(r"\w+", generation_instruction.casefold())
        request_texts = [
            part["text"] for part in message["content"] if part["type"] == "text"
        ]
        request_texts.extend(
            event["text"]
            for event in trace[:generation_position]
            if event["event"] == "artifact"
            and event["step_id"] == "user_transcript"
            and event.get("text")
        )
        request_words = re.findall(r"\w+", " ".join(request_texts).casefold())
        request_overlap = (
            SequenceMatcher(None, instruction_words, request_words, autojunk=False)
            .find_longest_match()
            .size
        )
        for event in trace[:generation_position]:
            if event["event"] != "artifact" or event["step_id"] not in response_ids:
                continue
            else:
                pass
            response_id = event["step_id"]
            response_artifact = artifacts.get(response_id)
            if (
                response_id in generation_inputs
                or response_artifact is None
                or response_artifact["kind"] != "text"
            ):
                continue
            else:
                pass
            response_words = re.findall(r"\w+", response_artifact["value"].casefold())
            copied_words = (
                SequenceMatcher(None, instruction_words, response_words, autojunk=False)
                .find_longest_match()
                .size
            )
            if copied_words >= MIN_COPIED_WORDS and copied_words > request_overlap:
                generation_inputs.add(response_id)
            else:
                pass
    else:
        pass
    transcription_positions = [
        index
        for index, event in enumerate(trace)
        if event["event"] == "input_transcription"
    ]
    transcript_position = next(
        (
            index
            for index, event in enumerate(trace)
            if event["event"] == "artifact"
            and event["step_id"] == "user_transcript"
            and bool(event.get("text"))
        ),
        -1,
    )
    first_controller_position = next(
        (
            index
            for index, event in enumerate(trace)
            if event["event"] == "model_call" and event["role"] == "control"
        ),
        -1,
    )
    available = {
        "text_result_drives_generation": bool(response_ids & generation_inputs),
        "publishes_text_and_image": "text" in kinds and "image" in kinds,
        "reference_seen_by_vl": bool(described),
        "description_drives_generation": bool(described & generation_inputs),
        "generated_image_checked": len(generations) == 1
        and any(generations[0]["step_id"] in reads(event) for event in responses),
        "two_images_generated_and_viewed": len(generation_ids) == 2
        and generation_ids <= viewed,
        "publishes_two_images": kinds.count("image") == 2,
        "publishes_text_that_drove_generation": any(
            event["step_id"] in generation_inputs for event in published_responses
        ),
        "publishes_review_of_generated_images": bool(generation_ids)
        and generation_ids <= published_review_inputs,
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
        "one_pre_transcription": len(transcription_positions) == 1
        and transcription_positions[0]
        < transcript_position
        < first_controller_position,
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
    mode: str | None = None,
) -> CaseReport:
    directory = output / case.id
    directory.mkdir(parents=True)
    report = CaseReport(
        case=case.id, purpose=case.purpose, task=case.task, category=case.category
    )
    request = build_request(case, case_file)
    if mode is not None:
        request["stage_params"]["orchestrator"] = {
            **case_file.options,
            "mode": mode,
            **({"oracle_plan": case.oracle_plan} if mode == "oracle" else {}),
        }
    else:
        pass
    http_request = client.build_request("POST", "/v1/chat/completions", json=request)
    (directory / "request.json").write_bytes(http_request.content)
    before = {path.name for path in trace_root.glob("*")}
    started = time.monotonic()
    try:
        response = client.send(http_request)
    except httpx.TransportError as error:
        report.error = str(error)
        report.elapsed_seconds = round(time.monotonic() - started, 3)
        return report
    report.elapsed_seconds = round(time.monotonic() - started, 3)
    report.http_status = response.status_code
    (directory / "response.json").write_bytes(response.content)
    segments: list[Json] = []
    if response.status_code == 200:
        segments = response.json()["choices"][0]["message"]["segments"]
        report.segment_kinds = [segment["kind"] for segment in segments]
        report.execution_id = segments[0]["execution_id"] if segments else None
        save_segments(segments, directory)
        if case.scoring is not None:
            answer = "\n\n".join(
                segment["data"] for segment in segments if segment["kind"] == "text"
            )
            report.scored_answer = read_scored_answer(answer, case.scoring)
        else:
            pass
        report.outcome_failures = outcome_failures(
            case, report.segment_kinds, report.scored_answer
        )
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
    report.model_calls = [event for event in trace if event["event"] == "model_call"]
    report.control_rejections = sum(
        event["event"] == "plan_rejected" for event in trace
    )
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
    errors, dispatched = check_integrity(
        trace, plan, artifacts, client_inputs(request), allow_repairs=mode is not None
    )
    publish = plan.get("final", {}).get("publish", [])
    if len(segments) != len(publish) or any(
        reference not in artifacts
        or sha256_of(segment["data"], segment["kind"]) != artifacts[reference]["sha256"]
        for segment, reference in zip(segments, publish)
    ):
        errors.append("published segments differ from the planned outputs")
    else:
        pass
    leaked = leaked_step_ids(
        {step["id"] for step in plan.get("steps", [])} | artifacts.keys(),
        [segment["data"] for segment in segments if segment["kind"] == "text"],
    )
    if leaked:
        errors.append(f"returned text names internal step IDs {leaked}")
    else:
        pass
    report.integrity_errors = errors
    if case.scoring is not None and report.scored_answer is not None and not errors:
        report.scored_answer = read_published_scores(
            report.scored_answer, case.scoring, dispatched, publish, artifacts
        )
        report.outcome_failures = outcome_failures(
            case, report.segment_kinds, report.scored_answer
        )
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
        artifacts,
    )
    return report


def save_report(report: CaseReport, output: Path) -> None:
    (output / report.case / "report.json").write_text(
        json.dumps(report.as_json(), ensure_ascii=False, indent=2)
    )
    failed = [name for name, passed in report.checks.items() if not passed]
    problems = [
        *report.integrity_errors,
        *(f"check {name}" for name in failed),
        *(f"outcome: {failure}" for failure in report.outcome_failures),
    ]
    if report.error is not None:
        problems.append(report.error)
    else:
        pass
    print(
        f"{output.name}/{report.case}: {'PASS' if report.passed else 'FAIL'} "
        f"outcome={'pass' if report.outcome_passed else 'fail'} "
        f"http={report.http_status} {report.elapsed_seconds}s "
        f"segments={report.segment_kinds} {'; '.join(problems)}",
        flush=True,
    )


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
        save_report(report, output)
        reports.append(report)
    (output / "summary.json").write_text(
        json.dumps(
            [report.as_json() for report in reports], ensure_ascii=False, indent=2
        )
    )
    return reports


def compare_modes(
    client: httpx.Client,
    case_file: CaseFile,
    trace_root: Path,
    output: Path,
    only: set[str] | None,
    repeat: int,
    modes: list[str],
) -> list[CaseReport]:
    """Run every case in every mode each round, rotating which mode goes first."""
    reports: list[CaseReport] = []
    attempts: list[Json] = []
    cases = [case for case in case_file.cases if only is None or case.id in only]
    for round_number in range(1, repeat + 1):
        for case_index, case in enumerate(cases):
            first = (round_number + case_index) % len(modes)
            for mode in [*modes[first:], *modes[:first]]:
                if mode == "oracle" and case.oracle_plan is None:
                    continue
                else:
                    pass
                directory = output / f"round-{round_number}" / mode
                report = run_case(client, case, case_file, trace_root, directory, mode)
                save_report(report, directory)
                reports.append(report)
                attempts.append(
                    {"round": round_number, "mode": mode, **report.as_json()}
                )
    (output / "summary.json").write_text(
        json.dumps(attempts, ensure_ascii=False, indent=2)
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
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES,
        help="A case file, or a held-out task file such as benchmarks/heldout/tasks.json.",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        required=True,
        help="The server's composition.trace_dir, on this host.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", help="Comma-separated case IDs, such as F1,F3.")
    parser.add_argument(
        "--compare-agent",
        action="store_true",
        help="Pair each plan run with an Agent run; the same as --modes plan,agent.",
    )
    parser.add_argument(
        "--modes",
        help="Comma-separated request modes to run on every case, such as "
        "plan,oracle,agent,agent_constrained. The server must route each one; "
        "oracle runs each case's hand-written plan and skips cases without one.",
    )
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
        only = set(options.only.split(",")) if options.only else None
        case_file = load_cases(options.cases)
        if case_file.skipped:
            print(
                f"Skipping {len(case_file.skipped)} cases with missing inputs: "
                + ", ".join(
                    f"{case} ({', '.join(names)})"
                    for case, names in case_file.skipped.items()
                ),
                flush=True,
            )
        else:
            pass
        if options.modes or options.compare_agent:
            reports = compare_modes(
                client,
                case_file,
                options.trace_dir,
                options.output,
                only,
                options.repeat,
                options.modes.split(",") if options.modes else ["plan", "agent"],
            )
        else:
            reports = run_rounds(
                client,
                case_file,
                options.trace_dir,
                options.output,
                only,
                options.repeat,
            )
    passed = sum(report.passed for report in reports)
    print(f"{passed}/{len(reports)} case runs passed; results in {options.output}")
    sys.exit(0 if passed == len(reports) else 1)


if __name__ == "__main__":
    main()
else:
    pass
