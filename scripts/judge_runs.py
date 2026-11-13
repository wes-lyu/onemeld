# SPDX-License-Identifier: Apache-2.0
"""Grade each returned response against its case's requirements with a local vision model.

    python scripts/judge_runs.py results --cases benchmarks/heldout/tasks.json \\
        --url http://127.0.0.1:30001

Reads the summary.json that run_cases.py --modes writes. The judge sees the request,
the user's input images, the requirements and the returned items in order, never the
mode, trace or plan, and internal result IDs are masked. A response passes when it
returns the required number of images and the judge finds every requirement met.
Writes judgments.json next to summary.json and prints passes per mode and category.
"""

import argparse
import base64
import json
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict

from run_cases import DEFAULT_CASES, Case, Json, load_cases

JUDGE_TIMEOUT_S = 600.0
SEGMENT_NUMBER = re.compile(r"segment_(\d+)\.")
INTERNAL_ID = re.compile(r"\b(?:step_?\d+|final_reply|user_image_\d+)\b", re.IGNORECASE)
MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
JUDGE_RULES = (
    "You grade what a system returned for a user's request. You see the request, the "
    "user's input images if any, a numbered list of requirements, and the returned items "
    "in order. Judge each requirement only from the returned items and the input images. "
    "A requirement about order or process, such as writing a design before generating, is "
    "met when the returned content is consistent with it. A returned text that describes, "
    "checks, scores or compares images must agree with those images; a claim the images "
    "contradict does not meet the requirement. Do not reward polite filler or claims that "
    "work was done; only what was returned counts. For each requirement, give a short "
    "reason first, then whether it is met. Reply with JSON only."
)


class RequirementVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    reason: str
    met: bool


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    requirements: list[RequirementVerdict]


def data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f"data:{MEDIA_TYPES[path.suffix.lower()]};base64,{encoded}"


def returned_segments(directory: Path) -> list[Path]:
    """The saved segments in the order they were returned."""
    return sorted(
        (
            path
            for path in directory.glob("segment_*")
            if path.suffix in (".txt", ".png")
        ),
        key=lambda path: int(SEGMENT_NUMBER.search(path.name).group(1)),
    )


def build_judge_request(case: Case, case_directory: Path, directory: Path) -> Json:
    """One blind grading request: no mode, trace or plan, and internal IDs masked."""
    request_text = case.prompt or case.transcript or ""
    numbered = "\n".join(
        f"{number}. {requirement}"
        for number, requirement in enumerate(case.constraints, 1)
    )
    content: list[Json] = [
        {
            "type": "text",
            "text": f"User's request:\n{request_text}\n\nRequirements:\n{numbered}",
        }
    ]
    for number, image in enumerate(case.images, 1):
        content += [
            {"type": "text", "text": f"User's input image {number}:"},
            {
                "type": "image_url",
                "image_url": {"url": data_url(case_directory / image)},
            },
        ]
    segments = returned_segments(directory)
    content.append(
        {
            "type": "text",
            "text": f"The system returned {len(segments)} items, in this order:",
        }
    )
    for number, segment in enumerate(segments, 1):
        if segment.suffix == ".txt":
            masked = INTERNAL_ID.sub("[id]", segment.read_text())
            content.append(
                {"type": "text", "text": f"Returned item {number}, text:\n{masked}"}
            )
        else:
            content += [
                {"type": "text", "text": f"Returned item {number}, image:"},
                {"type": "image_url", "image_url": {"url": data_url(segment)}},
            ]
    schema = Verdict.model_json_schema()
    requirement_list = schema["properties"]["requirements"]
    requirement_list["minItems"] = len(case.constraints)
    requirement_list["maxItems"] = len(case.constraints)
    return {
        "model": "judge",
        "messages": [
            {"role": "system", "content": JUDGE_RULES},
            {"role": "user", "content": content},
        ],
        "temperature": 0.0,
        "max_tokens": 8192,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "verdict", "strict": True, "schema": schema},
        },
    }


def grade(
    entry: Json, case: Case, case_directory: Path, results: Path, client: httpx.Client
) -> Json:
    directory = results / f"round-{entry['round']}" / entry["mode"] / entry["case"]
    images = entry["segment_kinds"].count("image")
    base = {
        "round": entry["round"],
        "mode": entry["mode"],
        "case": entry["case"],
        "task": case.task or case.id,
        "category": case.category,
        "images_returned": images,
        "images_required": case.deliverables.images,
    }
    if entry["http_status"] != 200:
        return {**base, "requirements": [], "passed": False, "note": "no response"}
    else:
        pass
    response = client.post(
        "/v1/chat/completions",
        json=build_judge_request(case, case_directory, directory),
    )
    response.raise_for_status()
    verdict = Verdict.model_validate_json(
        response.json()["choices"][0]["message"]["content"]
    )
    requirements = [
        {"requirement": requirement, "reason": item.reason, "met": item.met}
        for requirement, item in zip(
            case.constraints, verdict.requirements, strict=True
        )
    ]
    return {
        **base,
        "requirements": requirements,
        "passed": images == case.deliverables.images
        and all(item["met"] for item in requirements),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--url", default="http://127.0.0.1:30001")
    parser.add_argument("--workers", type=int, default=4)
    options = parser.parse_args()
    case_file = load_cases(options.cases)
    cases = {case.id: case for case in case_file.cases}
    entries = json.loads((options.results / "summary.json").read_text())
    # Shuffled so the judge never sees one system's responses as a block.
    random.Random(0).shuffle(entries)
    with httpx.Client(
        base_url=options.url, timeout=JUDGE_TIMEOUT_S, trust_env=False
    ) as client:
        with ThreadPoolExecutor(options.workers) as pool:
            judgments = list(
                pool.map(
                    lambda entry: grade(
                        entry,
                        cases[entry["case"]],
                        case_file.directory,
                        options.results,
                        client,
                    ),
                    entries,
                )
            )
    judgments.sort(key=lambda item: (item["round"], item["case"], item["mode"]))
    (options.results / "judgments.json").write_text(
        json.dumps(judgments, ensure_ascii=False, indent=2)
    )
    passes: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for judgment in judgments:
        for group in (judgment["category"] or "all", "overall"):
            counts = passes[judgment["mode"], group]
            counts[0] += judgment["passed"]
            counts[1] += 1
    for (mode, group), (passed, total) in sorted(passes.items()):
        print(f"{mode:18} {group:10} {passed}/{total}")


if __name__ == "__main__":
    main()
else:
    pass
