# SPDX-License-Identifier: Apache-2.0
"""The blind judge's requests and verdicts, against a fake judge server."""

import importlib.util
import json
import sys
from pathlib import Path

import httpx

SCRIPTS = Path(__file__).parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("judge_runs", SCRIPTS / "judge_runs.py")
judge_runs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge_runs)
run_cases = sys.modules["run_cases"]

CASE = run_cases.Case(
    id="T1-p1",
    purpose="chain: two requirements",
    checks=[],
    deliverables=run_cases.Deliverables(min_texts=1, images=1),
    prompt="Draw a fox, then caption it.",
    task="T1",
    category="chain",
    constraints=["Returns one image of a fox", "Returns a caption of that image"],
)


def write_response(results: Path, mode: str) -> dict[str, object]:
    directory = results / "round-1" / mode / CASE.id
    directory.mkdir(parents=True)
    (directory / "segment_1.png").write_bytes(b"png bytes")
    (directory / "segment_2.txt").write_text("A fox in snow (step_1, final_reply).")
    return {
        "round": 1,
        "mode": mode,
        "case": CASE.id,
        "http_status": 200,
        "segment_kinds": ["image", "text"],
    }


def test_the_judge_request_is_blind_and_masks_internal_ids(tmp_path: Path) -> None:
    write_response(tmp_path, "agent_rules")
    request = judge_runs.build_judge_request(
        CASE, tmp_path, tmp_path / "round-1" / "agent_rules" / CASE.id
    )
    text = json.dumps(request)
    assert "agent" not in text and "step_1" not in text and "final_reply" not in text
    assert "[id]" in text
    kinds = [part["type"] for part in request["messages"][1]["content"]]
    assert kinds == ["text", "text", "text", "image_url", "text"]
    items = request["response_format"]["json_schema"]["schema"]["properties"][
        "requirements"
    ]
    assert items["minItems"] == items["maxItems"] == 2


def fake_judge(met: list[bool]) -> httpx.Client:
    def respond(request: httpx.Request) -> httpx.Response:
        verdict = {"requirements": [{"reason": "seen", "met": value} for value in met]}
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(verdict)}}]}
        )

    return httpx.Client(base_url="http://judge", transport=httpx.MockTransport(respond))


def test_a_response_passes_only_with_every_requirement_and_the_image_count(
    tmp_path: Path,
) -> None:
    entry = write_response(tmp_path, "plan")
    with fake_judge([True, True]) as client:
        assert judge_runs.grade(entry, CASE, tmp_path, tmp_path, client)["passed"]
    with fake_judge([True, False]) as client:
        assert not judge_runs.grade(entry, CASE, tmp_path, tmp_path, client)["passed"]
    extra_image = {**entry, "segment_kinds": ["image", "image", "text"]}
    with fake_judge([True, True]) as client:
        assert not judge_runs.grade(extra_image, CASE, tmp_path, tmp_path, client)[
            "passed"
        ]


def test_a_failed_request_fails_without_a_judge_call(tmp_path: Path) -> None:
    entry = {**write_response(tmp_path, "plan"), "http_status": 422}
    with fake_judge([]) as client:
        judgment = judge_runs.grade(entry, CASE, tmp_path, tmp_path, client)
    assert not judgment["passed"] and judgment["note"] == "no response"
