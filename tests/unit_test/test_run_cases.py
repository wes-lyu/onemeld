# SPDX-License-Identifier: Apache-2.0
"""The validation case runner, driven through real worker processes."""

import importlib.util
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from onemeld.server import create_app
from tests.unit_test.fixtures.scripted_backends import scripted_config

SCRIPT = Path(__file__).parents[2] / "scripts" / "run_cases.py"
spec = importlib.util.spec_from_file_location("run_cases", SCRIPT)
run_cases = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_cases)
CASES = {case.id: case for case in run_cases.load_cases(run_cases.DEFAULT_CASES).cases}


@pytest.fixture(scope="module")
def reports(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict]:
    """F4, F6, F7 and F9 against a planner that always plans two images and a comparison."""
    root = tmp_path_factory.mktemp("cases")
    case_file = run_cases.load_cases(run_cases.DEFAULT_CASES)
    with TestClient(create_app(scripted_config(root / "trace"))) as client:
        run_cases.run_cases(
            client,
            case_file,
            root / "trace",
            root / "results",
            {"F4", "F6", "F7", "F9"},
        )
    summary = json.loads((root / "results" / "summary.json").read_text())
    return {
        report["case"]: report | {"directory": root / "results" / report["case"]}
        for report in summary
    }


def test_a_matching_plan_passes_with_every_output_saved(
    reports: dict[str, dict]
) -> None:
    f4 = reports["F4"]
    assert f4["passed"], f4
    assert f4["integrity_errors"] == []
    assert f4["checks"] == {
        "two_images_generated_and_viewed": True,
        "publishes_two_images": True,
        "publishes_review_of_generated_images": True,
        "states_scores_and_recommendation": True,
    }
    directory = f4["directory"]
    assert {path.name for path in directory.glob("segment_*")} == {
        "segment_1.png",
        "segment_2.png",
        "segment_3.txt",
    }
    trace = directory / "trace"
    assert (trace / "plan.json").exists()
    assert len((trace / "planner-results.jsonl").read_text().splitlines()) == 1
    events = [
        json.loads(line)["event"]
        for line in (trace / "events.jsonl").read_text().splitlines()
    ]
    assert events[0] == "planner_start" and events.count("dispatch") == 3


@pytest.mark.parametrize("case_id", ["F7", "F9"])
def test_a_voice_request_is_transcribed_once_before_planning(
    reports: dict[str, dict], case_id: str
) -> None:
    report = reports[case_id]
    assert report["passed"], report
    assert report["checks"] == {
        "audio_only_client_input": True,
        "one_pre_transcription": True,
        "two_images_generated_and_viewed": True,
        "publishes_two_images": True,
        "publishes_review_of_generated_images": True,
        "states_scores_and_recommendation": True,
    }


def test_two_images_without_their_scores_fail(reports: dict[str, dict]) -> None:
    """F4's steps all ran, but only the two images reached the client."""
    case = CASES["F4"]
    trace = reports["F4"]["directory"] / "trace"
    events = [
        json.loads(line) for line in (trace / "events.jsonl").read_text().splitlines()
    ]
    request = json.loads((reports["F4"]["directory"] / "request.json").read_text())
    dispatched = [event for event in events if event["event"] == "dispatch"]
    checks = run_cases.contract_checks(
        case.checks,
        request,
        events,
        dispatched,
        ["image", "image"],
        ["cat_0", "cat_1"],
        case.scoring,
        run_cases.read_scored_answer("", case.scoring),
    )
    assert checks == {
        "two_images_generated_and_viewed": True,
        "publishes_two_images": True,
        "publishes_review_of_generated_images": False,
        "states_scores_and_recommendation": False,
    }


@pytest.mark.parametrize(
    ("answer", "scores", "recommended_image"),
    [
        (
            "**Photo 1 Score:**\n- **Clarity:** 9/10\n- **Subject Completeness:** 8/10\n"
            "- **Composition:** 8/10\n\n**Photo 2 Score:**\n"
            "- **Clarity:** 8/10 — softer than Photo 1\n"
            "- **Subject Completeness:** 9/10 — but the composition is busy\n"
            "- **Composition:** 9/10\n\n**Recommendation: Photo 2**\n"
            "Photo 2 wins on completeness (9/10 vs. 8/10).",
            {"1": (9, 8, 8), "2": (8, 9, 9)},
            "2",
        ),
        (
            "照片1评分：\n- 清晰度：9分\n- 主体完整性：9分\n- 构图：8分\n\n"
            "照片2评分：\n- 清晰度：6分\n- 主体完整性：7分\n- 构图：7分\n\n"
            "推荐照片1，理由：照片1在清晰度上明显更高。",
            {"1": (9, 9, 8), "2": (6, 7, 7)},
            "1",
        ),
        (
            "**Image step1:** Clarity: 9/10, Subject Completeness: 8/10, "
            "Composition: 8/10\n**Image step2:** Clarity: 9/10, "
            "Subject Completeness: 9/10, Composition: 8/10\n"
            "**Recommendation: Image step2**",
            {"step1": (9, 8, 8), "step2": (9, 9, 8)},
            "step2",
        ),
        (
            "**Photo ID: step1**\n- **Clarity**: 9/10\n- **Subject Completeness**: 8/10\n"
            "- **Composition**: 8/10\n\n**Photo ID: step2**\n- **Clarity**: 10/10\n"
            "- **Subject Completeness**: 9/10\n- **Composition**: 10/10\n\n"
            "**Recommendation**: Photo ID: step2",
            {"step1": (9, 8, 8), "step2": (10, 9, 10)},
            "step2",
        ),
    ],
)
def test_a_scored_answer_is_read_per_image_and_criterion(
    answer: str, scores: dict[str, tuple[int, int, int]], recommended_image: str
) -> None:
    scoring = CASES["F4"].scoring
    scored_answer = run_cases.read_scored_answer(answer, scoring)
    assert scored_answer.scores == {
        image: dict(zip(scoring.criteria, values)) for image, values in scores.items()
    }
    assert scored_answer.recommended_image == recommended_image
    assert scored_answer.is_complete(scoring)


@pytest.mark.parametrize(
    "answer",
    [
        "Photo 1: clarity 9, subject completeness 8, composition 8\n"
        "Photo 2: clarity 7, subject completeness 9, composition 9",
        "Photo 1: clarity 9, subject completeness 8\n"
        "Photo 2: clarity 7, subject completeness 9, composition 9\n"
        "I recommend Photo 2.",
        "Photo 1: clarity 9, subject completeness 8, composition 8\n"
        "I recommend Photo 1.",
        "Both photos look great. I recommend Photo 1.",
    ],
)
def test_an_answer_missing_a_score_or_the_recommendation_is_incomplete(
    answer: str,
) -> None:
    scoring = CASES["F4"].scoring
    assert not run_cases.read_scored_answer(answer, scoring).is_complete(scoring)


def test_a_plan_that_breaks_the_request_fails_its_check(
    reports: dict[str, dict]
) -> None:
    """F6 asks for a description only, but the scripted planner also generates."""
    f6 = reports["F6"]
    assert not f6["passed"]
    assert f6["checks"]["only_visual_response"] is False
    assert f6["integrity_errors"] == []


def test_a_tampered_artifact_is_an_integrity_error(reports: dict[str, dict]) -> None:
    trace = reports["F4"]["directory"] / "trace"
    events = [
        json.loads(line) for line in (trace / "events.jsonl").read_text().splitlines()
    ]
    plan = json.loads((trace / "plan.json").read_text())
    artifacts = {
        path.stem: json.loads(path.read_text())
        for path in (trace / "artifacts").glob("*.json")
    }
    artifacts["verdict"]["value"] = "Edited after the fact"
    errors, _ = run_cases.check_integrity(events, plan, artifacts, {})
    assert errors == ["artifact verdict does not match its own hash"]


def test_repeated_rounds_each_keep_their_own_results(tmp_path: Path) -> None:
    case_file = run_cases.load_cases(run_cases.DEFAULT_CASES)
    with TestClient(create_app(scripted_config(tmp_path / "trace"))) as client:
        reports = run_cases.run_rounds(
            client, case_file, tmp_path / "trace", tmp_path / "results", {"F4"}, 2
        )
    assert [report.passed for report in reports] == [True, True]
    assert reports[0].execution_id != reports[1].execution_id
    for round_number in (1, 2):
        summary = tmp_path / "results" / f"round-{round_number}" / "summary.json"
        assert [entry["case"] for entry in json.loads(summary.read_text())] == ["F4"]
