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


@pytest.fixture(scope="module")
def reports(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict]:
    """F4 and F6 against a planner that always plans two images and a comparison."""
    root = tmp_path_factory.mktemp("cases")
    case_file = run_cases.load_cases(run_cases.DEFAULT_CASES)
    with TestClient(create_app(scripted_config(root / "trace"))) as client:
        run_cases.run_cases(
            client, case_file, root / "trace", root / "results", {"F4", "F6"}
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
