# SPDX-License-Identifier: Apache-2.0
"""Request bounds, deadlines, failures and health of the orchestrator."""

import json

import pytest

from onemeld.composition.plan import UMMCompositionSpec
from onemeld.composition.types import (
    QueueFullError,
    UMMDeadlineError,
    UMMLimits,
    UMMRequestError,
)
from tests.unit_test.fixtures.orchestrator import Harness, limits_for, settle, task_of

pytest_plugins = ["tests.unit_test.fixtures.composition"]

TWO_PAINTS = json.dumps(
    {
        "steps": [
            {
                "id": name,
                "action": "paint",
                "instruction": "Draw",
                "input_refs": [],
                "publish": True,
            }
            for name in ("a", "b")
        ]
    }
)


class ManualClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.mark.asyncio
async def test_requests_beyond_max_active_are_refused(
    composition: UMMCompositionSpec,
) -> None:
    harness = Harness(composition)
    await harness.submit(task_of("First"))
    with pytest.raises(QueueFullError):
        async for _ in harness.orchestrator.stream(task_of("Second")):
            pass
    assert harness.held() == {"planning_worker"}


@pytest.mark.asyncio
async def test_deadline_fails_the_request_and_cancels_running_steps(
    composition: UMMCompositionSpec,
) -> None:
    """The next result after the deadline ends the request instead of advancing it."""
    clock = ManualClock()
    harness = Harness(composition)
    harness.orchestrator.clock = clock
    await harness.submit(task_of("Draw two"))
    await harness.reply("planning_worker", {"text": TWO_PAINTS})
    assert harness.held() == {"image_a", "image_b"}
    clock.now = 61.0
    await harness.reply("image_a", {"modality": "image", "image_b64": "eA=="})
    assert isinstance(harness.error(), UMMDeadlineError)
    assert harness.backends["image_b"].cancelled
    assert not harness.orchestrator.adapter.executions


@pytest.mark.asyncio
async def test_client_input_over_the_byte_budget_is_a_client_error(
    composition: UMMCompositionSpec,
) -> None:
    limits = limits_for(composition)
    harness = Harness(
        composition,
        UMMLimits(
            max_dispatches=limits.max_dispatches,
            max_outputs=limits.max_outputs,
            max_active=1,
            max_retained_bytes=1024,
            timeout_s=60,
        ),
    )
    await harness.submit(task_of("Draw"))
    error = harness.error()
    assert isinstance(error, UMMRequestError)
    assert "max_retained_bytes=1024" in str(error)
    assert harness.held() == set()


@pytest.mark.asyncio
async def test_failed_step_fails_the_request_and_cancels_its_siblings(
    composition: UMMCompositionSpec,
) -> None:
    harness = Harness(composition)
    await harness.submit(task_of("Draw two"))
    await harness.reply("planning_worker", {"text": TWO_PAINTS})
    ((_, future),) = harness.backends["image_a"].held.values()
    future.set_exception(RuntimeError("image worker exited"))
    await settle()
    assert "image worker exited" in str(harness.error())
    assert harness.backends["image_b"].cancelled
    assert not harness.orchestrator.active
    assert not harness.orchestrator.adapter.executions


def test_health_reports_each_backend(composition: UMMCompositionSpec) -> None:
    harness = Harness(composition)
    harness.backends["image_b"].healthy = False
    health = harness.orchestrator.health()
    assert health["image_b"] is False
    assert all(healthy for name, healthy in health.items() if name != "image_b")
