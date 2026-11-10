# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import pytest

from sglang_omni.profiler.torch_profiler import TorchProfiler


class StoppableProfiler:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


@pytest.fixture
def active_profiler(monkeypatch: pytest.MonkeyPatch) -> StoppableProfiler:
    existing = StoppableProfiler()
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setattr(TorchProfiler, "profiler", existing)
    monkeypatch.setattr(TorchProfiler, "active_run_id", "run-a")
    monkeypatch.setattr(TorchProfiler, "trace_template", "/traces/run-a")
    return existing


def test_start_with_the_active_run_id_returns_its_trace_path(
    active_profiler: StoppableProfiler,
) -> None:
    """Starting the run that is already active used to raise UnboundLocalError on rank."""
    trace_path = TorchProfiler.start("/traces/other", run_id="run-a")

    assert trace_path == "/traces/run-a_rank3.trace.json.gz"
    assert TorchProfiler.profiler is active_profiler
    assert not active_profiler.stopped


def test_start_with_a_new_run_id_restarts_the_profiler(
    active_profiler: StoppableProfiler, tmp_path: Path
) -> None:
    """Restarting an active profiler used to raise UnboundLocalError on rank."""
    trace_path = TorchProfiler.start(str(tmp_path / "run-b"), run_id="run-b")
    try:
        assert trace_path == f"{tmp_path / 'run-b'}_rank3.trace.json.gz"
        assert active_profiler.stopped
        assert TorchProfiler.active_run_id == "run-b"
        assert TorchProfiler.profiler is not active_profiler
    finally:
        TorchProfiler.profiler.stop()
