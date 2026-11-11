# SPDX-License-Identifier: Apache-2.0
"""Per-request trace files: the events, the plan, raw planner replies and step results.

Each execution writes to trace_dir/<execution_id>/:
- events.jsonl: one line per trace event, with timestamp_utc and monotonic_ns
- plan.json: the accepted plan
- planner-results.jsonl: each raw planner reply, before validation
- artifacts/<step_id>.json: each step result, media included as a data URL
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import JsonValue

from onemeld.composition.plan import UMMArtifact


class TraceRecorder:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def directory(self, execution_id: str) -> Path:
        directory = self.root / execution_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def record(
        self,
        execution_id: str,
        event: str,
        details: dict[str, JsonValue],
        artifact: UMMArtifact | None = None,
    ) -> None:
        directory = self.directory(execution_id)
        entry: dict[str, JsonValue] = {
            "event": event,
            "execution_id": execution_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "monotonic_ns": time.monotonic_ns(),
            **details,
        }
        if artifact is not None:
            artifacts = directory / "artifacts"
            artifacts.mkdir(exist_ok=True)
            saved = {
                "step_id": details["step_id"],
                "kind": artifact.kind,
                "action": artifact.action,
                "sha256": artifact.sha256,
                "truncated": artifact.truncated,
                "value": (
                    artifact.value
                    if isinstance(artifact.value, str)
                    else artifact.value.model_dump(mode="json")
                ),
            }
            (artifacts / f"{details['step_id']}.json").write_text(
                json.dumps(saved, ensure_ascii=False, indent=2)
            )
        elif event == "plan":
            (directory / "plan.json").write_text(
                json.dumps(details["reply"], ensure_ascii=False, indent=2)
            )
        else:
            pass
        with (directory / "events.jsonl").open("a", encoding="utf-8") as events:
            events.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def planner_result(self, execution_id: str, result: object) -> None:
        """Keep the planner's reply as it arrived, before it is validated."""
        with (self.directory(execution_id) / "planner-results.jsonl").open(
            "a", encoding="utf-8"
        ) as replies:
            replies.write(json.dumps(result, ensure_ascii=False) + "\n")
