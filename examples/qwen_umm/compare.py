# SPDX-License-Identifier: Apache-2.0
"""Submit one natural-language task to the plan-driven Qwen UMM demo."""

import argparse
import base64
import json
import time
from pathlib import Path

import httpx

from sglang_omni.client.image_media import InlineImage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        default=(
            "Generate two photorealistic photographs of a fluffy orange tabby kitten "
            "on a wooden windowsill, with both front paws visible, soft morning light "
            "and detailed fur. Then inspect both actual images, score each out of 10, "
            "and explain in Chinese which photograph is better using visible details."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    request = {
        "model": "qwen_umm_demo",
        "messages": [{"role": "user", "content": arguments.prompt}],
        "modalities": ["text", "image"],
        "stream": False,
        "max_completion_tokens": 1024,
        "seed": arguments.seed,
        "stage_params": {
            "orchestrator": {
                "width": 512,
                "height": 512,
                "steps": arguments.steps,
            }
        },
    }
    (arguments.output_dir / "request.json").write_text(
        json.dumps(request, ensure_ascii=False, indent=2)
    )
    started_seconds = time.monotonic()
    with httpx.Client(timeout=1200, trust_env=False) as client:
        response = client.post(
            f"{arguments.base_url.rstrip('/')}/v1/chat/completions", json=request
        )
    (arguments.output_dir / "response.json").write_text(response.text)
    response.raise_for_status()
    result = response.json()["choices"][0]["message"]
    if not result["segments"]:
        raise ValueError("Expected at least one output segment")
    else:
        pass
    image_hashes: list[str] = []
    for segment in result["segments"]:
        if segment["kind"] == "image":
            image = InlineImage.model_validate(segment["data"])
            image_hashes.append(image.sha256)
            (arguments.output_dir / f"image_{len(image_hashes)}.png").write_bytes(
                base64.b64decode(image.encoded_image(), validate=True)
            )
        elif (
            segment["kind"] == "text"
            and isinstance(segment["data"], str)
            and segment["data"].strip()
        ):
            pass
        else:
            raise ValueError("Expected valid image or nonempty text output")
    report = {
        "endpoint": "/v1/chat/completions",
        "http_status": response.status_code,
        "response_id": response.json()["id"],
        "execution_id": result["segments"][0]["execution_id"],
        "request": request,
        "elapsed_seconds": round(time.monotonic() - started_seconds, 3),
        "segment_kinds": [segment["kind"] for segment in result["segments"]],
        "image_sha256": image_hashes,
        "text": result.get("content", ""),
    }
    (arguments.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    (arguments.output_dir / "answer.txt").write_text(result.get("content", ""))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
else:
    pass
