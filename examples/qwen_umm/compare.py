# SPDX-License-Identifier: Apache-2.0
"""Submit one natural-language task, with optional image and audio, to a Qwen UMM demo."""

import argparse
import base64
import json
import mimetypes
import time
from pathlib import Path

import httpx

from sglang_omni.client.image_media import InlineMedia

SEGMENT_SUFFIXES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "video/mp4": ".mp4",
    "audio/wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/flac": ".flac",
}
INPUT_AUDIO_FORMATS = {".wav": "wav", ".mp3": "mp3"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="qwen_umm_demo")
    parser.add_argument(
        "--prompt",
        default=(
            "Generate two photorealistic photographs of a fluffy orange tabby kitten "
            "on a wooden windowsill, with both front paws visible, soft morning light "
            "and detailed fur. Then inspect both actual images, score each out of 10, "
            "and explain in Chinese which photograph is better using visible details."
        ),
    )
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--audio", type=Path, action="append", default=[])
    parser.add_argument("--mode", choices=["plan", "iterative"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    content: list[dict[str, str | dict[str, str]]] = [
        {"type": "text", "text": arguments.prompt}
    ]
    for path in arguments.image:
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
            }
        )
    for path in arguments.audio:
        content.append(
            {
                "type": "input_audio",
                "input_audio": {
                    "data": base64.b64encode(path.read_bytes()).decode("ascii"),
                    "format": INPUT_AUDIO_FORMATS[path.suffix.lower()],
                },
            }
        )
    options: dict[str, int | str] = {
        "width": 512,
        "height": 512,
        "steps": arguments.steps,
    }
    if arguments.mode is not None:
        options["mode"] = arguments.mode
    else:
        pass
    request = {
        "model": arguments.model,
        "messages": [{"role": "user", "content": content}],
        "modalities": ["text", "image", "audio"],
        "stream": False,
        "max_completion_tokens": 1024,
        "seed": arguments.seed,
        "stage_params": {"orchestrator": options},
    }
    (arguments.output_dir / "request.json").write_text(
        json.dumps(request, ensure_ascii=False, indent=2)
    )
    started_seconds = time.monotonic()
    with httpx.Client(timeout=1800, trust_env=False) as client:
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
    media_hashes: list[str] = []
    for index, segment in enumerate(result["segments"], start=1):
        if segment["kind"] == "text":
            if not isinstance(segment["data"], str) or not segment["data"].strip():
                raise ValueError("Expected a nonempty text segment")
            else:
                (arguments.output_dir / f"segment_{index}.txt").write_text(
                    segment["data"]
                )
        else:
            media = InlineMedia.model_validate(segment["data"])
            media_hashes.append(media.sha256)
            (
                arguments.output_dir
                / f"segment_{index}{SEGMENT_SUFFIXES[media.mime_type]}"
            ).write_bytes(base64.b64decode(media.encoded_content(), validate=True))
    report = {
        "endpoint": "/v1/chat/completions",
        "http_status": response.status_code,
        "response_id": response.json()["id"],
        "execution_id": result["segments"][0]["execution_id"],
        "elapsed_seconds": round(time.monotonic() - started_seconds, 3),
        "segment_kinds": [segment["kind"] for segment in result["segments"]],
        "media_sha256": media_hashes,
        "text": result.get("content", ""),
    }
    (arguments.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
else:
    pass
