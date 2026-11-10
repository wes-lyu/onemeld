# SPDX-License-Identifier: Apache-2.0
"""Messages between the orchestrator and a backend worker process.

The orchestrator binds one ZMQ PAIR socket per worker and the worker connects
to it. Every frame is one msgpack map with a kind:

- worker to orchestrator: ready, failed {error}, result {id, ok, result | error}, closed
- orchestrator to worker: run {id, request {inputs, params}}, cancel {id}, shutdown
"""

import msgpack
from pydantic import JsonValue

from onemeld.composition.types import StepRequest

WireValue = JsonValue | bytes
WireMessage = dict[str, WireValue]


def pack(message: WireMessage) -> bytes:
    return msgpack.packb(message, use_bin_type=True)


def unpack(frame: bytes) -> WireMessage:
    message = msgpack.unpackb(frame, raw=False)
    if not isinstance(message, dict) or not isinstance(message.get("kind"), str):
        raise ValueError(f"Malformed worker message {message!r:.200}")
    else:
        return message


def pack_request(request: StepRequest) -> WireMessage:
    return {"inputs": request.inputs, "params": request.params}


def unpack_request(value: object) -> StepRequest:
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("inputs"), (str, bytes, list))
        or not isinstance(value.get("params"), dict)
    ):
        raise ValueError(f"Malformed step request {value!r:.200}")
    else:
        return StepRequest(inputs=value["inputs"], params=value["params"])
