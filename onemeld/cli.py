# SPDX-License-Identifier: Apache-2.0
"""OneMeld's serving command: onemeld serve --config profile.yaml."""

import logging
from argparse import ArgumentParser

import uvicorn
from pydantic import ValidationError

from onemeld.config import load_config
from onemeld.server import create_app


def main() -> None:
    parser = ArgumentParser(prog="onemeld")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Serve one OneMeld profile.")
    serve.add_argument(
        "--config", required=True, help="Path to a OneMeld YAML profile."
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--model-name", help="Public model name; overrides the profile.")
    serve.add_argument("--log-level", default="info")
    options = parser.parse_args()
    try:
        config = load_config(options.config)
    except (OSError, ValidationError) as error:
        parser.error(str(error))
    if options.model_name:
        config = config.model_copy(update={"model_name": options.model_name})
    else:
        pass
    logging.basicConfig(
        level=options.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(
        create_app(config),
        host=options.host,
        port=options.port,
        log_level=options.log_level,
    )


if __name__ == "__main__":
    main()
