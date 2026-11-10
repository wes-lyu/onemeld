# SPDX-License-Identifier: Apache-2.0
"""OneMeld's local serving command."""

from argparse import ArgumentParser

import sglang_omni.serve as serving
from sglang_omni.config.resolver import ConfigResolver
from sglang_omni.config.sources import sources_from_config_file
from sglang_omni.models.qwen_umm.config import OneMeldPipelineConfig


def main() -> None:
    parser = ArgumentParser(prog="onemeld")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Serve one OneMeld pipeline.")
    serve.add_argument(
        "--config", required=True, help="Path to a OneMeld YAML profile."
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument(
        "--model-name", help="Public model name. Defaults to the profile name."
    )
    serve.add_argument("--log-level", default="info")
    options = parser.parse_args()
    try:
        base_config, patches = sources_from_config_file(
            options.config, OneMeldPipelineConfig
        )
        config = ConfigResolver(base_config).resolve(patches).config
    except ValueError as error:
        parser.error(str(error))
    serving.launch_server(
        config,
        host=options.host,
        port=options.port,
        model_name=options.model_name,
        log_level=options.log_level,
    )
