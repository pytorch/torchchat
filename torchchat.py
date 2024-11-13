# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import signal
import sys

# MPS ops missing with Multimodal torchtune
# https://github.com/pytorch/torchtune/issues/1723
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from torchchat.cli.cli import (
    add_arguments_for_verb,
    arg_init,
    check_args,
    INVENTORY_VERBS,
    KNOWN_VERBS,
)

default_device = "cpu"


def signal_handler(sig, frame):
    print("\nInterrupted by user. Bye!\n")
    sys.exit(0)


if __name__ == "__main__":
    # Set the signal handler for SIGINT
    signal.signal(signal.SIGINT, signal_handler)

    # Initialize the top-level parser
    parser = argparse.ArgumentParser(
        prog="torchchat",
        add_help=True,
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="The specific command to run",
    )
    subparsers.required = True

    VERB_HELP = {
        "chat": "Chat interactively with a model via the CLI",
        "generate": "Generate responses from a model given a prompt",
        "browser": "Chat interactively with a model in a locally hosted browser",
        "export": "Export a model artifact to AOT Inductor or ExecuTorch",
        "download": "Download model artifacts",
        "list": "List all supported models",
        "remove": "Remove downloaded model artifacts",
        "where": "Return directory containing downloaded model artifacts",
        "server": "[WIP] Starts a locally hosted REST server for model interaction",
        "eval": "Evaluate a model via lm-eval",
    }
    for verb, description in VERB_HELP.items():
        subparser = subparsers.add_parser(verb, help=description)
        add_arguments_for_verb(subparser, verb)

    # Now parse the arguments
    args = parser.parse_args()

    # Don't initialize for Inventory management subcommands
    # TODO: Remove when arg_init is refactored
    if args.command not in INVENTORY_VERBS:
        args = arg_init(args)
    logging.basicConfig(
        format="%(message)s", level=logging.DEBUG if args.verbose else logging.INFO
    )

    if args.command == "chat":
        # enable "chat"
        args.chat = True
        check_args(args, "chat")
        from generate import main as generate_main

        generate_main(args)
    elif args.command == "browser":
        print(
            "\nTo test out the browser please use: streamlit run torchchat/usages/browser.py <args>\n"
        )
    elif args.command == "server":
        check_args(args, "server")
        from torchchat.usages.server import main as server_main

        server_main(args)
    elif args.command == "generate":
        check_args(args, "generate")
        from torchchat.generate import main as generate_main

        generate_main(args)
    elif args.command == "eval":
        from torchchat.usages.eval import main as eval_main

        eval_main(args)
    elif args.command == "export":
        check_args(args, "export")
        from torchchat.export import main as export_main

        export_main(args)
    elif args.command == "download":
        check_args(args, "download")
        from torchchat.cli.download import download_main

        download_main(args)
    elif args.command == "list":
        check_args(args, "list")
        from torchchat.cli.download import list_main

        list_main(args)
    elif args.command == "where":
        check_args(args, "where")
        from torchchat.cli.download import where_main

        where_main(args)
    elif args.command == "remove":
        check_args(args, "remove")
        from torchchat.cli.download import remove_main

        remove_main(args)
    else:
        parser.print_help()
