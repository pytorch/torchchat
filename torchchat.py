# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import subprocess
import sys

from cli import (
    add_arguments_for_verb,
    arg_init,
    check_args,
    INVENTORY_VERBS,
    KNOWN_VERBS,
)

default_device = "cpu"


if __name__ == "__main__":
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
        # enable "chat" and "gui" when entering "browser"
        args.chat = True
        args.gui = True
        check_args(args, "browser")

        from browser.browser import main as browser_main

        browser_main(args)
    elif args.command == "server":
        check_args(args, "server")
        from server import main as server_main

        server_main(args)
    elif args.command == "generate":
        check_args(args, "generate")
        from generate import main as generate_main

        generate_main(args)
    elif args.command == "eval":
        from eval import main as eval_main

        eval_main(args)
    elif args.command == "export":
        check_args(args, "export")
        from export import main as export_main

        export_main(args)
    elif args.command == "download":
        check_args(args, "download")
        from download import download_main

        download_main(args)
    elif args.command == "list":
        check_args(args, "list")
        from download import list_main

        list_main(args)
    elif args.command == "where":
        check_args(args, "where")
        from download import where_main

        where_main(args)
    elif args.command == "remove":
        check_args(args, "remove")
        from download import remove_main

        remove_main(args)
    else:
        parser.print_help()
