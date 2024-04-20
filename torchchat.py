# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import subprocess
import sys

from cli import (
    add_arguments,
    add_arguments_for_browser,
    add_arguments_for_chat,
    add_arguments_for_download,
    add_arguments_for_eval,
    add_arguments_for_export,
    add_arguments_for_generate,
    arg_init,
    check_args,
)

# Prefer CUDA if available, otherwise MPS if available, otherwise CPU
default_device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'


if __name__ == "__main__":
    # Initialize the top-level parser
    parser = argparse.ArgumentParser(
        prog="torchchat",
        description="Welcome to the torchchat CLI!",
        add_help=True,
    )
    # Default command is to print help
    parser.set_defaults(func=lambda args: self._parser.print_help())

    add_arguments(parser)
    subparsers = parser.add_subparsers(
        dest="command",
        help="The specific command to run",
    )

    parser_chat = subparsers.add_parser(
        "chat",
        help="Chat interactively with a model",
    )
    add_arguments_for_chat(parser_chat)

    parser_browser = subparsers.add_parser(
        "browser",
        help="Chat interactively in a browser",
    )
    add_arguments_for_browser(parser_browser)

    parser_download = subparsers.add_parser(
        "download",
        help="Download a model from Hugging Face or others",
    )
    add_arguments_for_download(parser_download)

    parser_generate = subparsers.add_parser(
        "generate",
        help="Generate responses from a model given a prompt",
    )
    add_arguments_for_generate(parser_generate)

    parser_eval = subparsers.add_parser(
        "eval",
        help="Evaluate a model given a prompt",
    )
    add_arguments_for_eval(parser_eval)

    parser_export = subparsers.add_parser(
        "export",
        help="Export a model for AOT Inductor or ExecuTorch",
    )
    add_arguments_for_export(parser_export)

    # Move all flags to the front of sys.argv since we don't
    # want to use the subparser syntax
    flag_args = []
    positional_args = []
    i = 1
    while i < len(sys.argv):
        if sys.argv[i].startswith("-"):
            flag_args += sys.argv[i : i + 2]
            i += 2
        else:
            positional_args.append(sys.argv[i])
            i += 1
    sys.argv = sys.argv[:1] + flag_args + positional_args

    # Now parse the arguments
    args = parser.parse_args()
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

        # Look for port from cmd args. Default to 5000 if not found.
        # The port args will be passed directly to the Flask app.
        port = 5000
        i = 2
        while i < len(sys.argv):
            # Check if the current argument is '--port'
            if sys.argv[i] == "--port":
                # Check if there's a value immediately following '--port'
                if i + 1 < len(sys.argv):
                    # Extract the value and remove '--port' and the value from sys.argv
                    port = sys.argv[i + 1]
                    del sys.argv[i : i + 2]  # Delete '--port' and the value
                    break  # Exit loop since port is found
            else:
                i += 1

        # Construct arguments for the flask app minus 'browser' command
        # plus '--chat'
        args_plus_chat = ['"{}"'.format(s) for s in sys.argv[1:] if s != "browser"] + [
            '"--chat"'
        ]
        formatted_args = ", ".join(args_plus_chat)
        command = [
            "flask",
            "--app",
            "chat_in_browser:create_app(" + formatted_args + ")",
            "run",
            "--port",
            f"{port}",
        ]
        subprocess.run(command)
    elif args.command == "download":
        check_args(args, "download")
        from download import main as download_main

        download_main(args)
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
    else:
        parser.print_help()
