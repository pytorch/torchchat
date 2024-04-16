# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import subprocess
import sys

from cli import (
    add_arguments_for_download,
    add_arguments_for_eval,
    add_arguments_for_export,
    add_arguments_for_generate,
    add_arguments_for_browser,
    arg_init,
    check_args,
)

default_device = "cpu"  # 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Top-level command")
    subparsers = parser.add_subparsers(
        dest="subcommand",
        help="Use `download`, `generate`, `eval`, `export` or `browser` followed by subcommand specific options.",
    )

    parser_chat = subparsers.add_parser("chat")
    add_arguments_for_generate(parser_chat)

    parser_download = subparsers.add_parser("download")
    add_arguments_for_download(parser_download)

    parser_generate = subparsers.add_parser("generate")
    add_arguments_for_generate(parser_generate)

    parser_eval = subparsers.add_parser("eval")
    add_arguments_for_eval(parser_eval)

    parser_export = subparsers.add_parser("export")
    add_arguments_for_export(parser_export)

    parser_browser = subparsers.add_parser("browser")
    add_arguments_for_browser(parser_browser)

    args = parser.parse_args()
    args = arg_init(args)
    logging.basicConfig(
        format="%(message)s", level=logging.DEBUG if args.verbose else logging.INFO
    )

    if args.subcommand == "download":
        check_args(args, "download")
        from download import main as download_main

        download_main(args)
    elif args.subcommand == "generate" or args.subcommand == "chat":
        check_args(args, args.subcommand)
        from generate import main as generate_main

        generate_main(args)
    elif args.subcommand == "eval":
        from eval import main as eval_main

        eval_main(args)
    elif args.subcommand == "export":
        check_args(args, "export")
        from export import main as export_main

        export_main(args)
    elif args.subcommand == "browser":
        # TODO: add check_args()

        # Look for port from cmd args. Default to 5000 if not found.
        # The port args will be passed directly to the Flask app.
        port = 5000
        i = 2
        while i < len(sys.argv):
            # Check if the current argument is '--port'
            if sys.argv[i] == '--port':
                # Check if there's a value immediately following '--port'
                if i + 1 < len(sys.argv):
                    # Extract the value and remove '--port' and the value from sys.argv
                    port = sys.argv[i + 1]
                    del sys.argv[i:i+2]  # Delete '--port' and the value
                    break  # Exit loop since port is found
            else:
                i += 1

        # Assume the user wants "chat" when entering "browser". TODO: add support for "generate" as well
        args_plus_chat = ['"{}"'.format(s) for s in sys.argv[2:]] + ['"--chat"'] + ['"--num-samples"'] + ['"1000000"']
        formatted_args = ", ".join(args_plus_chat)
        command = ["flask", "--app", "chat_in_browser:create_app(" + formatted_args + ")", "run", "--port", f"{port}"]
        subprocess.run(command)
    else:
        raise RuntimeError("Must specify valid subcommands: generate, export, eval")
