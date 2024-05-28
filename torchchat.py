# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import subprocess
import sys
import webbrowser as wb

from cli import (
    add_arguments_for_verb,
    KNOWN_VERBS,
    arg_init,
    check_args,
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
       "chat": "Chat interactively with a model",
       "browser": "Chat interactively in a browser",
       "download": "Download a model from Hugging Face or others",
       "generate": "Generate responses from a model given a prompt",
       "eval": "Evaluate a model given a prompt",
       "export": "Export a model for AOT Inductor or ExecuTorch",
       "list": "List supported models",
       "remove": "Remove downloaded model artifacts",
       "where": "Return directory containing downloaded model artifacts",
    }
    for verb in KNOWN_VERBS:
       subparser = subparsers.add_parser(verb, help=VERB_HELP[verb])
       add_arguments_for_verb(subparser, verb)

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

        subprocess.Popen("python -m http.server 8000", cwd="browser/build/", shell=True)

        wb.open_new_tab('http://localhost:8000')

        from generate import main as generate_main
        generate_main(args)







    elif args.command == "download":
        check_args(args, "download")
        from download import download_main

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
