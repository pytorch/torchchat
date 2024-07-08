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
    KNOWN_VERBS,
    arg_init,
    check_args,
)

default_device = "cpu"


if __name__ == "__main__":
    # Initialize the top-level parser
    parser = argparse.ArgumentParser(
        prog="torchchat",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="The specific command to run",
    )
    subparsers.required = True

    # Top Level Command List
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
    
    # Add subparsers for each verb
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
        # BuilderArgs, Speculative, Tokenizer, Generator, profile, quantize, draft_quantize, chat
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
    elif args.command == "generate":
        # BuilderArgs, TokenizerArgs, Speculative, Generator, profile, quantize, draft_quantize
        check_args(args, "generate")
        from generate import main as generate_main

        generate_main(args)
    elif args.command == "eval":
        # BuilderArgs, TokenizerArgs, quantize, tasks, limit, compile, max_seq_length
        from eval import main as eval_main

        eval_main(args)
    elif args.command == "export":
        # BuilderArgs, TokenizerArgs, quantize
        check_args(args, "export")
        from export import main as export_main

        export_main(args)
    elif args.command == "download":
        # model_directory, model, hf_token
        check_args(args, "download")
        from download import download_main

        download_main(args)
    elif args.command == "list":
        # model_directory
        check_args(args, "list")
        from download import list_main

        list_main(args)
    elif args.command == "where":
        # model_directory, model
        check_args(args, "where")
        from download import where_main

        where_main(args)
    elif args.command == "remove":
        # model_directory, model 
        check_args(args, "remove")
        from download import remove_main

        remove_main(args)
    else:
        parser.print_help()
