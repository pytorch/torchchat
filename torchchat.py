# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse

from cli import (
    add_arguments_for_eval,
    add_arguments_for_export,
    add_arguments_for_generate,
    arg_init,
    check_args,
)

default_device = "cpu"  # 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Top-level command")
    subparsers = parser.add_subparsers(
        dest="subcommand",
        help="Use `generate`, `eval` or `export` followed by subcommand specific options.",
    )

    parser_generate = subparsers.add_parser("generate")
    add_arguments_for_generate(parser_generate)

    parser_eval = subparsers.add_parser("eval")
    add_arguments_for_eval(parser_eval)

    parser_export = subparsers.add_parser("export")
    add_arguments_for_export(parser_export)

    args = parser.parse_args()
    args = arg_init(args)

    if args.subcommand == "generate":
        check_args(args, "generate")
        from generate import main as generate_main

        generate_main(args)
    elif args.subcommand == "eval":
        from eval import main as eval_main

        eval_main(args)
    elif args.subcommand == "export":
        check_args(args, "export")
        from export import main as export_main

        export_main(args)
    else:
        raise RuntimeError("Must specify valid subcommands: generate, export, eval")
