# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import torch

from torchchat.distributed.logging_utils import SingletonLogger
logger = SingletonLogger.get_logger()


try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

# this is used for pp placement
def string_list(raw_arg):
    return raw_arg.split(",")


class InferenceConfig:
    """
    A helper class to manage the inference configuration.
    Semantics:
    - Default config is loaded from a toml file. If no toml file is provided,
    then the default config is loaded from argparse defaults.
    - if toml file has missing keys, they are filled with argparse defaults.
    - if additional explicit cmd args are provided in addition to the toml
    file, they will override the toml config and the argparse defaults

    precedence order: cmdline > toml > argparse default

    Arg parsing semantics:

    Each argument starts with <prefix>_ which is the section name in the toml file
    followed by name of the option in the toml file. For ex,
    model.name translates to:
        [model]
        name
    in the toml file
    """

    def __init__(self):
        # main parser
        self.parser = argparse.ArgumentParser(description="torchchat arg parser.")

    def parse_args(self, config_file):

        args_dict = defaultdict(defaultdict)
        local_path = "inference_configs/"+ config_file
        full_path = os.path.join(os.getcwd(), local_path)
        file_path = Path(full_path)

        logger.info(f"Loading config file {config_file}")

        if not file_path.is_file():
            raise FileNotFoundError(f"Config file {full_path} does not exist")

        try:
            with open(file_path, "rb") as f:
                for k, v in tomllib.load(f).items():
                    # to prevent overwrite of non-specified keys
                    print(f"{k} {v}")
                    args_dict[k] |= v
        except (FileNotFoundError, tomllib.TOMLDecodeError) as e:
            logger.exception(
                f"Error while loading the configuration file: {config_file}"
            )
            logger.exception(f"Error details: {str(e)}")
            raise e

        for k, v in args_dict.items():
            class_type = type(k.title(), (), v)
            setattr(self, k, class_type())


    def _args_to_two_level_dict(self, args: argparse.Namespace) -> defaultdict:
        args_dict = defaultdict(defaultdict)
        for k, v in vars(args).items():
            first_level_key, second_level_key = k.split(".", 1)
            args_dict[first_level_key][second_level_key] = v
        return args_dict

    def _validate_config(self) -> bool:
        # TODO: Add more mandatory validations
        assert self.model.name and self.model.flavor and self.model.tokenizer_path
        return True

    def parse_args_from_command_line(
        self, args_list
    ) -> Tuple[argparse.Namespace, argparse.Namespace]:
        """
        Parse command line arguments and return the parsed args and the command line only args
        """
        args = self.parser.parse_args(args_list)

        # aux parser to parse the command line only args, with no defaults from main parser
        aux_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
        for arg, val in vars(args).items():
            if isinstance(val, bool):
                aux_parser.add_argument(
                    "--" + arg, action="store_true" if val else "store_false"
                )
            elif arg == "inference.pipeline_parallel_split_points":
                # without this special case, type inference breaks here,
                # since the inferred type is just 'list' and it ends up flattening
                # e.g. from ["layers.0", "layers.1"] into ["l", "a", "y", "e", "r", "s", ".0", ...]
                aux_parser.add_argument("--" + arg, type=string_list)
            else:
                aux_parser.add_argument("--" + arg, type=type(val))

        cmd_args, _ = aux_parser.parse_known_args(args_list)

        return args, cmd_args
