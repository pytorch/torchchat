
"""
This is the top level description of what generate does in torchchat 

This is essentially what the CLI in generate.py does
"""

# torchchat.py: Parse Args
import argparse
from cli import add_arguments_for_verb, arg_init

parser = argparse.ArgumentParser(
    prog="torchchat",
    add_help=True,
)
add_arguments_for_verb(parser, "generate")
args = parser.parse_args()
args = arg_init(args)


# torchchat.py: Generate
from generate import main as generate_main

generate_main(args)
