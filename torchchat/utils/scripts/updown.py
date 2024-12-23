# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import os
import re

skip_nesting_level = 0

###############################################################################################
###
### print, with the ability to replace strings, or suppress lines
###


def output(*args, **kwargs):
    """
    Prints the given arguments after performing replacements and suppressions.
    Args:
        *args: The values to print.
        replace_list (list, optional): A list of tuples where the first element of each tuple is replaced with the second element in the output. Defaults to None.
        suppress_list (list, optional): A list of strings. If any string in this list is found in the output, the line is not printed. Defaults to None.
        file (file, optional): A file-like object (stream). Defaults to the current sys.stdout.
        end (str, optional): Specifies what to print at the end. Defaults to '\n'.
    Returns:
        None
    """
    # expand kwargs, make it error to not specify replace list or suppress list
    # because we should always have these for updown processor
    replace_list = kwargs["replace_list"]  # .get("replace_list", None)
    suppress_list = kwargs["suppress_list"]  # get("suppress_list", None)
    file = kwargs.get("file", None)
    end = kwargs.get("end", "\n")

    # Convert args to a list of strings
    str_args = [str(arg) for arg in args]
    # If replace_list is provided
    if replace_list:
        # For each tuple in replace_list
        for before, after in replace_list:
            # Replace all occurrences of 'before' with 'after' in each string
            str_args = [arg.replace(before, after) for arg in str_args]
    # If suppress_list is provided
    if suppress_list:
        # For each string in suppress_list
        for suppress in suppress_list:
            # If the suppress string is found in any of the str_args, return without printing
            if any(suppress in arg for arg in str_args):
                return
    # Print the modified strings
    print(*str_args, file=file, end=end)


###############################################################################################
###
### processing logic for optional argument in lines
###


def select_first_option_between_brackets(text):
    return re.sub(r"\[([^]|]*?)\|[^]]*]", r"\1", text)


def select_last_option_between_brackets(text):
    return re.sub(r"\[[^]]*\|([^]|]*)\]", r"\1", text)


def remove_text_between_brackets(text):
    return re.sub(r"\[.*?\]", "", text)


def extract_text_between_brackets(text):
    return re.findall(r"\[(.*?)\]", text)


def specialize_option(text, replacement):
    return re.sub(r"\[.*?\]", replacement, text)


###############################################################################################
###
### process a line either suppressing or expanding options
###


def updown_process_line(
    line, lineno, filename, replace_list, suppress_list, expand_options
):
    if not expand_options:
        # [ x1 | c2 | x3 ] means "pick one", so we may have to check that and pick one
        # of the options.  Probably pick the last option because testing has more likely
        # been performed with the first option!
        last = True
        if last:
            line = select_last_option_between_brackets(line)
        else:
            line = select_first_option_between_brackets(line)

        output(
            remove_text_between_brackets(line),
            replace_list=replace_list,
            suppress_list=suppress_list,
        )
    else:
        options = extract_text_between_brackets(line)
        if len(options) == 0:
            output(
                line,
                replace_list=replace_list,
                suppress_list=suppress_list,
            )
            return
        if len(options) > 1:
            output(
                "echo 'cross product of options not yet supported in line {line} of {filename}'\nexit 1",
                suppress_list=None,
                replace_list=None,
            )
            exit(1)
        for option in options[0].split("|"):
            output(
                specialize_option(line, option),
                replace_list=replace_list,
                suppress_list=suppress_list,
            )


###############################################################################################
###
### process an updown command
###


def process_command(
    line, lineno, filename, predicate_list, replace_list, suppress_list, create_sections
) -> bool:
    
    global skip_nesting_level
    command = r"^\[\s*(\w+)\s+(\w+)\s*\]\s*:\s*(.*)"
    match = re.search(command, line)

    if not match:
        # We have not processed this line as a command
        return False

    keyword = match.group(1)
    predicate = match.group(2)
    trailing_command = match.group(3)

    if predicate not in predicate_list:
        # We have processed this line as a command
        return True

    if keyword == "shell":
        output(
            trailing_command,
            replace_list=replace_list,
            suppress_list=suppress_list,
        )
    elif keyword == "prefix":
        output(
            trailing_command,
            end="",
            replace_list=replace_list,
            suppress_list=suppress_list,
        )
    elif keyword == "skip":
        if trailing_command == "begin":
            skip_nesting_level += 1
            if skip_nesting_level > 1:
                output(
                    "echo 'nested skips are discouraged in line {lineno} of {filename} and may be prohibited in the future'",
                    replace_list=replace_list,
                    suppress_list=suppress_list,
                )
            output(
                "if false; then",
                replace_list=replace_list,
                suppress_list=suppress_list,
            )
        elif trailing_command == "end":
            skip_nesting_level -= 1
            if skip_nesting_level < 0:
                output(
                    "echo 'skip end without matching skip begin in line {lineno} of {filename}'\nexit 1;",
                    replace_list=replace_list,
                    suppress_list=suppress_list,
                )
            output(
                "fi",
                replace_list=replace_list,
                suppress_list=suppress_list,
            )
        else:
            output(
                f"echo 'error in line {lineno} of {filename}'\nexit 1;",
                suppress_list=None,
                replace_list=None,
            )
            exit(1)
    elif keyword == "end":
        if skip_nesting_level > 0:
            output(
                "echo 'skip begin without matching skip end in line {lineno} of {filename}'\nexit 1;",
                replace_list=replace_list,
                suppress_list=suppress_list,
            )
        if create_sections:
            output(
                "echo '::endgroup::'",
                suppress_list=None,
                replace_list=None,
            )
        output(
            "exit 0",
            replace_list=replace_list,
            suppress_list=suppress_list,
        )
        exit(0)
    elif keyword == "comment":
        output(
            "# " + trailing_command,
            suppress_list=None,
            replace_list=None,
        )
    else:
        output(
            "echo 'unknown updown command'\nexit 1",
            suppress_list=None,
            replace_list=None,
        )
        exit(1)

    # We have processed this line as a command
    return True


###############################################################################################
###
### updown processing of the input file
###


def updown_processor(
    filename,
    predicate_list,
    replace_list,
    suppress_list,
    expand_options,
    create_sections,
):
    """
    Processes a file based on the given predicates, replacements, and suppressions.
    Args:
        filename (str): The name of the file to process.
        predicate_list (list): A list of predicates to match in the file.
        replace_list (list): A list of tuples where the first element of each tuple is replaced with the second element in the output.
        suppress_list (list): A list of strings. If any string in this list is found in the output, the line is not printed.
    Returns:
        None
    """
    with open(filename, "r") as file:
        lines = file.readlines()
    print_flag = False

    # Use bash; set it to fail on the first failing command
    output("#! /bin/bash", replace_list=None, suppress_list=None)
    output("set -eou pipefail", replace_list=None, suppress_list=None)

    if create_sections:
        output(
            "echo '::group::start-of-document'",
            suppress_list=None,
            replace_list=None,
        )
    for lineno, line in enumerate(lines):
        # clip trailing newline
        if line.endswith("\n"):
            line = line[:-1]

        if process_command(
            line=line,
            lineno=lineno,
            filename=filename,
            predicate_list=predicate_list,
            replace_list=replace_list,
            suppress_list=suppress_list,
            create_sections=create_sections,
        ):
            pass
        elif re.search(r"^\s*```", line):
            print_flag = not print_flag
        elif (match := re.search(r"^#\s+([\w\s]+)", line)) and create_sections:
            output(
                f"echo '::endgroup::'",
                suppress_list=None,
                replace_list=None,
            )
            output(
                f"echo '::group::{match.group(0)}'",
                suppress_list=None,
                replace_list=None,
            )
        elif (match := re.search(r"^##\s+([\w\s]+)", line)) and create_sections:
            output(
                f"echo '::endgroup::'",
                suppress_list=None,
                replace_list=None,
            )
            output(
                f"echo '::group::{match.group(0)}'",
                suppress_list=None,
                replace_list=None,
            )
        elif print_flag:
            updown_process_line(
                line=line,
                lineno=lineno,
                filename=filename,
                expand_options=expand_options,
                replace_list=replace_list,
                suppress_list=suppress_list,
            )

    output(
        "echo 'reached end of file without `end` command'\nexit 1;",
        suppress_list=None,
        replace_list=None,
    )


###############################################################################################
###
### updown processing of the input file
###


def main():
    # Initialize the ArgumentParser object
    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument("-f", "--filename", help="Input filename", default="README.md")
    parser.add_argument("-p", "--predicate", help="Input predicates", default="")
    parser.add_argument("-r", "--replace", help="Input replace pairs", default="")
    parser.add_argument("-s", "--suppress", help="Input suppress strings", default="")
    parser.add_argument(
        "-e", "--expand-options", action="store_true", help="Expand options flag"
    )
    parser.add_argument(
        "-g", "--create-sections", action="store_true", help="Expand options flag"
    )
    args = parser.parse_args()
    # Get filename
    filename = args.filename
    # Check if file exists
    if not os.path.isfile(filename):
        output(f"echo 'File {filename} does not exist.'\n exit 1;")
        exit(1)
    # Get predicates, split by comma, and add "default"
    predicate_list = args.predicate.split(",") if args.predicate else []
    predicate_list.append("default")
    # Get replace pairs, split by comma, and turn into a list of tuples
    replace_list = (
        [tuple(pair.split(":")) for pair in args.replace.split(",")]
        if args.replace
        else []
    )
    # Get suppress strings, split by comma
    suppress_list = args.suppress.split(",") if args.suppress else []
    # Call updown_processor function
    updown_processor(
        filename,
        predicate_list,
        replace_list,
        suppress_list,
        args.expand_options,
        args.create_sections,
    )


if __name__ == "__main__":
    main()
