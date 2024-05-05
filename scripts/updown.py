import argparse
import os
import re
import sys


def output(*args, replace_list=None, suppress_list=None, file=None, end="\n"):
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


def command_regexp(command):
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
    return rf"^\[\s*{command}\s+(\w+)\s*\]\s*:\s*(.*)"


def updown_processor(
    filename, predicate_list, replace_list, suppress_list
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
    output("set -eou pipefail")
    for i, line in enumerate(lines):
        shell = command_regexp("shell")
        prefix = command_regexp("prefix")
        skip = command_regexp("skip")
        end = command_regexp("end")
        if match := re.search(shell, line):
            # Extract the matched groups
            predicate = match.group(1)
            trailing_command = match.group(2)
            if predicate in predicate_list:
                output(
                    trailing_command,
                    replace_list=replace_list,
                    suppress_list=suppress_list,
                )
        elif match := re.search(prefix, line):
            # Extract the matched groups
            predicate = match.group(1)
            trailing_command = match.group(2)
            if predicate in predicate_list:
                output(
                    trailing_command,
                    end="",
                    replace_list=replace_list,
                    suppress_list=suppress_list,
                )
        elif match := re.search(skip, line):
            # Extract the matched groups
            predicate = match.group(1)
            trailing_command = match.group(2)
            if predicate in predicate_list:
                if trailing_command == "begin":
                    output(
                        "if false; then",
                        replace_list=replace_list,
                        suppress_list=suppress_list,
                    )
                elif trailing_command == "end":
                    output(
                        "fi",
                        replace_list=replace_list,
                        suppress_list=suppress_list,
                    )
                else:
                    output(f"echo 'error in line {i} of README.md'\nexit 1;")
                    exit(1)
        elif match := re.search(end, line):
            # Extract the matched groups
            predicate = match.group(1)
            trailing_command = match.group(2)
            if predicate in predicate_list:
                output(
                    "exit 0",
                    replace_list=replace_list,
                    suppress_list=suppress_list,
                )
        elif line.startswith("```"):
            print_flag = not print_flag
        elif print_flag:
            output(line, end="", suppress_list=suppress_list)

    output("echo 'reached end of file without exit command'\nexit 1;")


# Initialize the ArgumentParser object
parser = argparse.ArgumentParser()
# Add arguments
parser.add_argument("-f", "--filename", help="Input filename", default="README.md")
parser.add_argument("-p", "--predicate", help="Input predicates", default="")
parser.add_argument("-r", "--replace", help="Input replace pairs", default="")
parser.add_argument("-s", "--suppress", help="Input suppress strings", default="")
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
    [tuple(pair.split(":")) for pair in args.replace.split(",")] if args.replace else []
)
# Get suppress strings, split by comma
suppress_list = args.suppress.split(",") if args.suppress else []
# Call updown_processor function
updown_processor(
    filename, predicate_list, replace_list, suppress_list
)
