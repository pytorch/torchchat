import sys


def print_between_triple_backticks(filename, predicate):
    with open(filename, "r") as file:
        lines = file.readlines()
    print_flag = False
    for line in lines:
        command = f"[shell {predicate}]:"
        prefix = f"[prefix {predicate}]:"
        end = f"[end {predicate}]:"
        if line.startswith(prefix):
            print(line[len(prefix) : -1], end="")
        elif line.startswith(command):
            print(line[len(command) :])
        elif line.startswith(end):
            return
        elif line.startswith("```"):
            print_flag = not print_flag
        elif print_flag:
            print(line, end="")


if len(sys.argv) > 1:
    predicate = sys.argv[1]
else:
    predicate = "default"

print_between_triple_backticks("README.md", predicate)
