import sys


def print_between_triple_backticks(filename, predicate):
    with open(filename, "r") as file:
        lines = file.readlines()
    print_flag = False
    for i, line in enumerate(lines):
        command = f"[shell {predicate}]:"
        prefix = f"[prefix {predicate}]:"
        end = f"[end {predicate}]:"
        skip = f"[skip {predicate}]:"
        if line.startswith(prefix):
            print(line[len(prefix) : -1], end="")
        elif line.startswith(command):
            print(line[len(command) :])
        elif line.startswith(end):
            return
        elif line.startswith(skip):
            keyword = line[len(skip) :]-1).strip
            if keyword == "begin"
                print("if false; then")
            elif keyword == "end"
                print("fi")
            else:
                print(f"echo 'error in line {i} of README.md")
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
