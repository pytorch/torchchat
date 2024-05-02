def print_between_triple_backticks(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
    print_flag = False
    for line in lines:
        if line.startswith("```"):
            print_flag = not print_flag
        elif print_flag:
            print(line, end="")


print_between_triple_backticks("README.md")
