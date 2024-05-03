
import sys

def print_between_triple_backticks(filename, predicate):
    with open(filename, 'r') as file:
        lines = file.readlines()
    print_flag = False
    for line in lines:
        command_prefix=f"[shell {predicate}]:"
        command_end=f"[end {predicate}]:"
        if line.startswith(command_prefix):
            print(line[len(command_prefix):])
        elif line.startswith(command_end):
            return;
        elif line.startswith('```'):
            print_flag = not print_flag
        elif print_flag:
            print(line, end='')


if len(sys.argv )>1:
    predicate=sys.argv[1]
else:
    predicate="default"
    
print_between_triple_backticks('README.md', predicate)
