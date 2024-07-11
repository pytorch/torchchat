import sys


def print_until_equals(filename):
    output = False
    past_output = False
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("-" * 8):
                output = True
            if output and line.startswith("=" * 8):
                if past_output:
                    print("Double end-of-sequence line")
                    exit(1)
                past_output = True
                output = False
            if output:
                print(line)

        if not past_output:
            print("Did find sequence to output")
            exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage:\n {sys.executable} {sys.argv[0]} filename")
        sys.exit(1)
    filename = sys.argv[1]
    print_until_equals(filename)
