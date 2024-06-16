import os
import subprocess
import sys

from pythia.analysis import analyze_and_transform


def main() -> None:
    this, pythonfile, *args = sys.argv
    if not pythonfile.endswith(".py"):
        print(f"Usage: {this} <pythonfile> [<args>...]", file=sys.stderr)
        sys.exit(1)
    instrumented = pythonfile[:-3] + "_instrumented.py"
    analyze_and_transform(
        pythonfile,
        print_invariants=False,
        outfile=instrumented,
        simplify=False,
    )
    try:
        subprocess.run(
            [sys.executable, instrumented] + args,
            env=os.environ | {"PYTHONPATH": os.getcwd()},
        )
    except subprocess.CalledProcessError as ex:
        print(f"Error running {instrumented}: {ex}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"Interrupted", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
