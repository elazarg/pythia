import os
import pathlib
import subprocess
import sys

from pythia.analysis import analyze_and_transform


def main() -> None:
    this, pythonfile, *args = sys.argv
    if not pythonfile.endswith(".py"):
        print(f"Usage: {this} <pythonfile> [<args>...]", file=sys.stderr)
        sys.exit(1)
    instrumented = pythonfile[:-3] + "_instrumented.py"
    output = analyze_and_transform(
        filename=pythonfile,
        function_name=None,
        print_invariants=False,
        simplify=False,
    )
    pathlib.Path(instrumented).write_text(output)
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
