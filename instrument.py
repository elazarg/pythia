import argparse
import os
import pathlib
import subprocess
import sys

from experiment import persist
from pythia import ast_transform
from pythia.analysis import analyze_and_transform


def main() -> None:
    parser = argparse.ArgumentParser(usage="%(prog)s [options] pythonfile [args]")
    parser.add_argument(
        "--naive",
        action="store_true",
        help="Use naive instrumentation: all local variables",
    )
    parser.add_argument("--func", type=str, help="Function to instrument")
    parser.add_argument(
        "--fuel", type=int, help="Number of iterations to run before inducing a crash"
    )
    parser.add_argument("pythonfile", type=str, help="Python file to run")
    args, remaining_args = parser.parse_known_args()

    if args.fuel:
        os.environ[persist.FUEL_ENV] = str(args.fuel)

    if not args.pythonfile.endswith(".py"):
        parser.error("Python file must end with .py")

    if args.naive:
        instrumented = args.pythonfile[:-3] + "_naive.py"
        output = ast_transform.transform(args.pythonfile, dirty_map=None)
    else:
        instrumented = args.pythonfile[:-3] + "_instrumented.py"
        output = analyze_and_transform(
            filename=args.pythonfile,
            function_name=args.func,
            print_invariants=False,
            simplify=False,
        )
    pathlib.Path(instrumented).write_text(output)
    try:
        subprocess.run(
            [sys.executable, instrumented] + remaining_args,
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
