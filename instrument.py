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
    parser.add_argument("--function", type=str, help="Function to instrument")
    parser.add_argument(
        "--fuel",
        type=int,
        default=10**6,
        help="Number of iterations to run before inducing a crash",
    )
    parser.add_argument("pythonfile", type=str, help="Python file to run")
    args, remaining_args = parser.parse_known_args()

    if not args.pythonfile.endswith(".py"):
        parser.error("Python file must end with .py")

    if not args.function.isidentifier():
        parser.error(f"Function must be a valid identifier; got {args.function}")
    code = pathlib.Path(args.pythonfile).read_text()
    if f"\ndef {args.function}(" not in code:
        parser.error(f"Function {args.function} not found in {args.pythonfile}")

    if args.fuel is not None and args.fuel <= 0:
        parser.error(f"Fuel must be a positive integer; got {args.fuel}")

    if args.naive:
        instrumented = args.pythonfile[:-3] + "_naive.py"
        output = ast_transform.transform(args.pythonfile, dirty_map=None)
    else:
        instrumented = args.pythonfile[:-3] + "_instrumented.py"
        output = analyze_and_transform(
            filename=args.pythonfile,
            function_name=args.function,
            print_invariants=False,
            simplify=False,
        )
    pathlib.Path(instrumented).write_text(output)
    try:
        passed_args = [instrumented] + remaining_args
        persist.sneak_in_fuel_argument(args.fuel, passed_args)
        subprocess.run(
            [sys.executable] + passed_args, env=os.environ | {"PYTHONPATH": os.getcwd()}
        )
    except subprocess.CalledProcessError as ex:
        print(f"Error running {instrumented}: {ex}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"Interrupted", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
