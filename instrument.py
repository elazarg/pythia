import argparse
import pathlib
import subprocess
import sys
from typing import Sequence

from experiment import persist
from pythia import ast_transform
from pythia.analysis import analyze_and_transform


def parse_args(args: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
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
    args, remaining_args = parser.parse_known_args(args)

    if not args.pythonfile.endswith(".py"):
        parser.error("Python file must end with .py")

    if not args.function.isidentifier():
        parser.error(f"Function must be a valid identifier; got {args.function}")
    code = pathlib.Path(args.pythonfile).read_text()
    if f"\ndef {args.function}(" not in code:
        parser.error(f"Function {args.function} not found in {args.pythonfile}")

    if args.fuel is not None and args.fuel <= 0:
        parser.error(f"Fuel must be a positive integer; got {args.fuel}")

    return args, remaining_args


def generate_instrumented_file(naive: bool, pythonfile: str, function: str) -> str:
    if naive:
        instrumented = pythonfile[:-3] + "_naive.py"
        output = ast_transform.transform(pythonfile, dirty_map=None)
    else:
        instrumented = pythonfile[:-3] + "_instrumented.py"
        output = analyze_and_transform(
            filename=pythonfile,
            function_name=function,
            print_invariants=False,
            simplify=False,
        )
    pathlib.Path(instrumented).write_text(output)
    return instrumented


def main() -> None:
    args, remaining_args = parse_args(sys.argv[1:])
    instrumented = generate_instrumented_file(
        args.naive, args.pythonfile, args.function
    )
    try:
        persist.run_instrumented_file(instrumented, remaining_args, args.fuel)
    except subprocess.CalledProcessError as ex:
        print(f"Error running {instrumented}: {ex}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"Interrupted", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
