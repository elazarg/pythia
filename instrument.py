import argparse
import pathlib
import subprocess
import sys
from typing import Sequence

from checkpoint import persist
from pythia import ast_transform
from pythia.analysis import analyze_and_transform


def parse_args(args: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        allow_abbrev=False, usage="%(prog)s [options] pythonfile [args]"
    )
    parser.add_argument(
        "--kind",
        choices=["analyze", "naive", "vm", "proc"],
        default="analyze",
        help="kind of instrumentation to use",
    )
    parser.add_argument(
        "--function", type=str, default="run", help="Function to instrument"
    )
    parser.add_argument(
        "--fuel",
        type=int,
        default=10**6,
        help="Number of iterations to run before inducing a crash",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Number of iterations to run between checkpointing",
    )
    parser.add_argument("pythonfile", type=str, help="Python file to run")
    args, remaining_args = parser.parse_known_args(args)

    if not args.pythonfile.endswith(".py"):
        parser.error("Python file must end with .py")

    if not args.function.isidentifier():
        parser.error(f"Function must be a valid identifier; got {args.function}")
    code = pathlib.Path(args.pythonfile).read_text()
    if f"def {args.function}(" not in code:
        parser.error(f"Function {args.function} not found in {args.pythonfile}")

    if args.fuel is not None and args.fuel <= 0:
        parser.error(f"Fuel must be a positive integer; got {args.fuel}")

    if args.step is not None and args.step <= 0:
        parser.error(f"Step must be a positive integer; got {args.step}")

    return args, remaining_args


def generate_instrumented_file(
    kind: str, pythonfile: str, function: str
) -> pathlib.Path:
    pythonfile = pathlib.Path(pythonfile)
    match kind:
        case "naive":
            instrumented = pythonfile.with_name("naive.py")
            output = ast_transform.transform(pythonfile, dirty_map=None)
        case "vm":
            instrumented = pythonfile.with_name("vm.py")
            tag = pathlib.Path(pythonfile).parent.name
            output = ast_transform.tcp_client(tag, pythonfile, function)
        case "proc":
            instrumented = pythonfile.with_name("proc.py")
            tag = pathlib.Path(pythonfile).parent.name
            output = ast_transform.coredump(tag, pythonfile, function)
        case "analyze":
            instrumented = pythonfile.with_name("instrumented.py")
            output = analyze_and_transform(
                filename=pythonfile,
                function_name=function,
                print_invariants=False,
                simplify=False,
            )
        case _:
            raise ValueError(f"Unknown kind {kind}")
    pathlib.Path(instrumented).write_text(output)
    return instrumented


def main() -> None:
    args, remaining_args = parse_args(sys.argv[1:])
    instrumented = generate_instrumented_file(args.kind, args.pythonfile, args.function)
    if args.kind != "vm":
        try:
            persist.run_instrumented_file(
                instrumented, remaining_args, args.fuel, args.step
            )
        except subprocess.CalledProcessError as ex:
            print(f"Error running {instrumented}: {ex}", file=sys.stderr)
            sys.exit(1)
        except KeyboardInterrupt:
            print(f"Interrupted", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
