import pathlib

from pythia.analysis import analyze_and_transform


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("function_name", nargs="+")
    parser.add_argument("--output", default=None)
    parser.add_argument("--print-invariants", action="store_true")
    parser.add_argument(
        "--simplify", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    output = analyze_and_transform(
        args.filename,
        *args.function_name,
        print_invariants=args.print_invariants,
        simplify=args.simplify
    )
    outfile = args.output
    if outfile is None:
        print(output)
    else:
        pathlib.Path(outfile).write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
