from pythia.analysis import analyze_function


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('function_name', nargs='+')
    parser.add_argument('--output', default=None)
    parser.add_argument('--print-invariants', action='store_true')
    parser.add_argument('--simplify', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    analyze_function(args.filename, *args.function_name,
                     print_invariants=args.print_invariants,
                     outfile=args.output,
                     simplify=args.simplify)


if __name__ == '__main__':
    main()
