import os
import subprocess
import sys

from pythia.analysis import analyze_function


def main() -> None:
    this, module, *args = sys.argv
    base_dir = f"experiment/{module}"
    instrumented = f"{base_dir}/instrumented.py"
    analyze_function(
        f"{base_dir}/main.py",
        print_invariants=False,
        outfile=instrumented,
        simplify=True,
    )
    subprocess.run(
        [sys.executable, instrumented] + args, env={"PYTHONPATH": os.getcwd()}
    )


if __name__ == "__main__":
    main()
