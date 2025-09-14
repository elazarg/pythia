import pathlib
import re
import typing
import pytest

from spyte import analysis


def collect(*filenames: str) -> typing.Iterator[tuple[str, str, bool]]:
    filenames = [pathlib.Path(f) for f in filenames]
    for filename in filenames:
        funcnames = re.findall(
            r"def ([a-zA-Z][a-zA-Z0-9_]*)",
            pathlib.Path(filename).read_text(),
        )
        for f in funcnames:
            if f in ["new"]:
                continue
            yield filename, f, True
            yield filename, f, False


@pytest.mark.parametrize(
    "filename,func,simplify", collect("test_data/lists.py", "test_data/iteration.py")
)
def test_functions(filename: pathlib.Path, func, simplify) -> None:
    analysis.analyze_function(filename, func, simplify=simplify, print_invariants=True)
