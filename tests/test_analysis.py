import pathlib
import re
import typing
from itertools import product
import pytest

from pythia import analysis


def collect(*filenames: str) -> typing.Iterator[tuple[str, str, bool]]:
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
def test_lists(filename, func, simplify) -> None:
    analysis.analyze_function(filename, func, simplify=simplify)
