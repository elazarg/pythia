import pathlib
import re

import pytest

from pythia import analysis


def collect(filename: str) -> list[tuple[str, str]]:
    return [
        (filename, f)
        for f in re.findall(
            r"def ([a-zA-Z][a-zA-Z0-9_]*)",
            pathlib.Path(filename).read_text(),
        )
        if f not in ["new"]
    ]


def test_access():
    analysis.analyze_function("test_data/iteration.py", "access")


@pytest.mark.parametrize("filename,func", collect("test_data/lists.py"))
def test_lists(filename, func):
    analysis.analyze_function(filename, func)


@pytest.mark.parametrize("filename,func", collect("test_data/iteration.py"))
def test_iteration(filename, func):
    analysis.analyze_function(filename, func)
