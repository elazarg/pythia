import pathlib
from difflib import ndiff

import pytest

from pythia import analysis


def analyze_and_transform(
    filename: str, function_name: str, expected_outfile: str, simplify: bool
) -> None:
    expected = pathlib.Path(expected_outfile).read_text(encoding="utf-8")
    actual = analysis.analyze_and_transform(
        filename=filename,
        function_name=function_name,
        print_invariants=False,
        simplify=simplify,
    )
    assert expected == actual


@pytest.mark.parametrize("simplify", [True, False])
def test_feature_selection(simplify: bool) -> None:
    analyze_and_transform(
        filename="experiment/feature_selection/main.py",
        expected_outfile="experiment/feature_selection/main_instrumented.py",
        function_name="do_work",
        simplify=simplify,
    )


@pytest.mark.parametrize("simplify", [True, False])
def test_k_means(simplify: bool) -> None:
    analyze_and_transform(
        filename="experiment/k_means/main.py",
        expected_outfile="experiment/k_means/main_instrumented.py",
        function_name="k_means",
        simplify=simplify,
    )


@pytest.mark.parametrize("simplify", [True, False])
def test_pivoter(simplify: bool) -> None:
    analyze_and_transform(
        filename="experiment/pivoter/main.py",
        expected_outfile="experiment/pivoter/main_instrumented.py",
        function_name="run",
        simplify=simplify,
    )
