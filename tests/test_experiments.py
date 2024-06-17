import pathlib

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
    expected_transaction = [
        (i, line.strip())
        for i, line in enumerate(expected.splitlines())
        if "transaction.move" in line
    ]
    actual_transaction = [
        (i, line.strip())
        for i, line in enumerate(actual.splitlines())
        if "transaction.move" in line
    ]
    assert actual_transaction == expected_transaction
    assert actual == expected


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
