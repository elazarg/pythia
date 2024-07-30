import pathlib
from typing import Any

import pytest

from pythia import analysis, ast_transform


def find_transaction(text: str) -> list[tuple[int, Any]]:
    return [
        (i, line.strip())
        for i, line in enumerate(text.splitlines())
        if "transaction.move" in line
    ]


def compare_transformed_files(actual: str, expected_outfile: str) -> None:
    expected = pathlib.Path(expected_outfile).read_text(encoding="utf-8")
    expected_transaction = find_transaction(expected)
    actual_transaction = find_transaction(actual)
    assert actual_transaction == expected_transaction
    assert actual == expected


def naive_transform(filename: str, expected_outfile: str) -> None:
    actual = ast_transform.transform(filename, dirty_map=None)
    compare_transformed_files(actual, expected_outfile)


@pytest.mark.parametrize("experiment_name", ["k_means", "feature_selection", "pivoter"])
def test_naive_transformation(experiment_name: str) -> None:
    naive_transform(
        filename=f"experiment/{experiment_name}/main.py",
        expected_outfile=f"experiment/{experiment_name}/main_naive.py",
    )


def analyze_and_transform(
    experiment_name: str, function_name: str, simplify: bool
) -> None:
    filename = f"experiment/{experiment_name}/main.py"
    expected_outfile = f"experiment/{experiment_name}/main_instrumented.py"
    actual = analysis.analyze_and_transform(
        filename=filename,
        function_name=function_name,
        print_invariants=False,
        simplify=simplify,
    )
    compare_transformed_files(actual, expected_outfile)


@pytest.mark.parametrize("simplify", [True, False])
def test_analyze_feature_selection(simplify: bool) -> None:
    analyze_and_transform(
        experiment_name="feature_selection",
        function_name="run",
        simplify=simplify,
    )


@pytest.mark.parametrize("simplify", [True, False])
def test_k_means(simplify: bool) -> None:
    analyze_and_transform(
        experiment_name="k_means",
        function_name="run",
        simplify=simplify,
    )


@pytest.mark.parametrize("simplify", [True, False])
def test_pivoter(simplify: bool) -> None:
    analyze_and_transform(
        experiment_name="pivoter",
        function_name="run",
        simplify=simplify,
    )
