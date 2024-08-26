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


def compare_transformed_files(actual: str, expected_outfile: pathlib.Path) -> None:
    expected = expected_outfile.read_text(encoding="utf-8")
    expected_transaction = find_transaction(expected)
    actual_transaction = find_transaction(actual)
    assert actual == expected
    assert actual_transaction == expected_transaction


def naive_transform(filename: pathlib.Path, expected_outfile: pathlib.Path) -> None:
    actual = ast_transform.transform(filename, dirty_map=None)
    compare_transformed_files(actual, expected_outfile)


@pytest.mark.parametrize(
    "experiment_name", ["k_means", "omp", "pivoter", "trivial"]
)
def test_naive_transformation(experiment_name: str) -> None:
    exp = pathlib.Path("experiment") / experiment_name
    naive_transform(
        filename=exp / "main.py",
        expected_outfile=exp / "naive.py",
    )


def tcp_transform(
    tag: str, filename: pathlib.Path, function_name: str, expected_outfile: pathlib.Path
) -> None:
    actual = ast_transform.tcp_client(tag, filename, function_name)
    compare_transformed_files(actual, expected_outfile)


@pytest.mark.parametrize(
    "experiment_name", ["k_means", "omp", "pivoter", "trivial"]
)
def test_tcp_transformation(experiment_name: str) -> None:
    exp = pathlib.Path("experiment") / experiment_name
    filename = exp / "main.py"
    expected_outfile = exp / "vm.py"
    tcp_transform(
        tag=experiment_name,
        function_name="run",
        filename=filename,
        expected_outfile=expected_outfile,
    )


def analyze_and_transform(
    experiment_name: str, function_name: str, simplify: bool
) -> None:
    exp = pathlib.Path("experiment") / experiment_name
    filename = exp / "main.py"
    expected_outfile = exp / "instrumented.py"
    actual = analysis.analyze_and_transform(
        filename=filename,
        function_name=function_name,
        print_invariants=False,
        simplify=simplify,
    )
    compare_transformed_files(actual, expected_outfile)


@pytest.mark.parametrize("simplify", [True, False])
def test_analyze_omp(simplify: bool) -> None:
    analyze_and_transform(
        experiment_name="omp",
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


@pytest.mark.parametrize("simplify", [True, False])
def test_trivial(simplify: bool) -> None:
    analyze_and_transform(
        experiment_name="trivial",
        function_name="run",
        simplify=simplify,
    )
