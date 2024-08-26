import pathlib

from checkpoint import persist


def instrument_experiment(experiment_name: str, args: list[str]) -> None:
    def run(fuel: int) -> str:
        instrumented = pathlib.Path("experiment") / experiment_name / "instrumented.py"
        return persist.run_instrumented_file(
            instrumented=instrumented, args=args, fuel=fuel, capture_stdout=True
        )

    expected = run(10**6)
    run(5)
    actual = run(10**6)
    assert actual == expected


def test_instrument_omp() -> None:
    instrument_experiment(
        experiment_name="omp",
        args="healthstudy --k 50".split(),
    )


def test_k_means() -> None:
    instrument_experiment(
        experiment_name="k_means",
        args="1000 4".split(),
    )


def test_pivoter() -> None:
    instrument_experiment(
        experiment_name="pivoter",
        args="--filename=experiment/pivoter/enron.small.edges".split(),
    )


def test_trivial() -> None:
    instrument_experiment(
        experiment_name="trivial",
        args=[],
    )
