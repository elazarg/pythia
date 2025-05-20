import pytest
import pythia.graph_utils as gu
from pythia.strategy import (
    IterationStrategy,
    ForwardIterationStrategy,
    BackwardIterationStrategy,
    iteration_strategy,
)


# Create a simple CFG for testing
def create_test_cfg():
    # Create a simple CFG with the following structure:
    # 0 -> 1 -> 2
    #      |    |
    #      v    v
    #      3 -> 4 -> EXIT
    edges = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (4, gu.EXIT_LABEL)]
    blocks = {0: ["entry"], 1: ["branch"], 2: ["path1"], 3: ["path2"], 4: ["exit"]}
    return gu.Cfg(edges, blocks, add_sink=False)


def test_forward_iteration_strategy():
    # Create a test CFG
    cfg = create_test_cfg()

    # Create a forward iteration strategy
    strategy = ForwardIterationStrategy(cfg)

    # Test entry_label
    assert strategy.entry_label == 0

    # Test successors
    assert set(strategy.successors(0)) == {1}
    assert set(strategy.successors(1)) == {2, 3}
    assert set(strategy.successors(2)) == {4}
    assert set(strategy.successors(3)) == {4}
    assert set(strategy.successors(4)) == {gu.EXIT_LABEL}

    # Test predecessors
    assert set(strategy.predecessors(0)) == set()
    assert set(strategy.predecessors(1)) == {0}
    assert set(strategy.predecessors(2)) == {1}
    assert set(strategy.predecessors(3)) == {1}
    assert set(strategy.predecessors(4)) == {2, 3}
    assert set(strategy.predecessors(gu.EXIT_LABEL)) == {4}

    # Test __getitem__
    assert list(strategy[0]) == ["entry"]
    assert list(strategy[1]) == ["branch"]
    assert list(strategy[2]) == ["path1"]
    assert list(strategy[3]) == ["path2"]
    assert list(strategy[4]) == ["exit"]

    # Test order
    assert strategy.order((1, 2)) == (1, 2)
    assert strategy.order(("a", "b")) == ("a", "b")


def test_backward_iteration_strategy():
    # Create a test CFG
    cfg = create_test_cfg()

    # Create a backward iteration strategy
    strategy = BackwardIterationStrategy(cfg)

    # Test entry_label (which is actually the exit label in backward iteration)
    assert strategy.entry_label == gu.EXIT_LABEL

    # Test successors (which are predecessors in the original CFG)
    assert set(strategy.successors(gu.EXIT_LABEL)) == {4}
    assert set(strategy.successors(4)) == {2, 3}
    assert set(strategy.successors(3)) == {1}
    assert set(strategy.successors(2)) == {1}
    assert set(strategy.successors(1)) == {0}
    assert set(strategy.successors(0)) == set()

    # Test predecessors (which are successors in the original CFG)
    assert set(strategy.predecessors(0)) == {1}
    assert set(strategy.predecessors(1)) == {2, 3}
    assert set(strategy.predecessors(2)) == {4}
    assert set(strategy.predecessors(3)) == {4}
    assert set(strategy.predecessors(4)) == {gu.EXIT_LABEL}
    assert set(strategy.predecessors(gu.EXIT_LABEL)) == set()

    # Test __getitem__ (should return a BackwardBlock)
    assert isinstance(strategy[0], gu.BackwardBlock)
    assert list(strategy[0]) == ["entry"]  # BackwardBlock reverses the order
    assert list(strategy[1]) == ["branch"]
    assert list(strategy[2]) == ["path1"]
    assert list(strategy[3]) == ["path2"]
    assert list(strategy[4]) == ["exit"]

    # Test order (should swap the elements)
    assert strategy.order((1, 2)) == (2, 1)
    assert strategy.order(("a", "b")) == ("b", "a")


def test_iteration_strategy_factory():
    # Create a test CFG
    cfg = create_test_cfg()

    # Test with backward=False
    forward_strategy = iteration_strategy(cfg, backward=False)
    assert isinstance(forward_strategy, ForwardIterationStrategy)
    assert forward_strategy.entry_label == 0

    # Test with backward=True
    backward_strategy = iteration_strategy(cfg, backward=True)
    assert isinstance(backward_strategy, BackwardIterationStrategy)
    assert backward_strategy.entry_label == gu.EXIT_LABEL


def test_iteration_with_complex_cfg():
    # Create a more complex CFG with a loop
    # 0 -> 1 -> 2 -> 3 -> 6
    #      ^         |
    #      |         v
    #      5 <- 4 <- 3
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 1), (3, 6)]
    blocks = {
        0: ["entry"],
        1: ["loop_header"],
        2: ["loop_body1"],
        3: ["loop_body2"],
        4: ["loop_body3"],
        5: ["loop_tail"],
        6: ["after_loop"],
    }
    cfg = gu.Cfg(edges, blocks)

    # Test forward iteration
    forward_strategy = ForwardIterationStrategy(cfg)

    # Check that we can follow the loop in the forward direction
    assert set(forward_strategy.successors(0)) == {1}
    assert set(forward_strategy.successors(1)) == {2}
    assert set(forward_strategy.successors(2)) == {3}
    assert set(forward_strategy.successors(3)) == {4, 6}
    assert set(forward_strategy.successors(4)) == {5}
    assert set(forward_strategy.successors(5)) == {1}  # Back to the loop header

    # Test backward iteration
    backward_strategy = BackwardIterationStrategy(cfg)

    # Check that we can follow the loop in the backward direction
    assert set(backward_strategy.successors(1)) == {
        0,
        5,
    }  # Loop header has two predecessors
    assert set(backward_strategy.successors(2)) == {1}
    assert set(backward_strategy.successors(3)) == {2}
    assert set(backward_strategy.successors(4)) == {3}
    assert set(backward_strategy.successors(5)) == {4}
    assert set(backward_strategy.successors(1)) == {0, 5}
    assert set(backward_strategy.successors(5)) == {4}
    assert set(backward_strategy.successors(6)) == {3}
