import math
import networkx as nx
from pythia.graph_utils import (
    EXIT_LABEL,
    is_exit,
    Block,
    BackwardBlock,
    Cfg,
    simplify_cfg,
    refine_to_chain,
    find_loop_end,
    find_loop_entry,
    find_loops,
)


def test_label_and_exit():
    # Test that EXIT_LABEL is infinity
    assert EXIT_LABEL == math.inf

    # Test is_exit function
    assert is_exit(EXIT_LABEL)
    assert not is_exit(0)
    assert not is_exit(42)


def test_block_init():
    # Test initializing a Block with a list of instructions
    instructions = [1, 2, 3, 4, 5]
    block = Block(instructions)
    assert len(block) == 5
    assert block._instructions == instructions


def test_block_iter():
    # Test iterating through a Block
    instructions = [1, 2, 3, 4, 5]
    block = Block(instructions)

    # Test __iter__
    assert list(block) == instructions

    # Test items
    assert list(block.items()) == list(enumerate(instructions))


def test_block_indices():
    # Test first_index and last_index
    instructions = [1, 2, 3, 4, 5]
    block = Block(instructions)

    assert block.first_index() == 0
    assert block.last_index() == 4

    # Test with empty block
    empty_block = Block([])
    assert empty_block.first_index() == 0
    assert empty_block.last_index() == 0


def test_block_len_and_bool():
    # Test __len__ and __bool__
    instructions = [1, 2, 3, 4, 5]
    block = Block(instructions)

    assert len(block) == 5
    assert bool(block) is True

    # Test with empty block
    empty_block = Block([])
    assert len(empty_block) == 0
    assert bool(empty_block) is False


def test_block_getitem():
    # Test __getitem__
    instructions = [1, 2, 3, 4, 5]
    block = Block(instructions)

    assert block[0] == 1
    assert block[2] == 3
    assert block[4] == 5

    # Test with negative indices
    assert block[-1] == 5
    assert block[-3] == 3


def test_block_reversed():
    # Test __reversed__
    instructions = [1, 2, 3, 4, 5]
    block = Block(instructions)
    backward_block = reversed(block)

    assert isinstance(backward_block, BackwardBlock)
    assert list(backward_block) == list(reversed(instructions))


def test_backward_block():
    # Test BackwardBlock functionality
    instructions = [1, 2, 3, 4, 5]
    block = Block(instructions)
    backward_block = BackwardBlock(block)

    # Test __iter__
    assert list(backward_block) == [5, 4, 3, 2, 1]

    # Test items
    assert list(backward_block.items()) == [(4, 5), (3, 4), (2, 3), (1, 2), (0, 1)]

    # Test indices
    assert backward_block.first_index() == 4
    assert backward_block.last_index() == 0

    # Test __len__ and __bool__
    assert len(backward_block) == 5
    assert bool(backward_block) is True

    # Test __getitem__
    assert backward_block[0] == 1
    assert backward_block[4] == 5

    # Test __reversed__
    assert reversed(backward_block) is block


def test_cfg_init_from_digraph():
    # Test initializing a Cfg from a DiGraph
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)

    cfg = Cfg(graph)
    assert isinstance(cfg.graph, nx.DiGraph)
    assert list(cfg.graph.edges()) == [(0, 1), (1, 2), (2, EXIT_LABEL)]


def test_cfg_init_from_edge_list():
    # Test initializing a Cfg from a list of edges
    edges = [(0, 1), (1, 2)]

    cfg = Cfg(edges)
    assert isinstance(cfg.graph, nx.DiGraph)
    assert list(cfg.graph.edges()) == [(0, 1), (1, 2), (2, EXIT_LABEL)]


def test_cfg_init_with_blocks():
    # Test initializing a Cfg with blocks
    edges = [(0, 1), (1, 2)]
    blocks = {0: [10, 11], 1: [20, 21, 22], 2: [30]}

    cfg = Cfg(edges, blocks)

    # Check that blocks were properly added
    assert isinstance(cfg[0], Block)
    assert list(cfg[0]) == [10, 11]
    assert list(cfg[1]) == [20, 21, 22]
    assert list(cfg[2]) == [30]


def test_cfg_entry_exit():
    # Test entry_label, exit_label, and entry methods
    edges = [(0, 1), (1, 2)]

    cfg = Cfg(edges, add_sink=True)

    assert cfg.entry_label == 0
    assert cfg.exit_label == EXIT_LABEL
    assert isinstance(cfg.entry, Block)
    assert not cfg.entry  # Empty block by default


def test_cfg_nodes_and_labels():
    # Test nodes and labels methods
    edges = [(0, 1), (1, 2)]

    cfg = Cfg(edges, add_sink=True)

    assert set(cfg.nodes()) == {0, 1, 2, EXIT_LABEL}
    assert set(cfg.labels) == {0, 1, 2, EXIT_LABEL}


def test_cfg_items():
    # Test items method
    edges = [(0, 1), (1, 2)]
    blocks = {0: [10, 11], 1: [20, 21], 2: [30]}

    cfg = Cfg(edges, blocks, add_sink=True)

    items = list(cfg.items())
    assert len(items) == 4

    labels, blocks_from_items = zip(*items)
    assert set(labels) == {0, 1, 2, EXIT_LABEL}
    assert all(isinstance(block, Block) for block in blocks_from_items)


def test_cfg_getitem_setitem():
    # Test __getitem__ and __setitem__
    edges = [(0, 1), (1, 2)]

    cfg = Cfg(edges)

    # Test __getitem__
    assert isinstance(cfg[0], Block)
    assert list(cfg[0]) == []  # Empty block by default

    # Test __setitem__
    new_block = Block([100, 101])
    cfg[0] = new_block
    assert cfg[0] is new_block
    assert list(cfg[0]) == [100, 101]


def test_cfg_predecessors_successors():
    # Test predecessors and successors methods
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]

    cfg = Cfg(edges, add_sink=True)

    assert set(cfg.predecessors(0)) == set()
    assert set(cfg.predecessors(1)) == {0}
    assert set(cfg.predecessors(3)) == {1, 2}

    assert set(cfg.successors(0)) == {1, 2}
    assert set(cfg.successors(1)) == {3}
    assert set(cfg.successors(3)) == {EXIT_LABEL}


def test_cfg_dominance():
    # Test dominance analysis methods
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]

    cfg = Cfg(edges, add_sink=True)

    # Test immediate_dominators
    idom = cfg.immediate_dominators()
    assert idom[1] == 0
    assert idom[2] == 0
    assert idom[3] == 0
    assert idom[4] == 3

    # Test dominance_frontiers
    df = cfg.dominance_frontiers()
    assert df[0] == set()
    assert df[1] == {3}
    assert df[2] == {3}
    assert df[3] == set()
    assert df[4] == set()


def test_simplify_cfg():
    # Create a CFG with a chain that can be simplified
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    blocks = {0: [10], 1: [20], 2: [30], 3: [40], 4: [50]}

    cfg = Cfg(edges, blocks, add_sink=True)

    # Simplify the CFG
    simplified = simplify_cfg(cfg)

    # Check that nodes 1, 2, and 3 were merged
    assert set(simplified.labels) == {0, EXIT_LABEL}
    assert list(simplified[0]) == [10, 20, 30, 40, 50]


def test_refine_to_chain():
    cfg = Cfg(
        graph=[],
        blocks={0: [10, 20]},
    )
    refined = Cfg(
        graph=[(0, 1)],
        blocks={0: [10], 1: [20]},
    )
    actual = refine_to_chain(cfg)
    assert list(actual.graph.edges) == list(refined.graph.edges)
    assert list(actual.items()) == list(refined.items())


def test_find_loop_boundaries():
    # Create a CFG with a loop
    edges = [(0, 1), (1, 2), (2, 3), (3, 1), (1, 4)]

    cfg = Cfg(edges, add_sink=True)

    # Find loop boundaries
    loop_label = 1
    loop_end = find_loop_end(cfg, loop_label)
    loop_entry = find_loop_entry(cfg, loop_label)

    assert loop_end == 3
    assert (  # loop entry is the first node inside the loop, not the header
        loop_entry == 2
    )


def test_find_loops():
    # Create a CFG with a loop
    edges = [(0, 1), (1, 2), (2, 3), (3, 1), (1, 4)]

    # Create blocks with a special instruction that marks loops
    blocks = {
        0: ["start"],
        1: ["loop_header"],
        2: ["loop_body"],
        3: ["loop_end"],
        4: ["exit"],
    }

    cfg = Cfg(edges, blocks)

    # Define a function to identify loop headers
    def is_loop(instruction):
        return instruction == "loop_header"

    loops = find_loops(cfg, is_loop)

    assert list(loops) == [(1, 0)]
