from __future__ import annotations as _

import math
import typing
from copy import deepcopy
from dataclasses import dataclass
from itertools import chain
from typing import Callable, Iterator, Optional

import networkx as nx
from networkx.classes.reportviews import NodeView

BLOCK_NAME = "block"

type Label = int | float
type Location = tuple[Label, int]


@dataclass
class Block[T]:
    _instructions: list[T]

    def __init__(self, instructions: list[T]):
        self._instructions = instructions

    def __iter__(self) -> Iterator[T]:
        return iter(self._instructions)

    def items(self) -> Iterator[tuple[int, T]]:
        return iter(enumerate(self._instructions))

    def first_index(self) -> int:
        return 0

    def last_index(self) -> int:
        return max(len(self) - 1, 0)

    def __len__(self) -> int:
        return len(self._instructions)

    def __bool__(self) -> bool:
        return bool(self._instructions)

    def __getitem__(self, index: int) -> T:
        return self._instructions[index]

    def __reversed__(self) -> BackwardBlock[T]:
        return BackwardBlock(self)


@dataclass
class BackwardBlock[T]:
    block: Block[T]

    def __iter__(self) -> Iterator[T]:
        return iter(reversed(self.block._instructions))

    def items(self) -> Iterator[tuple[int, T]]:
        return iter(reversed(list(self.block.items())))

    def first_index(self) -> int:
        return self.block.last_index()

    def last_index(self) -> int:
        return self.block.first_index()

    def __len__(self) -> int:
        return len(self.block)

    def __bool__(self) -> bool:
        return bool(self.block)

    def __getitem__(self, index: int) -> T:
        return self.block[index]

    def __reversed__(self) -> Block[T]:
        return self.block


class Cfg[T]:
    graph: nx.DiGraph
    _annotator: Callable[[Location, T], str]

    @property
    def annotator(self) -> Callable[[Location, T], str]:
        return self._annotator

    @annotator.setter
    def annotator(self, annotator: Callable[[Location, T], str]) -> None:
        self._annotator = staticmethod(annotator)

    def __init__(
        self,
        graph: (
            nx.DiGraph
            | dict
            | list[tuple[Label, Label]]
            | list[tuple[Label, Label, dict]]
        ),
        blocks: Optional[dict[Label, list[T]]] = None,
        add_sink: bool = True,
        add_source=False,
    ) -> None:
        if isinstance(graph, nx.DiGraph):
            self.graph = graph
        else:
            self.graph = nx.DiGraph(graph)

        if blocks is not None:
            self.graph.add_nodes_from(blocks.keys())
            nx.set_node_attributes(
                self.graph,
                name=BLOCK_NAME,
                values={k: Block(block) for k, block in blocks.items()},
            )

        sinks = {label for label in self.labels if self.graph.out_degree(label) == 0}
        if add_sink:
            # Connect all sink nodes to exit:
            sink = self.exit_label
            self.graph.add_node(sink, block=Block([]))
            for label in sinks:
                self.graph.add_edge(label, sink)
        else:
            [sink] = sinks

        sources = {label for label in self.labels if self.graph.in_degree(label) == 0}
        if add_source:
            # Connect all source nodes to entry:
            source = self.entry_label
            self.graph.add_node(source, block=Block([]))
            for label in sources:
                self.graph.add_edge(source, label)
        else:
            assert len(sources) == 1, sources
            [source] = sources
        assert {sink, source} == {self.exit_label, self.entry_label}, {sink, source}
        self.annotator = lambda tup, x: ""

    @property
    def entry_label(self) -> Label:
        return 0

    @property
    def exit_label(self) -> Label:
        return math.inf

    @property
    def entry(self) -> Block:
        return self.graph.nodes[self.entry_label]

    @property
    def nodes(self) -> NodeView:
        return self.graph.nodes

    @property
    def labels(self) -> set[Label]:
        return set(self.graph.nodes.keys())

    def items(self) -> Iterator[tuple[Label, Block[T]]]:
        yield from ((label, self[label]) for label in self.labels)

    def __getitem__(self, label: Label) -> Block[T]:
        return self.graph.nodes[label][BLOCK_NAME]

    def __setitem__(self, label: Label, block: Block[T]) -> None:
        self.graph.nodes[label][BLOCK_NAME] = block

    def reverse(self: Cfg[T], copy: bool) -> Cfg[T]:
        return Cfg(self.graph.reverse(copy=copy), add_sink=False)

    def draw(self) -> None:
        import matplotlib.pyplot as plt  # type: ignore

        nx.draw_networkx(self.graph, with_labels=True)
        plt.show()

    def predecessors(self, label: Label) -> Iterator[Label]:
        return self.graph.predecessors(label)

    def successors(self, label: Label) -> Iterator[Label]:
        return self.graph.successors(label)

    def __deepcopy__(self, memodict={}) -> Cfg:
        return Cfg(self.graph.copy(), add_sink=False)

    def dominance_frontiers(self) -> dict[Label, set[Label]]:
        return nx.dominance_frontiers(self.graph, self.entry_label)

    def immediate_dominators(self) -> dict[Label, set[Label]]:
        return nx.immediate_dominators(self.graph, self.entry_label)

    def reverse_dom_tree(self) -> set[tuple[Label, Label]]:
        dominators = nx.DiGraph((y, x) for x, y in self.immediate_dominators().items())
        return {(y, x) for x, y in nx.transitive_closure(dominators).edges}


def simplify_cfg(cfg: Cfg, exception_labels=None) -> Cfg:
    """Contract chains with in_degree=out_degree=1:
    i.e. turns > [-] [-] [-] <
         into  >[---]<
    The label of the chain is the label of its first element.
    """
    if exception_labels is None:
        exception_labels = set()
    # pretty_print_cfg(cfg)

    g = cfg.graph
    starts = (
        set(
            chain(
                (n for n in g if g.in_degree(n) != 1),
                chain.from_iterable(g.successors(n) for n in g if g.out_degree(n) > 1),
            )
        )
        | {cfg.exit_label}
    ) - exception_labels
    blocks = {}
    labels = set()
    edges: list[tuple[Label, Label]] = []
    for label in starts:
        n = label
        instructions = []
        while True:
            instructions.extend(g.nodes[n][BLOCK_NAME])
            if g.out_degree(n) != 1:
                break
            next_n = next(iter(g.successors(n)))
            if next_n in starts:
                break
            if len(instructions) and type(instructions[-1]).__name__ == "Jump":
                del instructions[-1]
            n = next_n
        labels.add(label)
        edges.extend(
            (label, suc)
            for suc in g.successors(n)
            if not {label, suc} & exception_labels
        )
        blocks[label] = instructions
    # print(edges)
    simplified_cfg = Cfg(edges, blocks=blocks, add_sink=False)

    # remove empty blocks, and connect their predecessors to their successors
    empty = {label for label, block in simplified_cfg.items() if not block}
    empty.remove(simplified_cfg.exit_label)
    for label in empty:
        preds = set(simplified_cfg.predecessors(label)) - empty
        sucs = set(simplified_cfg.successors(label)) - empty
        simplified_cfg.graph.remove_node(label)
        simplified_cfg.graph.add_edges_from(
            [(pred, suc) for pred in preds for suc in sucs]
        )

    simplified_cfg.annotator = cfg.annotator
    # pretty_print_cfg(simplified_cfg)
    return simplified_cfg


def refine_to_chain(cfg: Cfg) -> Cfg:
    """can be used to refine basic blocks into blocks - the dual of simplify_cfg()
    assume cfg.graph.nodes[n][attr] is a list
    returns a graph whose nodes are the refinement of the lists into paths
    the elements of the lists are held as to_attr
    the nodes become tuples (node_index, list_index)"""
    g = cfg.graph
    paths = []
    for n in g.nodes():
        block = g.nodes[n][BLOCK_NAME]
        size = len(block)
        assert size or n == math.inf, n
        path = nx.path_graph(size, create_using=nx.DiGraph())
        nx.relabel_nodes(path, mapping={x: n + x for x in path.nodes()}, copy=False)
        path.add_edges_from((n + max(0, size - 1), s) for s in g.successors(n))
        paths.append(path)
    blocks = {
        n + i: [block]
        for n in g.nodes()
        for i, block in enumerate(g.nodes[n][BLOCK_NAME])
    }
    res: nx.DiGraph = nx.compose_all(paths)
    simplified_cfg = Cfg(res.edges(), blocks=blocks, add_sink=True, add_source=False)
    simplified_cfg.annotator = cfg.annotator
    return simplified_cfg


def node_data_map[
    T, Q
](cfg: Cfg[T], f: Callable[[Label, Block[T]], Block[Q]]) -> Cfg[Q]:
    cfg = deepcopy(cfg)
    for n, data in cfg.nodes.items():
        data[BLOCK_NAME] = f(n, data[BLOCK_NAME])
    return typing.cast(Cfg[Q], cfg)


def print_block[
    T
](label: Label, block: Block[T], *annotators: Callable[[Location, T], object]) -> None:
    print(label, ":")
    for index, ins in enumerate(block):
        location = (label, index)
        str_location = f"{label}.{index}"
        print(
            f"\t{str_location:6}",
            *[f"{annotator(location, ins):7}" for annotator in annotators],
            ins,
        )


def pretty_print_cfg[T](cfg: Cfg[T]) -> None:
    for label, block in sorted(cfg.items()):
        if math.isinf(label):
            continue
        print(list(cfg.graph.predecessors(label)))
        print_block(label, block, cfg.annotator)
        print(list(cfg.graph.successors(label)))
        print()


def single_source_dijkstra_path_length(
    cfg: Cfg, source: int, weight: str
) -> dict[Label, int]:
    return nx.single_source_dijkstra_path_length(cfg.graph, source, weight=weight)


def find_first_for_loop[
    T
](cfg: Cfg[T], is_for: Callable[[T], bool]) -> tuple[Label, Label]:
    first_label = min(
        label for label, block in cfg.items() if block and any(is_for(b) for b in block)
    )
    block = cfg[first_label]
    assert len(block) == 1
    prev, *_, after = sorted(cfg.predecessors(first_label))
    return (first_label, after)
