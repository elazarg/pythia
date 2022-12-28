from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeVar, Generic, Callable, Any, Iterator, TypeAlias
import networkx as nx
from itertools import chain

T = TypeVar('T')
Q = TypeVar('Q')


@dataclass
class ForwardBlock(Generic[T]):
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
        return (len(self) - 1) if len(self) > 0 else 0

    def __len__(self) -> int:
        return len(self._instructions)

    def __bool__(self):
        return bool(self._instructions)

    def __getitem__(self, index: int) -> T:
        return self._instructions[index]

    def __setitem__(self, index: int, item: T) -> None:
        self._instructions[index] = item

    def __delitem__(self, index: int) -> None:
        del self._instructions[index]

    def __reversed__(self) -> Block[T]:
        return BackwardBlock(self)


@dataclass
class BackwardBlock(Generic[T]):
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

    def __bool__(self):
        return bool(self.block)

    def __getitem__(self, index: int) -> T:
        return self.block[index]

    def __reversed__(self) -> Block[T]:
        return self.block


Block: TypeAlias = ForwardBlock[T] | BackwardBlock[T]


class Cfg(Generic[T]):
    graph: nx.DiGraph
    _annotator: Callable[[tuple[int, int], T], str]

    @property
    def annotator(self) -> Callable[[tuple[int, int], T], str]:
        return self._annotator

    @annotator.setter
    def annotator(self, annotator: Callable[[tuple[int, int], T], str]) -> None:
        self._annotator = staticmethod(annotator)

    def __init__(self, graph: nx.DiGraph | dict | list[tuple[int, int, dict]], blocks: dict[int, list[T]]=None, add_sink=True) -> None:
        if isinstance(graph, nx.DiGraph):
            self.graph = graph
        else:
            self.graph = nx.DiGraph(graph)

        if blocks is not None:
            self.graph.add_nodes_from(blocks.keys())
            nx.set_node_attributes(self.graph, name='block', values={k: ForwardBlock(block) for k, block in blocks.items()})

        if add_sink:
            # Connect all sink nodes to exit:
            sinks = {label for label in self.labels if self.graph.out_degree(label) == 0}
            self.graph.add_node(self.exit_label, block=ForwardBlock([]))
            for label in sinks:
                self.graph.add_edge(label, self.exit_label)

        [sink] = {label for label in self.labels if self.graph.out_degree(label) == 0}
        [source] = {label for label in self.labels if self.graph.in_degree(label) == 0}
        assert {sink, source} == {self.exit_label, self.entry_label}, {sink, source}
        self.annotator = lambda x: ''

    @property
    def entry_label(self) -> int:
        return 0

    @property
    def exit_label(self) -> int:
        return math.inf

    @property
    def entry(self):
        return self.graph.nodes[self.entry_label]

    @property
    def nodes(self):
        return self.graph.nodes

    @property
    def labels(self):
        return self.graph.nodes.keys()

    def items(self) -> Iterator[tuple[int, Block[T]]]:
        yield from ((label, self[label]) for label in self.labels)

    def __getitem__(self, label: int) -> Block[T]:
        return self.graph.nodes[label]['block']

    def __setitem__(self, label: int, block: Block[T]) -> None:
        self.graph.nodes[label]['block'] = block

    def reverse(self: Cfg[T], copy) -> Cfg[T]:
        return Cfg(self.graph.reverse(copy=copy), add_sink=False)

    def draw(self) -> None:
        import matplotlib.pyplot as plt
        nx.draw_networkx(self.graph, with_labels=True)
        plt.show()

    def print_graph(self) -> None:
        for label in sorted(self.graph.nodes()):
            print(label, ':', self[label])

    def predecessors(self, label) -> Iterator[int]:
        return self.graph.predecessors(label)

    def successors(self, label) -> Iterator[int]:
        return self.graph.successors(label)

    def copy(self: Cfg) -> Cfg:
        return Cfg(self.graph.copy(), add_sink=False)

    def dominance_frontiers(self) -> dict[int, set[int]]:
        return nx.dominance_frontiers(self.graph, self.entry_label)

    def immediate_dominators(self) -> dict[int, set[int]]:
        return nx.immediate_dominators(self.graph, self.entry_label)

    def reverse_dom_tree(self) -> set[tuple[int, int]]:
        dominators = nx.DiGraph((y, x) for x, y in self.immediate_dominators().items())
        return {(y, x) for x, y in nx.transitive_closure(dominators).edges}


def reverse_weights(g: nx.DiGraph, weight='weight'):
    g = g.reverse()
    for s, t in g.edges():
        e = g[s][t]
        e[weight] = -e[weight]
    return g


def simplify_cfg(cfg: Cfg) -> Cfg:
    """Contract chains with in_degree=out_degree=1:
    i.e. turns > [-] [-] [-] <
         into  >[---]<
    The label of the chain is the label of its first element.
    """
    blockname = 'block'
    g = cfg.graph
    starts = set(chain((n for n in g if g.in_degree(n) != 1),
                       chain.from_iterable(g.successors(n) for n in g
                                           if g.out_degree(n) > 1)))
    blocks = {}
    labels = set()
    edges = []
    for label in starts:
        n = label
        instructions = []
        while True:
            instructions += g.nodes[n][blockname]
            if g.out_degree(n) != 1:
                break
            next_n = next(iter(g.successors(n)))
            if g.in_degree(next_n) != 1:
                break
            if len(instructions) and type(instructions[-1]).__name__ == 'Jump':
                del instructions[-1]
            n = next_n
        labels.add(label)
        edges.extend((label, suc) for suc in g.successors(n))
        blocks[label] = instructions
    simplified_cfg = Cfg(edges, blocks=blocks, add_sink=True)
    simplified_cfg.annotator = cfg.annotator
    return simplified_cfg


def refine_to_chain(g, from_attr, to_attr):
    """can be used to refine basic blocks into blocks - the dual of simplify_cfg()
    assume g.nodes[n][attr] is a list
    returns a graph whose nodes are the refinement of the lists into paths
    the elements of the lists are held as to_attr
    the nodes become tuples (node_index, list_index)"""
    paths = []
    for n in g.nodes_iter():
        block = g.nodes[n][from_attr]
        size = len(block)
        path = nx.path_graph(size, create_using=nx.DiGraph())
        nx.relabel_nodes(path, mapping={x: (n, x) for x in path.nodes()}, copy=False)
        path.add_edges_from(((n, size - 1), (s, 0)) for s in g.successors_iter(n))
        paths.append(path)
    values = {(n, x): block
              for n in g.nodes_iter()
              for x, block in enumerate(g.nodes[n][from_attr])}
    res = nx.compose_all(paths)
    nx.set_node_attributes(res, values, to_attr)
    return res


def node_data_map(cfg: Cfg[T], f: Callable[[int, Block[T]], Block[Q]]) -> Cfg[Q]:
    cfg: Cfg[Any] = cfg.copy()
    for n, data in cfg.nodes.items():
        data['block'] = f(n, data['block'])
    return cfg


def print_block(label: int, block: Block[T],
                *annotators: Callable[[tuple[int, int], T], object]) -> None:
    print(label, ':')
    for index, ins in enumerate(block):
        location = (label, index)
        str_location = f'{label}.{index}'
        print(f'\t{str_location:6}', *[f'{annotator(location, ins):7}' for annotator in annotators], ins)


def pretty_print_cfg(cfg: Cfg[T]) -> None:
    for label, block in sorted(cfg.items()):
        if math.isinf(label):
            continue
        print_block(label, block, cfg.annotator)
        print(list(cfg.graph.neighbors(label)))
        print()


def single_source_dijkstra_path_length(cfg: Cfg, source: int, weight='weight'):
    return nx.single_source_dijkstra_path_length(cfg.graph, source, weight=weight)
