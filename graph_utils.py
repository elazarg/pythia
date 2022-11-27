from __future__ import annotations

import math
import typing
from dataclasses import dataclass
from typing import TypeVar, Generic, Callable, Any, Iterator, TypeAlias
import networkx as nx
from itertools import chain

if typing.TYPE_CHECKING:
    from tac_analysis_domain import AbstractDomain

T = TypeVar('T')
Q = TypeVar('Q')


@dataclass
class ForwardBlock(Generic[T]):
    _instructions: list[T]
    pre: dict[str, AbstractDomain]
    post: dict[str, AbstractDomain]

    def __init__(self, instructions: list[T]):
        self._instructions = instructions
        self.pre = {}
        self.post = {}

    def __iter__(self) -> Iterator[T]:
        return iter(self._instructions)

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

    def __len__(self) -> int:
        return len(self.block)

    def __bool__(self):
        return bool(self.block)

    def __getitem__(self, index: int) -> T:
        return self.block[index]

    @property
    def pre(self) -> dict[str, AbstractDomain]:
        return self.block.post

    @property
    def post(self) -> dict[str, AbstractDomain]:
        return self.block.pre

    def __reversed__(self) -> Block[T]:
        return self.block


Block: TypeAlias = ForwardBlock | BackwardBlock


class Cfg(Generic[T]):
    graph: nx.DiGraph
    _annotator: Callable[[T], str] = staticmethod(lambda x: '')

    @property
    def annotator(self) -> Callable[[T], str]:
        return self._annotator

    @annotator.setter
    def annotator(self, annotator: Callable[[T], str]) -> None:
        self._annotator = staticmethod(annotator)

    def __init__(self, graph: nx.DiGraph | dict | list[tuple[int, int, dict]], blocks=None) -> None:
        if isinstance(graph, nx.DiGraph):
            self.graph = graph
        else:
            self.graph = nx.DiGraph(graph)

        if blocks is not None:
            nx.set_node_attributes(self, name='block', values={k: ForwardBlock(block) for k, block in blocks.items()})

        # Connect all sink nodes to exit:
        self.graph.add_node(self.exit_label, block=ForwardBlock([]))
        sinks = {label for label in self.labels if self.graph.out_degree(label) == 0}
        for label in sinks:
            self.graph.add_edge(label, self.exit_label)

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
        return Cfg(self.graph.reverse(copy=copy))

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
        return Cfg(self.graph.copy())

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
    blocks = Cfg(nx.DiGraph())
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
        if label not in blocks.graph:
            blocks.graph.add_node(label)
        blocks.graph.add_edges_from((label, suc) for suc in g.successors(n))
        blocks[label] = ForwardBlock(instructions)
    blocks.annotator = cfg.annotator
    return blocks


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


def test_refine_to_chain():
    import code_examples
    tac_name = 'tac_block'
    from tac_analysis import make_tac_cfg
    cfg = make_tac_cfg(code_examples.CreateSphere)
    refined = refine_to_chain(cfg, tac_name, 'tac')
    for x in sorted(refined.nodes_iter()):
        print(x, refined.nodes[x]['tac'].format())


def print_block(n: int, block: Block[T], annotator) -> None:
    print(n, ':')
    for i, ins in enumerate(block):
        label = f'{n}.{i}'
        print(f'\t{label:6} {annotator(ins):5}\t', ins)


def pretty_print_cfg(cfg: Cfg[T]) -> None:
    for label, block in sorted(cfg.items()):
        if math.isinf(label):
            continue
        print_block(label, block, cfg.annotator)
        print(list(cfg.graph.neighbors(label)))
        print()


if __name__ == '__main__':
    test_refine_to_chain()


def single_source_dijkstra_path_length(cfg: Cfg, source: int, weight='weight'):
    return nx.single_source_dijkstra_path_length(cfg.graph, source, weight=weight)
