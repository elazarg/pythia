from __future__ import annotations

from typing import TypeVar, Generic, Callable, Any
import networkx as nx
from itertools import chain

T = TypeVar('T')
Q = TypeVar('Q')


class Cfg(Generic[T]):
    graph: nx.DiGraph

    def __init__(self, graph: nx.DiGraph | dict | list[tuple[int, int, dict]], blocks=None) -> None:
        if isinstance(graph, nx.DiGraph):
            self.graph = graph
        else:
            self.graph = nx.DiGraph(graph)
        if blocks is not None:
            set_node_attributes(self, name='block', values=blocks)

    @property
    def entry(self):
        return self.graph.nodes[0]

    @property
    def nodes(self):
        return self.graph.nodes

    def reverse(self: Cfg[T], copy) -> Cfg[T]:
        return Cfg(self.graph.reverse(copy=copy))

    def draw(self) -> None:
        import matplotlib.pyplot as plt
        nx.draw_networkx(self.graph, with_labels=True)
        plt.show()

    def print_graph(self) -> None:
        for b in sorted(self.graph.nodes()):
            print(b, ':', self.graph.nodes[b]['block'])

    def predecessors(self, label):
        return self.graph.predecessors(label)

    def successors(self, label):
        return self.graph.successors(label)

    def copy(self: Cfg) -> Cfg:
        return Cfg(self.graph.copy())


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
    g.nodes[n]['block'] will hold list of dictionaries.
    """
    blockname = 'block'
    g = cfg.graph
    starts = set(chain((n for n in g if g.in_degree(n) != 1),
                       chain.from_iterable(g.successors(n) for n in g
                                           if g.out_degree(n) > 1)))
    blocks = nx.DiGraph()
    for label in starts:
        n = label
        block = []
        while True:
            block += g.nodes[n][blockname]
            if g.out_degree(n) != 1:
                break
            next_n = next(iter(g.successors(n)))
            if g.in_degree(next_n) != 1:
                break
            n = next_n
        if label not in blocks:
            blocks.add_node(label)
        blocks.add_edges_from((label, suc) for suc in g.successors(n))
        blocks.nodes[label][blockname] = block
    return Cfg(blocks)


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


def node_data_map(cfg: Cfg[T], f: Callable[[int, list[T]], list[Q]]) -> Cfg[Q]:
    cfg: Cfg[Any] = cfg.copy()
    for n, data in cfg.nodes.items():
        data['block'] = f(n, data['block'])
    return cfg


def test_refine_to_chain():
    import code_examples
    tac_name = 'tac_block'
    from tac_analysis import make_tacblock_cfg
    cfg = make_tacblock_cfg(code_examples.CreateSphere)
    refined = refine_to_chain(cfg, tac_name, 'tac')
    for x in sorted(refined.nodes_iter()):
        print(x, refined.nodes[x]['tac'].format())


if __name__ == '__main__':
    test_refine_to_chain()


def single_source_dijkstra_path_length(cfg: Cfg, source: int, weight='weight'):
    return nx.single_source_dijkstra_path_length(cfg.graph, source, weight=weight)


def set_node_attributes(cfg: Cfg, values, name):
    return nx.set_node_attributes(cfg.graph, values, name)
