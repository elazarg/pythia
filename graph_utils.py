import networkx as nx
from itertools import chain


def reverse_weights(g: nx.DiGraph, weight='weight'):
    g = g.reverse()
    for s, t in g.edges():
        e = g[s][t]
        e[weight] = -e[weight]
    return g


def copy_to_bidirectional(g: nx.DiGraph, weight='weight'):
    return nx.compose(g, reverse_weights(g, weight=weight))


def contract_chains(g: nx.DiGraph, blockname='block'):
    """Contract chains with indegree=outdegree=1:
    i.e. turns > - - - <
         into  [>---<]
    The label of the chain is the label of its first element.
    g.nodes[n][blockname] will hold list of dictionaries.
    """
    starts = set(chain((n for n in g if g.in_degree(n) != 1),
                       chain.from_iterable(g.successors(n) for n in g
                                           if g.out_degree(n) > 1)))
    blocks = nx.DiGraph()
    for label in starts:
        n = label
        block = []
        while True:
            block.append(g.nodes[n])
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
    return blocks


def node_data_map_inplace(g, f, attr=None):
    if attr is None:
        for n in g.nodes_iter():
            g.nodes[n] = f(n, g.nodes[n])
    else:
        for n, data in g.nodes.items():
            data[attr] = f(n, data)


def refine_to_chain(g, from_attr, to_attr):
    """can be used to refine basic blocks into blocks - the dual of contract_chains()
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


def node_data_map(g, f, attr=None):
    g = g.copy()
    node_data_map_inplace(g, f, attr)
    return g


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
