import networkx as nx

def reverse_weights(g:nx.DiGraph, weight='weight'):
    g = g.reverse()
    for s, t in g.edges_iter():
        e = g[s][t]
        e[weight] = -e[weight]
    return g


def copy_to_bidirectional(g:nx.DiGraph, weight='weight'):
    return nx.compose(g, reverse_weights(g, weight=weight))


def contract_chains(g:nx.DiGraph, blockname='block'):
    '''Contract chains with indegree=outdegree=1:
    i.e. turns > - - - <
         into  [>---<]  
    The label of the chain is the label of its first element.
    g.node[n][blockname] will hold list of dictionaries.
    '''
    from itertools import chain
    starts = set(chain((n for n in g if g.in_degree(n) != 1),
                       chain.from_iterable(g.successors_iter(n) for n in g
                                           if g.out_degree(n) > 1)))
    blocks = nx.DiGraph()
    for label in starts:
        n = label
        block = []
        while True:
            block.append(g.node[n])
            if g.out_degree(n) != 1:
                break
            next_n = next(g.successors_iter(n))
            if g.in_degree(next_n) != 1:
                break
            n = next_n
        if label not in blocks:
            blocks.add_node(label)
        blocks.add_edges_from((label, suc) for suc in g.successors_iter(n))
        blocks.node[label][blockname] = block
    return blocks


def node_data_map_inplace(g, f, attr=None):
    if attr is None:
        for n in g.nodes_iter():
            g.node[n] = f(n, g.node[n])
    else:
        for n in g.nodes_iter():
            data = g.node[n]
            data[attr] = f(n, data)


def node_data_map(g, f, attr=None):
    g = g.copy()
    node_data_map_inplace(g, f, attr)
    return g
