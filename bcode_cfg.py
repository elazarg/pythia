import networkx as nx
import bcode
import utils

def accumulate_stack_depth(cfg, source, into_name, weight):
    '''The stack depth supposed to be independent of path.
    So dijkstra on the undirected graph suffices.
    The `undirected` part is just because we want to work
    with unreachable code too. 
    '''
    lengths = nx.single_source_dijkstra_path_length(
              utils.copy_to_bidirectional(cfg), source=source, weight=weight)
    nx.set_node_attributes(cfg, name=into_name, values=lengths)


def make_graph(f, blockname):
    dbs = dict(bcode.get_instructions(f))
    cfg = nx.DiGraph([(b.offset, dbs[j].offset, {'weight': stack_effect})
                    for offset, b in dbs.items()
                    for (j, stack_effect) in b.next_list() if dbs.get(j)])
    for offset, b in dbs.items():
        cfg.node[offset]['BCode'] = b
    accumulate_stack_depth(cfg, source=0, into_name='tos', weight='weight')
    return cfg_to_basic_blocks(cfg, blockname=blockname,
                               data_map=lambda d: [(d['BCode'], d['tos'])])


def print_graph(cfg, code):
    for b in sorted(cfg.nodes()):
        print(b, ':', cfg.node[b][code])


def draw(g: nx.DiGraph):
    import matplotlib.pyplot as plt
    nx.draw_networkx(g, with_labels=True)
    plt.show()


def test():
    import code_examples
    name = 'bcode_block'
    cfg = make_graph(code_examples.CreatePlasmaCube, blockname=name)
    print_graph(cfg, code=name)
    

def cfg_to_basic_blocks(cfg, blockname='block', data_map:'data to list'=lambda x:[x]):
    '''I Tried to make it "general chain contraction algorithm"...
    Finds basic blocks in the graph.
    there's (almost) nothing special about cfg in particular.
    '''
    from itertools import chain
    starts = set(chain((n for n in cfg if cfg.in_degree(n) != 1),
                 chain.from_iterable(cfg.successors_iter(n) for n in cfg
                                     if cfg.out_degree(n) > 1)))
    blocks = nx.DiGraph()
    for label in starts:
        n = label
        block = []
        while True:
            block.extend(data_map(cfg.node[n]))
            if cfg.out_degree(n) != 1:
                break
            next_n = next(cfg.successors_iter(n))
            if cfg.in_degree(next_n) > 1:
                break
            n = next_n
        if label not in blocks:
            blocks.add_node(label)
        blocks.add_edges_from((label, suc) for suc in cfg.successors_iter(n))
        blocks.node[label][blockname] = block
    return blocks
    
if __name__ == '__main__':   
    test()
