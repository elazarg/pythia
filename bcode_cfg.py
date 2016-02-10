import networkx as nx
import bcode
import utils


def accumulate_stack_depth(cfg, weight='weight', source=0):
    '''The stack depth supposed to be independent of path.
    So dijkstra on the undirected graph suffices.
    The `undirected` part is just because we want to work
    with unreachable code too. 
    '''
    lengths = nx.single_source_dijkstra_path_length(
              utils.copy_to_bidirectional(cfg), source=source, weight=weight)
    nx.set_node_attributes(cfg, name='tos', values=lengths)
    print(cfg.nodes(data=True))


def cfg_to_basic_blocks(cfg):
    '''Finds basic blocks in the graph - contracting chains
    there's (almost) nothing special about cfg in particular.
    '''
    from itertools import chain
    #partition() is not complete since it does not hangle JUMP_FORWARD etc.
    starts = set(chain( (n for n in cfg if cfg.in_degree(n) != 1),
                 chain.from_iterable(cfg.successors_iter(n) for n in cfg
                                     if cfg.out_degree(n) > 1)))
    import tac
    blocks = nx.DiGraph()
    for n in starts:
        label = n
        block = []
        while True:
            bcode = cfg.node[n]['ins']
            r = tac.make_TAC(bcode.opname, bcode.argval, bcode.stack_effect(),
                             cfg.node[n]['tos'])
            block.extend(r)
            if cfg.out_degree(n) != 1:
                break
            next_n = next(cfg.successors_iter(n))
            if cfg.in_degree(next_n) > 1:
                break
            n = next_n
        if label not in blocks:
            blocks.add_node(label)
        blocks.add_edges_from( (label, suc) for suc in cfg.successors_iter(n))
        blocks.node[label]['block'] = block
    # for n in blocks: print(n, ':', blocks[n]['block'])
    return blocks


def make_graph(f):
    dbs = dict(bcode.get_instructions(f))
    cfg = nx.DiGraph([(b.offset, dbs[j].offset, {'weight': stack_effect})
                    for offset, b in dbs.items()
                    for (j, stack_effect) in b.next_list() if dbs.get(j)])
    accumulate_stack_depth(cfg)
    for offset, b in dbs.items():
        cfg.node[offset]['ins'] = b
    return cfg_to_basic_blocks(cfg)


def print_graph(cfg, code):
    for b in sorted(cfg.nodes()):
        print(b, ':', cfg.node[b][code])


def draw(g: nx.DiGraph):
    import matplotlib.pyplot as plt
    nx.draw_networkx(g, with_labels=True)
    plt.show()


def test():
    import code_examples
    cfg = make_graph(code_examples.CreatePlasmaCube)
    print_graph(cfg, code='block')
    

    
if __name__ == '__main__':   
    test()
