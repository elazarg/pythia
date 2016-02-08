import networkx as nx
import bcode


def accumulate_stack_depth(cfg):
    # edge_dfs() will give us edges to nodes we've already saw,
    # allowing us to validate that the stack is sensible on all paths.
    # Technically, we should find "leaders" - including first instruction in dead code
    # it means that we should "predict" what the starting flow for such leaders is...
    # I think it complicates matters, and there is no simple solution that will keep the loop invariant.
    cfg.node[0]['depth_in'] = 0 
    for src, dst in nx.edge_dfs(cfg, 0):
        stack_effect = cfg.edge[src][dst]['stack_effect']
        dst_in_prev = cfg.node[dst].get('depth_in')
        dst_in = cfg.node[src]['depth_in'] + stack_effect
        cfg.node[dst]['depth_in'] = dst_in
        assert dst_in_prev in [None, dst_in]
        assert dst_in >= 0


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
                             cfg.node[n].get('depth_in'))
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
        blocks[label]['block'] = block
    for n in blocks:
        print(n, ':', blocks[n]['block'])
    return blocks


def make_graph(f):
    dbs = dict(bcode.get_instructions(f))
    cfg = nx.DiGraph([(b.offset, dbs[j].offset, {'stack_effect': stack_effect})
                    for offset, b in dbs.items()# this ^ should be called weight to be used in algorithms
                    for (j, stack_effect) in b.next_list() if dbs.get(j)])
    for offset, b in dbs.items():
        cfg.node[offset]['ins'] = b
    accumulate_stack_depth(cfg)
    return cfg_to_basic_blocks(cfg)


def print_graph(cfg):
    for b in sorted(cfg.node):
        print(b, ':', cfg[b].get('depth_in', 'DEAD CODE'), ' <- ', cfg[b]['block'])


def draw(g: nx.DiGraph):
    import matplotlib.pyplot as plt
    nx.draw_networkx(g, with_labels=True)
    plt.show()


def test():
    import code_examples
    import cfg
    cfg = cfg.make_graph(code_examples.CreatePlasmaCube)
    print_graph(cfg)

    
if __name__ == '__main__':   
    test()
