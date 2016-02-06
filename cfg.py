import networkx as nx
import bcode


def accumulate_stack_depth(cfg):
    # edge_dfs() will give us edges to nodes we've already saw,
    # allowing us to validate that the stack is sensible on all paths
    cfg.node[0]['depth_in'] = 0 
    for src, dst in nx.edge_dfs(cfg, 0):
        stack_effect = cfg.edge[src][dst]['stack_effect']
        dst_in_prev = cfg.node[dst].get('depth_in')
        dst_in = cfg.node[src]['depth_in'] + stack_effect
        cfg.node[dst]['depth_in'] = dst_in
        assert dst_in_prev in [None, dst_in]
        assert dst_in >= 0


def make_graph(f):
    bs = bcode.get_instructions(f)
    dbs = dict((b.offset, b) for b in bs)
    cfg = nx.DiGraph([(b.offset, dbs[j].offset, {'stack_effect': stack_effect})
                    for b in bs             # this ^ should be called weight to be used in algorithms
                    for (j, stack_effect) in b.next_list() if dbs.get(j)])
    for b in bs:
        cfg.node[b.offset]['ins'] = b
    accumulate_stack_depth(cfg)
    return cfg


def print_graph(bs, cfg):
    for b in bs:
        print(b.offset, ':', b.next_list(), cfg.node[b.offset].get('depth_in', 'DEAD CODE'), ' <- ', b)


def draw(g: nx.DiGraph):
    import matplotlib.pyplot as plt
    nx.draw_networkx(g, with_labels=True)
    plt.show()


if __name__ == '__main__':   
    pass
