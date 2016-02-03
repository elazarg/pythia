import networkx as nx


def accumulate_stack_depth(cfg):
    # edge_dfs() will give us edges to nodes we've already saw,
    # allowing us to validate that the stack is sensible on all paths
    cfg.node[-1]['depth_in'] = 0 
    for src, dst in nx.edge_dfs(cfg, -1):
        se = cfg.edge[src][dst]['stack_effect']
        dst_in_prev = cfg.node[dst].get('depth_in')
        dst_in = cfg.node[src]['depth_in'] + se
        cfg.node[dst]['depth_in'] = dst_in
        assert dst_in_prev in [None, dst_in]
        assert dst_in >= 0


def make_graph(f):
    import bcode
    bs = bcode.get_instructions(f)
    dbs = dict((b.offset, b) for b in bs)
    cfg = nx.DiGraph([(b.offset, dbs[j].offset, {'stack_effect': stack_effect})
                    for b in bs             # this ^ should be called weight to be used in algorithms
                    for (j, stack_effect) in b.next_list() if dbs.get(j)])
    for b in bs:
        cfg.node[b.offset]['ins'] = b
    accumulate_stack_depth(cfg)
    for b in bs:
        print(b.offset, ':', b.next_list(), cfg.node[b.offset].get('depth_in', 'DEAD CODE'), ' <- ', b)
    return cfg


def draw(g: nx.DiGraph):
    import matplotlib.pyplot as plt
    nx.draw_networkx(g, with_labels=True)
    plt.show()

   
def test():
    import utils
    def example(x):
        len([])
        while x:
            if x:
                break
            dir()
        else:
            print(1)
        for y in [2, 3]:
            if y:
                break
            print(y**2)
        else:
            print(7)
        example(5 if x else x+x)
    cfg = make_graph(example)
    #draw(cfg)


if __name__ == '__main__':   
    test()
