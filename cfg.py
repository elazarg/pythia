import networkx as nx
import basic_block


def make_graph(blocks):
    dbs = dict((b.offset, b) for b in blocks)
    cfg = nx.DiGraph([(b.offset, dbs[j].offset, {'stack_effect': stack_effect})
                    for b in blocks             # this ^ should be called weight to be used in algorithms
                    for (j, stack_effect) in b.next_list() if dbs.get(j)])
    for b in blocks:
        cfg.node[b.offset]['block'] = b
    return cfg


def accumulate_stack_depth(cfg):
    # edge_dfs() will give us edges to nodes we've already saw,
    # allowing us to validate that the stack is sensible on all paths
    cfg.node[-1]['block'].depth_in = 0 
    for src, dst in nx.edge_dfs(cfg, -1):
        se = cfg.edge[src][dst]['stack_effect']
        dst_in = cfg.node[dst]['block'].depth_in  = cfg.node[src]['block'].depth_in + se
        assert dst_in >= 0

 
def build_cfg(f):
    blocks = basic_block.prepare(f)
    basic_block.print_blocks(blocks)
    cfg = make_graph(blocks)
    accumulate_stack_depth(cfg)
    return cfg

    
def draw(g: nx.DiGraph):
    import matplotlib.pyplot as plt
    nx.draw_networkx(g, with_labels=True)
    plt.show()

   
def test():
    import utils
    def example(x):
        while True: 
            break
        else:
            print(1)
        example(5 if x else x+x)
    cfg = build_cfg(example)
    #draw(cfg)


if __name__ == '__main__':   
    test()
