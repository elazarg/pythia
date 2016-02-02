import utils
import networkx as nx
import basic_block

'''
"roadmap": we need to reason about the depth of the stack in the CFG.
TODO:
1. add weights on the edges, taken from BasicBlock.next_list
2. compute and keep the absolute depth of the stack at each point using edge_dfs() as below
3. ???
4. profit
'''

def make_graph(blocks):
    dbs = dict((b.offset, b) for b in blocks)
    cfg = nx.DiGraph([(b.offset, dbs[j].offset, {'stack_effect': stack_effect})
                    for b in blocks             # this ^ should be called weight to be used in algorithms
                    for (j, stack_effect) in b.next_list() if dbs.get(j)])
    for b in blocks:
        cfg.node[b.offset]['block'] = b
    return cfg

def build_cfg(f):
    blocks = basic_block.prepare(f)
    basic_block.print_blocks(blocks)
    cfg = make_graph(blocks)
    # edge_dfs() will give us edges to nodes we've already saw,
    # allowing us to validate that the stack is sensible on all paths
    cfg.node[-1]['sin'] = 0 
    for src, dst, edge in edge_dfs(cfg):
        dst['sin'] = src['sin'] + edge['stack_effect']
        assert dst['sin'] >= 0
    return cfg


def edge_dfs(g, start=-1):
    return ( ( g.node[src_id],
               g.node[dst_id],
               g.edge[src_id][dst_id]  )
             for src_id, dst_id in nx.edge_dfs(g, -1))
    
def draw(g: nx.DiGraph):
    import matplotlib.pyplot as plt
    nx.draw_networkx(g, with_labels=True)
    plt.show()

   
def test():
    cfg = build_cfg(utils.partition)
    #draw(cfg)


if __name__ == '__main__':   
    test()
