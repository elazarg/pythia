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
    dbs = dict((x[0].offset, x) for x in blocks)
    return nx.DiGraph([(t[0].offset, dbs.get(j)[0].offset)
                    for t in blocks
                    for (j, _se) in t[-1].next_list() if dbs.get(j)])


def build_cfg(f):
    blocks = basic_block.prepare(f)
    cfg = make_graph(blocks)
    # edge_dfs() will give us edges to nodes we've already saw,
    # allowing us to validate that the stack is sensible on all paths 
    for e in nx.edge_dfs(cfg, -1):
        print(e)
    return cfg


def draw(g: nx.DiGraph):
    import matplotlib.pyplot as plt
    nx.draw_networkx(g, with_labels=True)
    plt.show()

   
def test():
    cfg = build_cfg(utils.partition)
    draw(cfg)


if __name__ == '__main__':   
    test()
