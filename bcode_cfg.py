import networkx as nx
import bcode
import graph_utils as gu

def accumulate_stack_depth(cfg, source, into_name, weight):
    '''The stack depth supposed to be independent of path.
    So dijkstra on the undirected graph suffices (and maybe too strong. we don't need minimality)
    The `undirected` part is just because we want to work
    with unreachable code too. 
    '''
    depths = nx.single_source_dijkstra_path_length(
              gu.copy_to_bidirectional(cfg), source=source, weight=weight)
    nx.set_node_attributes(cfg, name=into_name, values=depths)


def make_cfg(f, blockname):
    dbs = {b.offset: b for b in bcode.get_instructions(f)}
    cfg = nx.DiGraph([(b.offset, dbs[j].offset, {'weight': stack_effect})
                    for b in dbs.values()
                    for (j, stack_effect) in b.next_list() if dbs.get(j)])
    nx.set_node_attributes(cfg, name='BCode', values=dbs)
    
    accumulate_stack_depth(cfg, source=0, into_name='tos', weight='weight')
    
    pairs_graph = gu.node_data_map(cfg, f=lambda d:(d['BCode'], d['tos']))
    
    basic_block_cfg = gu.contract_chains(pairs_graph, blockname=blockname)
    
    return basic_block_cfg


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
    cfg = make_cfg(code_examples.CreatePlasmaCube, blockname=name)
    print_graph(cfg, code=name)

if __name__ == '__main__':   
    test()
