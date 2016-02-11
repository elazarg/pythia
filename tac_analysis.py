# TODO: full graph constant/copy propagation

def print_block(n, block):
    print(n, ':')
    for ins in block:
        print('\t', ins.format())


def test_single_block():
    import code_examples
    import tac
    name = 'tac_block'
    cfg = tac.make_tacblock_cfg(code_examples.RenderScene, blockname=name)
    for n in sorted(cfg.nodes()):
        block = cfg.node[n][name]
        print('uses:', single_block_uses(block))
        # print_block(n, block)
        # print('push up:')
        single_block_constant_prpagation_update(block)
        block = list(single_block_kills(block)); block.reverse()
        single_block_constant_prpagation_update(block)
        block = list(single_block_kills(block)); block.reverse()
        print_block(n, block)

def test():
    import code_examples
    import tac
    name = 'tac_block'
    cfg = tac.make_tacblock_cfg(code_examples.RayTrace, blockname=name)
    dataflow(cfg, 0, {})
    for n in sorted(cfg.nodes()):
        block = cfg.node[n][name]
        block = list(single_block_kills(block)); block.reverse()
        print_block(n, block)

    dataflow(cfg, 0, {})
    for n in sorted(cfg.nodes()):
        block = cfg.node[n][name]
        block = list(single_block_kills(block)); block.reverse()
        print_block(n, block)

def single_block_uses(block):
    uses = set()
    for ins in reversed(block):
        uses.difference_update(ins.gens)
        uses.update(ins.uses)
    return [x for x in uses if is_extended_identifier(x)]


def undef(kills, gens):
    return [('_' if v in kills else v) for v in gens]


def _filter_killed(ins, kills, new_kills):
    # moved here only because it is a transformation and not an analysis
    if ins.is_del or ins.is_assign and set(ins.gens).issubset(kills):
        return
    yield ins._replace(gens=undef(kills, ins.gens),
                       kills=kills - new_kills)


def single_block_kills(block, kills=frozenset()):
    'kills: the set of names that will no longer be used'
    for ins in reversed(block):
        new_kills = kills.union(ins.kills).difference(ins.uses)
        yield from _filter_killed(ins, kills, kills.difference(ins.uses))
        kills = new_kills


def single_block_gens(block, inb=frozenset()):
    gens = set()
    for ins in block:
        gens.difference_update(ins.kills)
        gens.update(ins.gens)
    return [x for x in gens if is_extended_identifier(x)]


def single_block_constant_prpagation_update(block, in_cons_map):
    cons_map = in_cons_map.copy()
    for i, ins in enumerate(block):
        if ins.is_assign and len(ins.gens) == len(ins.uses) == 1:
            [lhs], [rhs] = ins.gens, ins.uses
            if rhs in cons_map:
                rhs = cons_map[rhs]
            cons_map[lhs] = rhs
        elif not ins.is_inplace:
            block[i] = ins._replace(uses=[cons_map.get(v, v)
                                          for v in ins.uses])
            for v in ins.gens:
                if v in cons_map:
                    del cons_map[v]
    return cons_map


# mix of domain and analysis-specific choice of operations
class Domain:
    name = ''
    direction = (1, -1)
    def transfer(self): pass
    
    @classmethod
    def unify(self, b):
        'meet or join'
        pass
    
    @classmethod
    def join(self): pass
    
    @classmethod
    def meet(self): pass
    
    def __init__(self): pass
    
    def is_full(self): pass

class ConsProp(Domain): pass

def is_extended_identifier(name):
    return name.replace('.', '').isidentifier()


def run_analysis(cfg):
    import graph_utils as gu
    import networkx as nx
    
    Analysis = ConsProp 
    def compute_transfer_function(): pass
    gu.node_data_map_inplace(cfg, attr='transfer',
                             f=lambda n, d: compute_transfer_function(d))
    nx.set_node_attributes(cfg, 'out', {n:Analysis() for n in cfg.nodes_iter()})
    dataflow(cfg, 0, Analysis)


def dataflow(g:'graph', start:'node', start_value):
    import networkx as nx
    import graph_utils as gu
    
    def join(res, cms):
        for cm in cms[1:]:
            for v, c in cm.items():
                if v not in res:
                    res[v] = c
                elif res[v] != c:
                    del res[v]
    
    gu.node_data_map_inplace(g,
            f=lambda n, d: single_block_constant_prpagation_update,
            attr='transfer_function')
    
    nx.set_node_attributes(g, 'outb', {v: {} for v in g.nodes()})
    nx.set_node_attributes(g, 'inb', {v: {} for v in g.nodes()})
    g.node[start]['inb'] = {}
    wl = set(g.nodes())
    while wl:
        u = wl.pop()
        inb = g.node[u]['inb']
        join(inb, [g.node[x]['outb'] for x in g.predecessors(u)])
        outb = single_block_constant_prpagation_update(g.node[u]['tac_block'], inb)
        if outb != g.node[u]['outb']:
            g.node[u]['outb'] = outb
            wl.update(g.successors(u))

if __name__ == '__main__':
    test()
