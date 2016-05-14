# Data flow analysis and stuff.
# The work here is very very "half baked" to say the least, and is very messy,
# mixing notations and ideas, and some of the functions are unused and/or untested
# What seems to be working: 
# 1. constant propagation, to some degree 
# 2. single-block liveness
# These are important in order to make the TAC at least *look* different than 
# stack-oriented code, and it removes many variables.
# The analysis is based heavily on the information in the tac module - some of
# it is dedicated for the analysis
#
# A note about naming:
# Different sources give different names to gen/def/use/live/kill variables.
# here:
#    1. USES is the list of variables used by an instruction/block.
#       e.g. `x = f(a, b)` : USES=(f, a, b)   
#    2. GENS is the list of variables defined by an insruction/block
#       e.g. `x = f(a, b)` : GENS=(x,)
#    3. KILLS is the list of variables killed by an instruction/block, which do not appear in GENS
#       For most of the instructions, initially, KILLS is empty.
#       However, `DEL x` will have KILLS={x} but empty GENS
#       In addition, the per-block live variable analysis removes DEL operations,
#       and push them into each other and into other instructions.
#       So in some cases, `x = f(a, b)` might have e.g. KILLS={a}
#       (and implicitly x too) if it is the last command to use the varible `a`.
import tac
from tac import is_stackvar
from itertools import chain

BLOCKNAME = tac.BLOCKNAME

def make_tacblock_cfg(f, propagate_consts=True, liveness=True):
    cfg = tac.make_tacblock_cfg(f)
    if propagate_consts:
        dataflow(cfg, 0, {}, ConsProp)
    if liveness:
        for n in sorted(cfg.nodes()):
            block = cfg.node[n][BLOCKNAME]
            block = list(single_block_liveness(block))
            block.reverse()
            cfg.node[n][BLOCKNAME] = block
    return cfg

def print_block(n, block):
    print(n, ':')
    for ins in block:
        print('\t', ins.format())


def test_single_block():
    import code_examples
    cfg = tac.make_tacblock_cfg(code_examples.RenderScene)
    for n in sorted(cfg.nodes()):
        block = cfg.node[n][BLOCKNAME]
        print('uses:', single_block_uses(block))
        # print_block(n, block)
        # print('push up:')
        ConsProp.single_block_update(block)
        block = list(single_block_liveness(block)); block.reverse()
        ConsProp.single_block_update(block)
        block = list(single_block_liveness(block)); block.reverse()
        print_block(n, block)

def test():
    import code_examples
    cfg = make_tacblock_cfg(code_examples.simple)
    for n in sorted(cfg.nodes()):
        block = cfg.node[n][BLOCKNAME]
        print_block(n, block)


def single_block_uses(block):
    uses = set()
    for ins in reversed(block):
        uses.difference_update(ins.gens)
        uses.update(ins.uses)
    return tuple(x for x in uses if is_extended_identifier(x))


def undef(kills, gens):
    return tuple(('_' if v in kills and tac.is_stackvar(v) else v)
                 for v in gens)


def _filter_killed(ins, kills, new_kills):
    # moved here only because it is a transformation and not an analysis
    if ins.is_del() or ins.is_assign() and set(ins.gens).issubset(kills):
        return
    yield ins._replace(gens=undef(kills, ins.gens),
                       kills=kills - new_kills)


def single_block_liveness(block, kills=frozenset()):
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


# mix of domain and analysis-specific choice of operations
# nothing here really works...
class Domain:
    name = ''
    direction = (1, -1)
    def transfer(self): pass
    
    @staticmethod
    def unify(self, b):
        'meet or join'
        pass
    
    @staticmethod
    def join(self): pass
    
    @staticmethod
    def meet(self): pass
    
    def __init__(self): pass
    
    def is_full(self): pass


class ConsProp(Domain):

    @staticmethod
    def initial():
        return {}

    @staticmethod
    def join(res, cms):
        for cm in cms[1:]:
            for v, c in cm.items():
                if v not in res:
                    res[v] = c
                elif res[v] != c:
                    del res[v]

    @staticmethod
    def single_block_update(block, in_cons_map):
        cons_map = in_cons_map.copy()
        for i, ins in enumerate(block):
            if ins.is_assign() and len(ins.gens) == len(ins.uses) == 1 and is_stackvar(ins.gens[0]):
                [lhs], [rhs] = ins.gens, ins.uses
                if rhs in cons_map:
                    rhs = cons_map[rhs]
                cons_map[lhs] = rhs
            else:
                uses = [(cons_map.get(v, v))
                         for v in ins.uses]
                uses = tuple(uses)
                block[i] = ins._replace(uses=tuple(uses))
                for v in chain(ins.gens, ins.kills):
                    if v in cons_map:
                        del cons_map[v]
        return cons_map

class Liveness(Domain):
    
    @staticmethod
    def initial():
        return set()
    
    @staticmethod
    def single_block(block, live=frozenset()):
        'kills: the set of names that will no longer be used'
        for ins in reversed(block):
            live = frozenset(ins.gens).union(live.difference(ins.uses))

    @staticmethod
    def single_block_total_effect(block, live=frozenset()):
        'kills: the set of names that will no longer be used'
        gens = {ins.gens for ins in block}
        uses = {ins.uses for ins in block}
        return uses.union(live.difference(gens)) 


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
    dataflow(cfg, 0, {}, Analysis)


def dataflow(g:'graph', start:'node', start_value, Analysis=ConsProp):
    import networkx as nx
    import graph_utils as gu
    gu.node_data_map_inplace(g,
            f=lambda n, d: Analysis.single_block_update,
            attr='transfer_function')
    
    nx.set_node_attributes(g, 'outb', {v: Analysis.initial() for v in g.nodes()})
    nx.set_node_attributes(g, 'inb', {v: Analysis.initial() for v in g.nodes()})
    g.node[start]['inb'] = Analysis.initial()
    wl = set(g.nodes())
    while wl:
        u = wl.pop()
        udata = g.node[u]
        inb = udata['inb']
        Analysis.join(inb, [g.node[x]['outb'] for x in g.predecessors(u)])
        outb = udata['transfer_function'](udata[BLOCKNAME], inb)
        if outb != udata['outb']:
            udata['outb'] = outb
            wl.update(g.successors(u))


if __name__ == '__main__':
    test()
