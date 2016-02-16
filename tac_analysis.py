# TODO: full graph constant/copy propagation
import graph_utils

def print_block(n, block):
    print(n, ':')
    for ins in block:
        print(ins.format())
    
def print_tac_cfg(tac_cfg, blockname):
    for n in sorted(tac_cfg.nodes()):
        print("succ:", tac_cfg.successors(n))
        block = tac_cfg.node[n][blockname]
        print_block(n, block)
    
def simplify_tac_block(n, block_data, block_name):
    #print("block", str(block))
    #assert False
    block = block_data[block_name]
    single_block_constant_prpagation_update(block)
    block = list(single_block_kills(block)); block.reverse()
    single_block_constant_prpagation_update(block)
    block = list(single_block_kills(block)); block.reverse()
    return {block_name: block}
    
def simplified_tac_blocks_cfg(tac_cfg, blockname):
    return graph_utils.node_data_map(tac_cfg, lambda n, block_data: simplify_tac_block(n, block_data, blockname))

def test():
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


def single_block_constant_prpagation_update(block):
    cons_map = {}
    for i, ins in enumerate(block):
        if ins.is_assign and len(ins.gens) == len(ins.uses) == 1:    
            [lhs], [rhs] = ins.gens, ins.uses
            if rhs in cons_map:
                rhs = cons_map[rhs]
            cons_map[lhs] = rhs
        else:
            block[i] = ins._replace(uses=[cons_map.get(v, v)
                                          for v in ins.uses])
            for v in ins.gens:
                if v in cons_map:
                    del cons_map[v]


def chaotic(g:'graph', s:'node', lattice, f):
    'Not working. Only here to remember...'
    entry = {v: lattice.BOT
             for v in g.nodes()}
    entry[s] = lattice.EMPTYSET
    wl = set([s])
    while wl:
        u = wl.pop()
        for v in g.successors(u):
            t = f(g.edges(), entry(u))
            new = lattice.join(entry[v], t)
            if new != entry[v]:
                entry[v] = new
                wl.add(v)


def is_extended_identifier(name):
    return name.replace('.', '').isidentifier()


if __name__ == '__main__':
    test()
