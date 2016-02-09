import tac, itertools as it
import utils

def print_block(n, block):
    print(n, ':')
    for ins in block:
        cmd = ins.fmt.format(**ins._asdict())
        print(cmd, '\t\t', '' and ins)

def test():
    import code_examples
    import cfg
    cfg = cfg.make_graph(code_examples.main)
    # import tac
    # tac.print_3addr(cfg)
    for n in sorted(cfg.node):
        print('analyzed:')
        block = cfg[n]['block']
        print(single_block_uses(block))
        print_block(n, block)
        single_block_constant_prpagation_update(block)
        print('push up:')
        block = list(single_block_kills(block))
        block.reverse()
        print_block(n, block)


def single_block_uses(block):
    uses = set()
    for ins in reversed(block):
        uses.difference_update(ins.gens)
        uses.update(ins.uses)
    return [x for x in uses if is_extended_identifier(x)]

def undef(kills, gens):
    return [('_' if v in kills else v) for v in gens]


def _filter_killed(ins, kills, gens, new_kills):
    #moved here only because it is a transformation and not an analysis
    if isinstance(ins, tac.Del)\
        or isinstance(ins, tac.Assign) and kills.issuperset(ins.gens):
        return
    diff = kills - new_kills
    if diff:
        yield tac.delete(*diff)
    yield ins._replace(gens=undef(kills, gens))

def single_block_kills(block, kills=frozenset()):
    for ins in reversed(block):
        gens = ins.gens
        ins_kills = getattr(ins, 'kills', ins.gens)
        new_kills = kills.union(ins_kills).difference(ins.uses)
        yield from _filter_killed(ins, kills, gens, new_kills)
        kills = new_kills


def single_block_gens(block, inb=frozenset()):
    gens = set()
    for ins in block:
        gens.difference_update(getattr(ins, 'kills', ()))
        gens.update(ins.gens)
    return [x for x in gens if is_extended_identifier(x)]

def single_block_constant_prpagation_update(block):
    from tac import Assign
    cons_map = {}
    for i, ins in enumerate(block):
        if isinstance(ins, Assign) and len(ins.gens) == len(ins.uses) == 1:    
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


def chaotic(g:'graph', s:'node', lattice, yota, f):
    entry = {v: lattice.BOT
             for v in g.nodes()}
    entry[s] = lattice.yota
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
