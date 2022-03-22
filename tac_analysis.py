# Data flow analysis and stuff.
# The work here is very "half-baked" to say the least, and is very messy,
# mixing notations and ideas, and some functions are unused and/or untested
# What seems to be working: 
# 1. constant propagation, to some degree 
# 2. single-block liveness
# These are important in order to make the TAC at least *look* different from
# stack-oriented code, and it removes many variables.
# The analysis is based heavily on the information in the tac module - some of
# it is dedicated for the analysis
#
# A note about naming:
# Different sources give different names to gen/def/use/live/kill variables.
# here:
#    1. USES is the list of variables used by an instruction/block.
#       e.g. `x = f(a, b)` : USES=(f, a, b)   
#    2. GENS is the list of variables defined by an instruction/block
#       e.g. `x = f(a, b)` : GENS=(x,)
#    3. KILLS is the list of variables killed by an instruction/block, which do not appear in GENS
#       For most of the instructions, initially, KILLS is empty.
#       However, `DEL x` will have KILLS={x} but empty GENS
#       In addition, the per-block live variable analysis removes DEL operations,
#       and push them into each other and into other instructions.
#       So in some cases, `x = f(a, b)` might have e.g. KILLS={a}
#       (and implicitly x too) if it is the last command to use the variable `a`.
from __future__ import annotations
from typing import Protocol
from itertools import chain

import networkx as nx
import typing

import graph_utils as gu
import tac
from tac import is_stackvar

BLOCKNAME = tac.BLOCKNAME


def make_tacblock_cfg(f, propagate_consts=True, liveness=True, propagate_assignments=True):
    cfg = tac.make_tacblock_cfg(f)
    if propagate_consts:
        dataflow(cfg, ConstantPropagation)
    if liveness:
        for n in sorted(cfg.nodes()):
            block = cfg.nodes[n][BLOCKNAME]
            block = list(single_block_liveness(block))
            block.reverse()
            cfg.nodes[n][BLOCKNAME] = block
    return cfg


def print_block(n, block):
    print(n, ':')
    for ins in block:
        print('\t', ins.format())


def test_single_block():
    import code_examples
    cfg = tac.make_tacblock_cfg(code_examples.RenderScene)
    for n in sorted(cfg.nodes()):
        block = cfg.nodes[n][BLOCKNAME]
        print('uses:', single_block_uses(block))
        # print_block(n, block)
        # print('push up:')
        ConstantPropagation.single_block_update(block)
        block = list(single_block_liveness(block))
        block.reverse()
        ConstantPropagation.single_block_update(block)
        block = list(single_block_liveness(block))
        block.reverse()
        print_block(n, block)


def test():
    import code_examples
    cfg = make_tacblock_cfg(code_examples.simple_loop, propagate_consts=False, liveness=True)
    for n in sorted(cfg.nodes()):
        block = cfg.nodes[n][BLOCKNAME]
        # print(cfg.nodes[n]['pre_inv'].constants)
        print_block(n, block)
        # print(cfg.nodes[n]['post_inv'].constants)


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
    if ins.is_del or ins.is_assign and set(ins.gens).issubset(kills):
        return
    yield ins._replace(gens=undef(kills, ins.gens),
                       kills=kills - new_kills)


def single_block_liveness(block, kills=frozenset()):
    """kills: the set of names that will no longer be used"""
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


T = typing.TypeVar('T')


# mix of domain and analysis-specific choice of operations
# nothing here really works...
class Domain(Protocol):

    def transfer(self):
        ...

    @classmethod
    def top(cls: T) -> T:
        ...

    @classmethod
    def bottom(cls: T) -> T:
        ...

    def join(self: T, other) -> T:
        ...

    def meet(self: T, other) -> T:
        ...

    def is_bottom(self) -> bool:
        ...

    def is_top(self) -> bool:
        ...

    def is_full(self): pass


class ConstantPropagation(Domain):
    def __init__(self, constants: typing.Optional[dict[tac.Var, tac.Const]] = ()) -> None:
        super().__init__()
        self.constants = constants or {}

    def __le__(self, other):
        return self.join(other).constants == other.constants

    def __eq__(self, other):
        return self.constants == other.constants

    def __ne__(self, other):
        return self.constants != other.constants

    def copy(self: T) -> T:
        return ConstantPropagation(self.constants)

    @classmethod
    def top(cls: T) -> T:
        return ConstantPropagation({})

    def set_to_top(self) -> None:
        self.constants = {}

    @classmethod
    def bottom(cls: T) -> T:
        return ConstantPropagation(None)

    def join(self: T, other: T) -> T:
        if self.constants is None:
            return other.copy()
        if other.constants is None:
            return self.copy()
        return ConstantPropagation(dict(self.constants.items() & other.constants.items()))

    def single_block_update(self, block: list[tac.Tac]) -> None:
        if self.is_bottom:
            return
        for i, ins in enumerate(block):
            if ins.is_assign and len(ins.gens) == len(ins.uses) == 1 and is_stackvar(ins.gens[0]):
                [lhs], [rhs] = ins.gens, ins.uses
                if rhs in self.constants:
                    self.constants[lhs] = self.constants[rhs]
                elif isinstance(rhs, tac.Const):
                    self.constants[lhs] = rhs
            else:
                uses = [(self.constants.get(v, v)) for v in ins.uses]
                if ins.is_inplace:
                    uses[1] = ins.uses[1]
                uses = tuple(uses)
                block[i] = ins._replace(uses=tuple(uses))
                # for v in chain(ins.gens, ins.kills):
                #     if v in self.constants:
                #         del self.constants[v]

    @property
    def is_bottom(self):
        return self.constants is None


class Liveness(Domain):

    @classmethod
    def top(cls: T) -> T:
        return set()

    @staticmethod
    def single_block(block, live=frozenset()):
        """kills: the set of names that will no longer be used"""
        for ins in reversed(block):
            live = frozenset(ins.gens).union(live.difference(ins.uses))

    @staticmethod
    def single_block_total_effect(block, live=frozenset()):
        """kills: the set of names that will no longer be used"""
        gens = {ins.gens for ins in block}
        uses = {ins.uses for ins in block}
        return uses.union(live.difference(gens))


def is_extended_identifier(name):
    return name.replace('.', '').isidentifier()


def run_analysis(cfg):
    dataflow(cfg, ConstantPropagation)


def dataflow(g: nx.DiGraph, Analysis: typing.Type[Domain]):
    start = 0
    nx.set_node_attributes(g, {v: Analysis.bottom() for v in g.nodes()}, 'post_inv')
    nx.set_node_attributes(g, {v: Analysis.bottom() for v in g.nodes()}, 'pre_inv')
    g.nodes[start]['pre_inv'] = Analysis.top()
    wl = {start}
    while wl:
        u = wl.pop()
        udata = g.nodes[u]
        inv = udata['pre_inv'].copy()
        for x in g.predecessors(u):
            inv = inv.join(g.nodes[x]['post_inv'])
        inv.single_block_update(udata[BLOCKNAME])
        if inv != udata['post_inv']:
            udata['post_inv'] = inv
            wl.update(g.successors(u))


if __name__ == '__main__':
    test()
