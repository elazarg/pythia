"""
Liveness analsysis
These are important in order to make the TAC at least *look* different from
stack-oriented code, and it removes many variables.
The analysis is based heavily on the information in the tac module - some of
it is dedicated for the analysis

A note about naming:
Different sources give different names to gen/def/use/live/kill variables.
Here:
   1. USES is the list of variables used by an instruction/block.
      e.g. `x = f(a, b)` : USES=(f, a, b)
   2. GENS is the list of variables defined by an instruction/block
      e.g. `x = f(a, b)` : GENS=(x,)
   3. KILLS is the list of variables killed by an instruction/block, which do not appear in GENS.
      For most of the instructions, initially, KILLS is empty.
      However, `DEL x` will have KILLS={x} but empty GENS
      In addition, the per-block live variable analysis removes DEL operations,
      and push them into each other and into other instructions.
      So in some cases, `x = f(a, b)` might have e.g. KILLS={a}
      (and implicitly x too) if it is the last command to use the variable `a`.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from itertools import chain
from typing import Type, TypeVar, ClassVar

import graph_utils
import tac
from tac import Tac, Var
from tac_analysis_domain import AbstractDomain, IterationStrategy, BackwardIterationStrategy
import graph_utils as gu

T = TypeVar('T')


def update_liveness(ins, inv) -> Tac:
    uses = [(inv.get(v, v)) for v in ins.uses]
    if ins.is_inplace:
        uses[1] = ins.uses[1]
    for v in chain(ins.gens, ins.kills):
        if v in inv:
            del inv[v]


@dataclass(frozen=True)
class LivenessLattice:
    is_alive: bool = False

    @classmethod
    def bottom(cls: Type[T]) -> T:
        return LivenessLattice(False)

    @classmethod
    def top(cls: Type[T]) -> T:
        return LivenessLattice(True)

    def is_bottom(self) -> bool:
        return not self.is_alive

    def is_top(self) -> bool:
        return self.is_alive

    def __le__(self, other: LivenessLattice) -> bool:
        return other.is_alive or not self.is_alive

    def meet(self, other: LivenessLattice) -> LivenessLattice:
        return LivenessLattice(min(self.is_alive, other.is_alive))

    def join(self, other: LivenessLattice) -> LivenessLattice:
        return LivenessLattice(max(self.is_alive, other.is_alive))


class LivenessDomain(AbstractDomain):
    vars: set[Var] | None = None

    BOTTOM: ClassVar[None] = None

    @staticmethod
    def name() -> str:
        return "Liveness"

    def __init__(self, vars: set[Var] | None) -> None:
        super().__init__()
        if vars is not None:
            self.vars = vars.copy()

    def __le__(self, other):
        return self.join(other).vars == other.vars

    def __eq__(self, other):
        return self.vars == other.vars

    def __ne__(self, other):
        return self.vars != other.vars

    def copy(self: T) -> T:
        return LivenessDomain(self.vars)

    @classmethod
    def initial(cls: Type[T]) -> T:
        return cls.top()

    @classmethod
    def top(cls: Type[T]) -> T:
        return LivenessDomain(set())

    @classmethod
    def bottom(cls: Type[T]) -> T:
        return LivenessDomain(None)

    @property
    def is_bottom(self) -> bool:
        return self.vars is None

    def join(self: T, other: T) -> T:
        if self.is_bottom:
            return other.copy()
        if other.is_bottom:
            return self.copy()
        return LivenessDomain(self.vars | other.vars)

    def transfer(self, ins: tac.Tac, location: str) -> None:
        if self.vars is None:
            return
        self.vars -= tac.gens(ins)
        self.vars |= tac.free_vars(ins)

    def __str__(self) -> str:
        return 'Alive({})'.format(", ".join(f'{k}' for k in self.vars))

    def __repr__(self) -> str:
        return 'Alive({})'.format(", ".join(f'{k}' for k in self.vars))

    @classmethod
    def view(cls, cfg: gu.Cfg[T]) -> IterationStrategy:
        return BackwardIterationStrategy(cfg)


def single_block_uses(block):
    uses = set()
    for ins in reversed(block):
        uses.difference_update(ins.gens)
        uses.update(ins.uses)
    return tuple(x for x in uses if is_extended_identifier(x))


def undef(kills, gens):
    return tuple(('_' if v in kills and tac.is_stackvar(v) else v)
                 for v in gens)


def _filter_killed(ins: Tac, kills, new_kills):
    # moved here only because it is a transformation and not an analysis
    if isinstance(ins, tac.Del) or isinstance(ins, tac.Assign) and set(tac.gens(ins)).issubset(kills):
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


def is_extended_identifier(name):
    return name.replace('.', '').isidentifier()


def rewrite_remove_useless_movs(block: graph_utils.Block, label: int) -> None:
    invariant: LivenessDomain = block.post[LivenessDomain.name()]
    if invariant.is_bottom:
        return
    for i in reversed(range(len(block))):
        ins = block[i]
        if isinstance(ins, tac.Mov) and ins.lhs.is_stackvar and ins.lhs not in invariant.vars:
            del block[i]
            continue
        if isinstance(ins, tac.Assign) and isinstance(ins.lhs, tac.Var)\
                and ins.lhs.is_stackvar and ins.lhs.is_stackvar not in invariant.vars\
                and isinstance(ins.expr, tac.Attribute):
            del block[i]
            continue
        invariant.transfer(block[i], f'{label}.{i}')


# poor man's use-def
def rewrite_remove_useless_movs_pairs(block: graph_utils.Block, label: int) -> None:
    invariant: LivenessDomain = block.pre[LivenessDomain.name()]
    if invariant.is_bottom:
        return
    for i in reversed(range(1, len(block))):
        ins = block[i]
        prev = block[i-1]
        if (isinstance(prev, tac.Assign) and isinstance(prev.lhs, tac.Var) or isinstance(prev, tac.Import)) and prev.lhs.is_stackvar:
            if isinstance(ins, tac.Mov) and ins.rhs == prev.lhs and ins.rhs not in invariant.vars:
                del block[i]
                block[i-1] = dataclasses.replace(block[i-1], lhs=ins.lhs)
