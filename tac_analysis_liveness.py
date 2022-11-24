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

import typing

import graph_utils
import tac
from tac import Tac, Var
from tac_analysis_domain import IterationStrategy, BackwardIterationStrategy, Top, Bottom, Lattice, TOP, BOTTOM, \
    VarAnalysis
import graph_utils as gu

T = TypeVar('T')


def update_liveness(ins, inv) -> Tac:
    uses = [(inv.get(v, v)) for v in ins.uses]
    if ins.is_inplace:
        uses[1] = ins.uses[1]
    for v in chain(ins.gens, ins.kills):
        if v in inv:
            del inv[v]


Liveness = bool | Top | Bottom


@dataclass(frozen=True)
class LivenessLattice(Lattice[Liveness]):
    def join(self, left: Liveness, right: Liveness) -> Liveness:
        if self.is_bottom(left) or self.is_top(right):
            return right
        if self.is_bottom(right) or self.is_top(left):
            return left
        if left == right:
            return left
        return self.top()

    def meet(self, left: Liveness, right: Liveness) -> Liveness:
        if self.is_top(left) or self.is_bottom(right):
            return left
        if self.is_top(right) or self.is_bottom(left):
            return right
        if left == right:
            return left
        return self.bottom()

    def top(self) -> Liveness:
        return TOP

    def is_top(self, elem: Liveness) -> bool:
        return isinstance(elem, Top)

    def is_bottom(self, elem: Liveness) -> bool:
        return isinstance(elem, Bottom)

    @classmethod
    def bottom(cls) -> Liveness:
        return BOTTOM

    def back_assign_subscr(self, var: T, index: T) -> T:
        return self.top()

    def back_assign_attribute(self, var: T, attr: str) -> T:
        return self.top()

    def back_assign_tuple(self, values: tuple[T]) -> T:
        return self.top()

    def back_call(self, value: Liveness, size: int) -> tuple[Liveness, list[Liveness]]:
        result = (True, [True] * size)
        return result

    def back_binary(self, value: Liveness) -> tuple[Liveness, Liveness]:
        return (True, True)

    def back_inplace_binary(self, lhs: Liveness, right: Liveness, op: str) -> tuple[Liveness, Liveness]:
        return (True, True)

    def back_predefined(self, value: T) -> None:
        return None

    def back_const(self, value: T) -> None:
        return None

    def back_attribute(self, value: Liveness) -> Liveness:
        return True

    def back_subscr(self, value: Liveness) -> tuple[Liveness, Liveness]:
        return (True, True)

    def back_annotation(self, value: Liveness) -> Liveness:
        return True

    def back_imported(self, value: Liveness) -> None:
        return None

    def back_return(self) -> T:
        return True

    def default(self) -> T:
        return False

    @classmethod
    def view(cls, cfg: gu.Cfg[T]) -> IterationStrategy:
        return BackwardIterationStrategy(cfg)

    @staticmethod
    def name() -> str:
        return "Liveness"


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
    alive: VarAnalysis[Liveness] = block.post[LivenessLattice.name()]
    if alive.is_bottom:
        return
    if len(block) <= 1:
        return
    for i in reversed(range(len(block))):
        ins = block[i]
        if isinstance(ins, tac.Assign) and ins.no_side_effect and ins.lhs not in alive.vars:
                del block[i]
                continue
        alive.transfer(block[i], f'{label}.{i}')


# poor man's use-def
def rewrite_remove_useless_movs_pairs(block: graph_utils.Block, label: int) -> None:
    alive: VarAnalysis[Liveness] = block.post[LivenessLattice.name()]
    if alive.is_bottom:
        return
    for i in reversed(range(1, len(block))):
        ins = block[i]
        if isinstance(ins, tac.Assign) and ins.assign_stack and ins.lhs not in alive.vars:
            ins = block[i] = dataclasses.replace(ins, lhs=None)

        prev = block[i-1]
        merged_instruction = None
        killed_by_ins = tac.free_vars(ins) - (alive.vars - tac.gens(ins))
        if isinstance(prev, tac.Assign) and prev.assign_stack and prev.lhs in killed_by_ins:
            # $0 = Var
            # v = EXP($0)  # $0 is killed
            match ins:
                case tac.Return():
                    merged_instruction = tac.subst_var_in_ins(ins, prev.lhs, prev.expr)
                case tac.InplaceBinary():
                    if ins.right == prev.lhs:
                        merged_instruction = tac.subst_var_in_ins(ins, prev.lhs, prev.expr)
                case tac.Assign():
                    if isinstance(prev.expr, (tac.Var, tac.Liveness)):
                        merged_instruction = tac.subst_var_in_ins(ins, prev.lhs, prev.expr)
                    elif isinstance(ins.expr, tac.Var):
                        expr = tac.subst_var_in_expr(ins.expr, prev.lhs, prev.expr)
                        merged_instruction = dataclasses.replace(ins, expr=expr)
        if merged_instruction is not None:
            # print(f'{label}.{i}: {prev}; {ins} -> {merged_instruction}')
            block[i] = merged_instruction
            del block[i - 1]
            # if prev.lhs in tac.free_vars(merged_instruction):
            #     print(f'{prev}; {ins}: {prev.lhs} in {merged_instruction}')
        else:
            # print(f'{label}.{i}: {prev}; {ins} -> {merged_instruction}')
            alive.transfer(ins, f'{label}.{i}')
