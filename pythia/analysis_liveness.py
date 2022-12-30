"""
Liveness analysis
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

import typing
from dataclasses import dataclass
from typing import TypeVar

from pythia.graph_utils import Location

from pythia import tac
from pythia.analysis_domain import Top, Bottom, TOP, BOTTOM, \
    MapDomain, Lattice, Map, normalize, InstructionLattice

T = TypeVar('T')


Liveness = Top | Bottom


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

    def initial(self, values: dict[tac.Var, str]) -> Liveness:
        return TOP

    def is_top(self, elem: Liveness) -> bool:
        return isinstance(elem, Top)

    def is_bottom(self, elem: Liveness) -> bool:
        return isinstance(elem, Bottom)

    def is_equivalent(self, left: Liveness, right: Liveness) -> bool:
        return left == right

    def is_less_than(self, left: Liveness, right: Liveness) -> bool:
        return self.join(left, right) == right

    def copy(self, values: Liveness) -> Liveness:
        return values

    def default(self) -> Liveness:
        return BOTTOM

    def bottom(self) -> Liveness:
        return BOTTOM

    def name(self) -> str:
        return "Liveness"


class LivenessVarLattice(InstructionLattice[MapDomain[Liveness]]):
    lattice: LivenessLattice
    backward: bool = True

    def __init__(self) -> None:
        super().__init__()
        self.lattice = LivenessLattice()

    def name(self) -> str:
        return f"{self.lattice.name()}"

    def is_less_than(self, left: MapDomain[Liveness], right: MapDomain[Liveness]) -> bool:
        return self.join(left, right) == right

    def is_equivalent(self, left: MapDomain[Liveness], right: MapDomain[Liveness]) -> bool:
        return self.is_less_than(left, right) and self.is_less_than(right, left)

    def copy(self, values: MapDomain[Liveness]) -> MapDomain[Liveness]:
        return values.copy()

    def is_top(self, elem: T) -> bool:
        return isinstance(elem, Top)

    def is_bottom(self, values: MapDomain[Liveness]) -> bool:
        return isinstance(values, Bottom)

    def make_map(self, d: typing.Optional[dict[tac.Var, Liveness]] = None) -> Map[Liveness]:
        d = d or {}
        return Map(default=self.lattice.default(), d=d)

    def initial(self, annotations: dict) -> MapDomain[Liveness]:
        return self.top()

    def top(self) -> Map[Liveness]:
        return self.make_map()

    def bottom(self) -> MapDomain[Liveness]:
        return BOTTOM

    def join(self, left: MapDomain[Liveness], right: MapDomain[Liveness]) -> MapDomain[Liveness]:
        match left, right:
            case (Bottom(), _): return right
            case (_, Bottom()): return left
            case (Map() as left, Map() as right):
                res = self.top()
                for k in left.keys() | right.keys():
                    res[k] = self.lattice.join(left[k], right[k])
                return normalize(res)
        return self.top()

    def back_transformer_signature(self, signature: tac.Signature) -> tuple[set[tac.Var], set[tac.Var]]:
        return tac.gens_signature(signature), tac.free_vars_lval(signature)

    def back_transfer(self, values: MapDomain[Liveness], ins: tac.Tac, location: Location) -> MapDomain[Liveness]:
        if isinstance(values, Bottom):
            return BOTTOM
        values = values.copy()
        for v in tac.gens(ins):
            values[v] = BOTTOM
        for v in tac.free_vars(ins):
            values[v] = TOP
        return values

    def transfer(self, values: MapDomain[Liveness], ins: tac.Tac, location: Location) -> MapDomain[Liveness]:
        if isinstance(values, Bottom):
            return BOTTOM
        values = values.copy()
        to_update = self.back_transfer(values, ins, location)
        if isinstance(to_update, Bottom):
            return BOTTOM
        for var in tac.gens(ins):
            if var in values:
                del values[var]
        values.update(to_update)
        return normalize(values)
