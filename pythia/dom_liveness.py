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

from __future__ import annotations as _

from pythia import tac
from pythia.domains import (
    Top,
    Bottom,
    TOP,
    BOTTOM,
    InstructionLattice,
)
from pythia.dom_concrete import Set, SetDomain
from pythia.graph_utils import Location

type Liveness = SetDomain[tac.Var]


class LivenessVarLattice(InstructionLattice[Liveness]):
    lattice: Liveness
    backward: bool = True

    def __init__(self) -> None:
        super().__init__()
        self.lattice = Set[tac.Var]()

    @classmethod
    def name(cls) -> str:
        return "Liveness"

    def initial(self) -> Liveness:
        return Set[tac.Var]()

    def is_less_than(self, left: Liveness, right: Liveness) -> bool:
        return self.join(left, right) == right

    @classmethod
    def is_top(cls, elem: Liveness) -> bool:
        return isinstance(elem, Top)

    @classmethod
    def is_bottom(cls, elem: Liveness) -> bool:
        return isinstance(elem, Bottom)

    def top(self) -> Liveness:
        return TOP

    def bottom(self) -> Liveness:
        return BOTTOM

    def join(self, left: Liveness, right: Liveness) -> Liveness:
        match left, right:
            case (Bottom(), val) | (val, Bottom()):
                return val
            case (Top(), _) | (_, Top()):
                return TOP
            case (Set() as left, Set() as right):
                return left | right
            case _, _:
                raise ValueError(f"Invalid join: {left!r} and {right!r}")

    def transfer(self, values: Liveness, ins: tac.Tac, location: Location) -> Liveness:
        if isinstance(values, Bottom):
            return BOTTOM
        res = (values - Set[tac.Var](tac.gens(ins))) | Set[tac.Var](tac.free_vars(ins))
        return res
