from __future__ import annotations as _

import enum
import typing

from pythia import type_system as ts, tac, analysis_domain as domain
from pythia.analysis_domain import InvariantMap, VarMapDomain, VarLattice
from pythia.analysis_types import TypeLattice
from pythia.graph_utils import Location


class AllocationType(enum.StrEnum):
    NONE = ""
    STACK = "Stack"
    HEAP = "Heap"
    UNKNOWN = "Unknown"


Allocation: typing.TypeAlias = AllocationType


class AllocationChecker:
    type_invariant_map: InvariantMap[VarMapDomain[ts.TypeExpr]]
    type_lattice: VarLattice[ts.TypeExpr]
    backward = False

    def __init__(
        self,
        type_invariant_map: InvariantMap[VarMapDomain[ts.TypeExpr]],
        type_lattice: VarLattice[ts.TypeExpr],
    ) -> None:
        super().__init__()
        self.type_invariant_map = type_invariant_map
        self.type_lattice = type_lattice

    def __call__(self, ins: tac.Tac, location: Location) -> AllocationType:
        type_invariant: VarMapDomain[ts.TypeExpr] = self.type_invariant_map[location]
        if isinstance(type_invariant, domain.Bottom):
            return Allocation.UNKNOWN
        if isinstance(ins, tac.Assign):
            match ins.expr:
                case tac.Attribute(var=tac.Var() as var, field=tac.Var() as field):
                    var_type = self.type_lattice.transformer_expr(type_invariant, var)
                    function = self.type_lattice.lattice.attribute(var_type, field)
                    if isinstance(function, ts.FunctionType) and function.property:
                        return from_function(function, ts.BOTTOM)
                    return AllocationType.NONE
                case tac.Call(func, args):
                    function = self.type_lattice.transformer_expr(type_invariant, func)
                    returns = ts.call(
                        function,
                        make_rows(
                            *[
                                self.type_lattice.transformer_expr(type_invariant, arg)
                                for index, arg in enumerate(args)
                            ]
                        ),
                    )
                    return from_function(function, returns)
                case tac.Subscript(var=tac.Var() as var, index=tac.Var() as index):
                    var_type = self.type_lattice.transformer_expr(type_invariant, var)
                    index_type = self.type_lattice.transformer_expr(
                        type_invariant, index
                    )
                    return AllocationType.UNKNOWN
                case tac.Unary(var=tac.Var() as var, op=tac.UnOp() as op):
                    value = self.type_lattice.transformer_expr(type_invariant, var)
                    lattice = self.type_lattice.lattice
                    assert isinstance(lattice, TypeLattice)
                    function = lattice.get_unary_attribute(value, op)
                    return from_function(function, make_rows(value))
                case tac.Binary(
                    left=tac.Var() as left, right=tac.Var() as right, op=str() as op
                ):
                    left_type: ts.TypeExpr = self.type_lattice.transformer_expr(
                        type_invariant, left
                    )
                    right_type: ts.TypeExpr = self.type_lattice.transformer_expr(
                        type_invariant, right
                    )
                    function = ts.get_binop(left_type, right_type, op)
                    return from_function(function, make_rows(left_type, right_type))
        return AllocationType.NONE


def make_rows(*types: ts.TypeExpr) -> ts.Intersection:
    return ts.typed_dict([ts.make_row(index, None, t) for index, t in enumerate(types)])


def join(left: Allocation, right: Allocation) -> Allocation:
    if left == Allocation.NONE:
        return right
    if right == Allocation.NONE:
        return left
    if left == right:
        return left
    return Allocation.UNKNOWN


def from_function(function: ts.TypeExpr, returns: ts.TypeExpr) -> Allocation:
    if ts.is_immutable(returns):
        return Allocation.NONE

    def from_function(function: ts.TypeExpr) -> Allocation:
        if isinstance(function, ts.FunctionType):
            if function.new():
                return AllocationType.STACK
            else:
                return AllocationType.NONE
        if isinstance(function, ts.Overloaded):
            result = AllocationType.NONE
            for t in function.items:
                result = join(result, from_function(t))
            return result
        return AllocationType.UNKNOWN

    return from_function(function)
