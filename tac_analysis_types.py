# Data flow analysis and stuff.

from __future__ import annotations as _

from collections import defaultdict
from typing import Optional, Iterable

import tac
from tac import Predefined, UnOp
from tac_analysis_domain import Lattice
import type_system as ts


class TypeLattice(Lattice[ts.TypeExpr]):
    """
    Abstract domain for type analysis with lattice operations.
    """
    def __init__(self, functions, imports: dict[str, str]):
        # global_state = {
        #     **{name: resolve_module(path) for name, path in imports.items()},
        #     **BUILTINS_MODULE.fields,
        # }
        #
        # global_state.update({f: make_function_type(
        #                             return_type=parse_type_annotation(
        #                                 ast.parse(functions[f].__annotations__.get('return', 'object'), mode='eval').body,
        #                                 global_state,
        #                             ),
        #                             new=False,
        #                         )
        #                      for f in functions})
        #
        # self.globals = ts.Class('globals()', frozendict(global_state))
        self.globals = ts.intersect([])

    def name(self) -> str:
        return "Type"

    def join(self, left: ts.TypeExpr, right: ts.TypeExpr) -> ts.TypeExpr:
        return ts.join(left, right)

    def meet(self, left: ts.TypeExpr, right: ts.TypeExpr) -> ts.TypeExpr:
        return ts.meet(left, right)

    def top(self) -> ts.TypeExpr:
        return ts.TOP

    def is_top(self, elem: ts.TypeExpr) -> bool:
        return elem == ts.TOP

    def is_bottom(self, elem: ts.TypeExpr) -> bool:
        return elem == ts.BOTTOM

    @classmethod
    def bottom(cls) -> ts.TypeExpr:
        return ts.BOTTOM

    def resolve(self, ref: ts.TypeExpr) -> ts.TypeExpr:
        if isinstance(ref, ts.Ref):
            if ref.static:
                result = ts.resolve_static_ref(ref)
            else:
                result = self.globals
                for attr in ref.name.split('.'):
                    result = result.fields[attr]
            ref = result
        return ref

    def is_match(self, args: list[ts.TypeExpr], params):
        return True

    def join_all(self, types: Iterable[ts.TypeExpr]) -> ts.TypeExpr:
        result = TypeLattice.bottom()
        for t in types:
            result = self.join(result, t)
        return result

    def call(self, function: ts.TypeExpr, args: list[ts.TypeExpr]) -> ts.TypeExpr:
        return ts.call(self.resolve(function), ts.intersect([ts.Row(index, None, self.resolve(arg))
                                                             for index, arg in enumerate(args)]))

    def binary(self, left: ts.TypeExpr, right: ts.TypeExpr, op: str) -> ts.TypeExpr:
        result = ts.binop(left, right, op)
        # Shorthand for: "we assume that there is an implementation"
        if self.is_bottom(result):
            return self.top()
        return result

    def get_unary_attribute(self, value: ts.TypeExpr, op: UnOp) -> ts.TypeExpr:
        match op:
            case UnOp.NOT: dunder = '__bool__'
            case UnOp.POS: dunder = '__pos__'
            case UnOp.INVERT: dunder = '__invert__'
            case UnOp.NEG: dunder = '__neg__'
            case UnOp.ITER | UnOp.YIELD_ITER: dunder = '__iter__'
            case UnOp.NEXT: dunder = '__next__'
            case x:
                raise NotImplementedError(f'Unary operation {x} not implemented')
        return self._attribute(value, tac.Var(dunder))

    def unary(self, value: ts.TypeExpr, op: UnOp) -> ts.TypeExpr:
        if op == UnOp.NOT:
            return self.resolve(ts.Ref('builtins.bool'))
        f = self.get_unary_attribute(value, op)
        return self.call(f, [value])

    def predefined(self, name: Predefined) -> Optional[ts.TypeExpr]:
        match name:
            case Predefined.LIST: return self.top()
            case Predefined.TUPLE: return self.top()
            case Predefined.GLOBALS: return self.globals
            case Predefined.NONLOCALS: return self.top()
            case Predefined.LOCALS: return self.top()
            case Predefined.SLICE: return self.top()
            case Predefined.CONST_KEY_MAP: return self.top()
        return None

    def const(self, value: object) -> ts.TypeExpr:
        if value is None:
            return self.resolve(ts.Ref('builtins.NoneType'))
        return self.resolve(ts.Ref(f'builtins.{type(value).__name__}', static=True))

    def _attribute(self, var: ts.TypeExpr, attr: tac.Var) -> ts.TypeExpr:
        return ts.subscr(self.resolve(var), ts.Literal(attr.name))

    def attribute(self, var: ts.TypeExpr, attr: tac.Var) -> ts.TypeExpr:
        assert isinstance(attr, tac.Var)
        return self._attribute(var, attr)

    def subscr(self, array: ts.TypeExpr, index: ts.TypeExpr) -> ts.TypeExpr:
        return ts.subscr(self.resolve(array), self.resolve(index))

    def annotation(self, code: str) -> ts.TypeExpr:
        return self.resolve(ts.Ref(code, static=False))
        return ObjectType(type(code).__name__, {})

    def imported(self, modname: tac.Var) -> ts.TypeExpr:
        if isinstance(modname, tac.Var):
            return ts.resolve_static_ref(ts.Ref(modname.name, static=True))
        elif isinstance(modname, tac.Attribute):
            return self.attribute(modname.var, modname.field)

    def is_subtype(self, left: ts.TypeExpr, right: ts.TypeExpr) -> bool:
        left = self.resolve(left)
        right = self.resolve(right)
        return self.join(left, right) == right

    def is_supertype(self, left: ts.TypeExpr, right: ts.TypeExpr) -> bool:
        return self.join(left, right) == left

    def allocation_type_function(self, overloaded_function: ts.TypeExpr) -> tac.AllocationType:
        return tac.AllocationType.NONE
        if isinstance(overloaded_function, OverloadedFunctionType):
            if all(function.new for function in overloaded_function.types):
                return tac.AllocationType.STACK
            if not any(function.new for function in overloaded_function.types):
                return tac.AllocationType.NONE
        return tac.AllocationType.UNKNOWN

    def allocation_type_binary(self, left: ts.TypeExpr, right: ts.TypeExpr, op: str) -> tac.AllocationType:
        return tac.AllocationType.NONE
        overloaded_function = BINARY.get(op)
        narrowed = self.narrow_overload(overloaded_function, [left, right])
        if self.allocation_type_function(narrowed):
            return tac.AllocationType.STACK
        return tac.AllocationType.NONE

    def allocation_type_unary(self, value: ts.TypeExpr, op: tac.UnOp) -> tac.AllocationType:
        return tac.AllocationType.NONE
        if op is UnOp.NOT:
            return tac.AllocationType.NONE
        f = self.get_unary_attribute(value, op)
        if f.types[0].new:
            return tac.AllocationType.STACK
        return tac.AllocationType.NONE

    def allocation_type_attribute(self, var: ts.TypeExpr, attr: tac.Var) -> tac.AllocationType:
        return tac.AllocationType.NONE
        p = self._attribute(var, attr)
        if isinstance(p, Property) and p.new:
            return tac.AllocationType.STACK
        return tac.AllocationType.NONE


unseen = defaultdict(set)
