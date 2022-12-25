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
    def __init__(self, this_function: str, this_module: ts.Module, functions, imports: dict[str, str]):
        this_signature = ts.subscr(this_module, ts.Literal(this_function))
        assert isinstance(this_signature, ts.FunctionType)
        self.annotations = {tac.Var(row.index.name.value, is_stackvar=False): row.type
                            for row in this_signature.params.items}
        self.annotations[tac.Var('return', is_stackvar=False)] = this_signature.return_type
        self.globals = this_module
        self.builtins = ts.resolve_static_ref(ts.Ref('builtins'))

    def annotation(self, name: tac.Var, t: str) -> ts.TypeExpr:
        return self.annotations[name]

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
            if '.' in ref.name:
                module, name = ref.name.split('.', 1)
                if module == self.globals.name:
                    return ts.subscr(self.globals, ts.Literal[str](name))
            ref = ts.resolve_static_ref(ref)
        assert isinstance(ref, (ts.TypeExpr, ts.Module, ts.Class)), ref
        return ref

    def is_match(self, args: list[ts.TypeExpr], params):
        return True

    def join_all(self, types: Iterable[ts.TypeExpr]) -> ts.TypeExpr:
        result = TypeLattice.bottom()
        for t in types:
            result = self.join(result, t)
        return result

    def call(self, function: ts.TypeExpr, args: list[ts.TypeExpr]) -> ts.TypeExpr:
        return ts.call(self.resolve(function), ts.intersect([ts.Row(ts.Index(ts.Literal(index), None), self.resolve(arg))
                                                             for index, arg in enumerate(args)]))

    def binary(self, left: ts.TypeExpr, right: ts.TypeExpr, op: str) -> ts.TypeExpr:
        result = ts.binop(left, right, op)
        # Shorthand for: "we assume that there is an implementation"
        if self.is_bottom(result):
            return self.top()
        return result

    def get_unary_attribute(self, value: ts.TypeExpr, op: UnOp) -> ts.TypeExpr:
        return ts.get_unop(value, self.unop_to_str(op))

    def unop_to_str(self, op: UnOp) -> str:
        match op:
            case UnOp.NEG: return '-'
            case UnOp.NOT: return 'not'
            case UnOp.INVERT: return '~'
            case UnOp.POS: return '+'
            case UnOp.ITER: return 'iter'
            case UnOp.NEXT: return 'next'
            case UnOp.YIELD_ITER: return 'yield iter'
            case _:
                raise NotImplementedError(f"UnOp.{op.name}")

    def unary(self, value: ts.TypeExpr, op: UnOp) -> ts.TypeExpr:
        f = self.get_unary_attribute(value, op)
        return self.call(f, [])

    def predefined(self, name: Predefined) -> Optional[ts.TypeExpr]:
        match name:
            case Predefined.LIST: return ts.make_constructor(ts.Ref('builtins.list'))
            case Predefined.TUPLE: return ts.make_constructor(ts.Ref('builtins.tuple'))
            case Predefined.GLOBALS: return self.globals
            case Predefined.NONLOCALS: return self.top()
            case Predefined.LOCALS: return self.top()
            case Predefined.SLICE: return ts.make_slice_constructor()
            case Predefined.CONST_KEY_MAP: return self.top()
        return None

    def const(self, value: object) -> ts.TypeExpr:
        return ts.constant(value)

    def _attribute(self, var: ts.TypeExpr, attr: tac.Var) -> ts.TypeExpr:
        mod = self.resolve(var)
        assert mod != ts.TOP, f'Cannot resolve {var}'
        try:
            res = ts.subscr(mod, ts.Literal(attr.name))
            if self.is_bottom(res):
                if mod == self.globals:
                    return ts.subscr(self.builtins, ts.Literal(attr.name))
            return res
        except TypeError:
            if mod == self.globals:
                return ts.subscr(self.builtins, ts.Literal(attr.name))
            raise

    def attribute(self, var: ts.TypeExpr, attr: tac.Var) -> ts.TypeExpr:
        assert isinstance(attr, tac.Var)
        return self._attribute(var, attr)

    def subscr(self, array: ts.TypeExpr, index: ts.TypeExpr) -> ts.TypeExpr:
        return ts.subscr(self.resolve(array), self.resolve(index))

    def imported(self, modname: tac.Var) -> ts.TypeExpr:
        if isinstance(modname, tac.Var):
            return ts.resolve_static_ref(ts.Ref(modname.name))
        elif isinstance(modname, tac.Attribute):
            return self.attribute(modname.var, modname.field)

    def is_subtype(self, left: ts.TypeExpr, right: ts.TypeExpr) -> bool:
        left = self.resolve(left)
        right = self.resolve(right)
        return self.join(left, right) == right

    def is_supertype(self, left: ts.TypeExpr, right: ts.TypeExpr) -> bool:
        return self.join(left, right) == left

    def allocation_type_function(self, function: ts.TypeExpr) -> tac.AllocationType:
        if isinstance(function, ts.FunctionType):
            if function.new != ts.Literal(False):
                return tac.AllocationType.STACK
            else:
                return tac.AllocationType.NONE
        return tac.AllocationType.UNKNOWN

    def allocation_type_binary(self, left: ts.TypeExpr, right: ts.TypeExpr, op: str) -> tac.AllocationType:
        f = ts.get_binop(left, right, op)
        return self.allocation_type_function(f)

    def allocation_type_unary(self, value: ts.TypeExpr, op: tac.UnOp) -> tac.AllocationType:
        f = self.get_unary_attribute(value, op)
        return self.allocation_type_function(f)

    def allocation_type_attribute(self, var: ts.TypeExpr, attr: tac.Var) -> tac.AllocationType:
        return tac.AllocationType.NONE
        p = self._attribute(var, attr)
        if isinstance(p, Property) and p.new:
            return tac.AllocationType.STACK
        return tac.AllocationType.NONE


unseen = defaultdict(set)
