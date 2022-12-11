# Data flow analysis and stuff.

from __future__ import annotations as _

import ast
from collections import defaultdict
from typing import Optional, Iterable, TypeAlias

from frozendict import frozendict

import tac
from tac import Predefined, UnOp
from tac_analysis_domain import Lattice
from type_system import Ref, TypeExpr, BOTTOM, TOP
import type_system as ts


class TypeLattice(Lattice[TypeExpr]):
    """
    Abstract domain for type analysis with lattice operations.
    """
    def __init__(self, functions, imports: dict[str, str]):
        global_state = {
            **{name: resolve_module(path) for name, path in imports.items()},
            **BUILTINS_MODULE.fields,
        }

        global_state.update({f: make_function_type(
                                    return_type=parse_type_annotation(
                                        ast.parse(functions[f].__annotations__.get('return', 'object'), mode='eval').body,
                                        global_state,
                                    ),
                                    new=False,
                                )
                             for f in functions})

        self.globals = ObjectType('globals()', frozendict(global_state))

    def name(self) -> str:
        return "Type"

    def join(self, left: TypeExpr, right: TypeExpr) -> TypeExpr:
        if self.is_bottom(left) or self.is_top(right):
            return right
        if self.is_bottom(right) or self.is_top(left):
            return left
        if left == right:
            return left
        return self.top()

    def meet(self, left: TypeExpr, right: TypeExpr) -> TypeExpr:
        if self.is_top(left) or self.is_bottom(right):
            return left
        if self.is_top(right) or self.is_bottom(left):
            return right
        if left == right:
            return left
        return self.bottom()

    def top(self) -> TypeExpr:
        return TOP

    def is_top(self, elem: TypeExpr) -> bool:
        return elem == TOP

    def is_bottom(self, elem: TypeExpr) -> bool:
        return elem == BOTTOM

    @classmethod
    def bottom(cls) -> TypeExpr:
        return BOTTOM

    def resolve(self, ref: TypeExpr) -> TypeExpr:
        if isinstance(ref, Ref):
            if ref.static:
                result = resolve_module(ref.name)
            else:
                result = self.globals
                for attr in ref.name.split('.'):
                    result = result.fields[attr]
            assert result.name == 'type'
            return result.type_params['T']
        return ref

    def is_match(self, args: list[TypeExpr], params: ParamsType):
        if len(args) < len(params.params):
            return False
        if len(args) > len(params.params) and not params.varargs:
            return False
        return all(self.is_subtype(arg, param)
                   for arg, param in zip(args, params.params))

    def join_all(self, types: Iterable[TypeExpr]) -> TypeExpr:
        result = TypeLattice.bottom()
        for t in types:
            result = self.join(result, t)
        return result

    def call(self, function: TypeExpr, args: list[TypeExpr]) -> TypeExpr:
        return ts.apply(self.resolve(function), 'call', ts.intersect([ts.Row(index, None, self.resolve(arg))
                                                                      for index, arg in enumerate(args)]))

    def binary(self, left: TypeExpr, right: TypeExpr, op: str) -> TypeExpr:
        overloaded_function = BINARY.get(op)
        result = self.call(overloaded_function, [left, right])
        # Shorthand for: "we assume that there is an implementation"
        if self.is_bottom(result):
            return self.top()
        return result

    def get_unary_attribute(self, value: TypeExpr, op: UnOp) -> TypeExpr:
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

    def unary(self, value: TypeExpr, op: UnOp) -> TypeExpr:
        if op == UnOp.NOT:
            return self.resolve(Ref('builtins.bool'))
        f = self.get_unary_attribute(value, op)
        return self.call(f, [value])

    def predefined(self, name: Predefined) -> Optional[TypeExpr]:
        match name:
            case Predefined.LIST: return LIST_CONSTRUCTOR
            case Predefined.TUPLE: return TUPLE_CONSTRUCTOR
            case Predefined.GLOBALS: return self.globals
            case Predefined.NONLOCALS: return NONLOCALS_OBJECT
            case Predefined.LOCALS: return LOCALS_OBJECT
            case Predefined.SLICE: return SLICE_CONSTRUCTOR
            case Predefined.CONST_KEY_MAP: return DICT_CONSTRUCTOR
        return None

    def const(self, value: object) -> TypeExpr:
        if value is None:
            return self.resolve(Ref('builtins.NoneType'))
        return self.resolve(Ref(type(value).__name__))

    def _attribute(self, var: TypeExpr, attr: tac.Var) -> TypeExpr:
        return ts.apply(self.resolve(var), 'getattr', ts.Literal(attr.name))

    def attribute(self, var: TypeExpr, attr: tac.Var) -> TypeExpr:
        assert isinstance(attr, tac.Var)
        return self._attribute(var, attr)

    def subscr(self, array: TypeExpr, index: TypeExpr) -> TypeExpr:
        return ts.apply(self.resolve(array), 'getitem', self.resolve(index))

    def annotation(self, code: str) -> TypeExpr:
        return self.resolve(Ref(code, static=False))
        return ObjectType(type(code).__name__, {})

    def imported(self, modname: tac.Var) -> TypeExpr:
        if isinstance(modname, tac.Var):
            return resolve_module(modname.name)
        elif isinstance(modname, tac.Attribute):
            return self.attribute(modname.var, modname.field)

    def is_subtype(self, left: TypeExpr, right: TypeExpr) -> bool:
        left = self.resolve(left)
        right = self.resolve(right)
        return self.join(left, right) == right

    def is_supertype(self, left: TypeExpr, right: TypeExpr) -> bool:
        return self.join(left, right) == left

    def allocation_type_function(self, overloaded_function: TypeExpr) -> tac.AllocationType:
        if isinstance(overloaded_function, OverloadedFunctionType):
            if all(function.new for function in overloaded_function.types):
                return tac.AllocationType.STACK
            if not any(function.new for function in overloaded_function.types):
                return tac.AllocationType.NONE
        return tac.AllocationType.UNKNOWN

    def allocation_type_binary(self, left: TypeExpr, right: TypeExpr, op: str) -> tac.AllocationType:
        overloaded_function = BINARY.get(op)
        narrowed = self.narrow_overload(overloaded_function, [left, right])
        if self.allocation_type_function(narrowed):
            return tac.AllocationType.STACK
        return tac.AllocationType.NONE

    def allocation_type_unary(self, value: TypeExpr, op: tac.UnOp) -> tac.AllocationType:
        if op is UnOp.NOT:
            return tac.AllocationType.NONE
        f = self.get_unary_attribute(value, op)
        if f.types[0].new:
            return tac.AllocationType.STACK
        return tac.AllocationType.NONE

    def allocation_type_attribute(self, var: TypeExpr, attr: tac.Var) -> tac.AllocationType:
        p = self._attribute(var, attr)
        if isinstance(p, Property) and p.new:
            return tac.AllocationType.STACK
        return tac.AllocationType.NONE


unseen = defaultdict(set)
