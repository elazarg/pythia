# Data flow analysis and stuff.

from __future__ import annotations as _

import ast
from collections import defaultdict
from typing import Optional, Iterable, TypeAlias

from frozendict import frozendict

import tac
from tac import Predefined, UnOp
from tac_analysis_domain import Lattice, BOTTOM, Bottom
from type_system import Ref, ObjectType, ParamsType, OverloadedFunctionType, make_function_type, Property, \
    OBJECT, BOOL, NONE, TUPLE_CONSTRUCTOR, LIST_CONSTRUCTOR, SLICE_CONSTRUCTOR, DICT_CONSTRUCTOR, BUILTINS_MODULE, \
    resolve_module, BINARY, make_tuple, SimpleType


TypeElement: TypeAlias = ObjectType | OverloadedFunctionType | Property | Ref | Bottom


def parse_type_annotation(expr: ast.expr, global_state: dict[str, SimpleType]) -> SimpleType:
    match expr:
        case ast.Name(id=name):
            res = global_state[name]
            assert res.name == 'type'
            return res.type_params['T']
        case ast.Attribute(value=ast.Name(id=name), attr=attr):
            obj = global_state[name]
            res = obj.fields[attr]
            assert res.name == 'type'
            return res.type_params['T']
        case ast.Subscript(value=ast.Name(id=generic), slice=index):
            if generic == 'tuple':
                return make_tuple(OBJECT)
            generic = global_state[generic]
            index = global_state[index]
            return generic[index]
        case ast.Constant(value=None):
            return NONE
    assert False, f'Unsupported type annotation: {expr}'


class TypeLattice(Lattice[TypeElement]):
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

    def join(self, left: TypeElement, right: TypeElement) -> TypeElement:
        if self.is_bottom(left) or self.is_top(right):
            return right
        if self.is_bottom(right) or self.is_top(left):
            return left
        if left == right:
            return left
        return self.top()

    def meet(self, left: TypeElement, right: TypeElement) -> TypeElement:
        if self.is_top(left) or self.is_bottom(right):
            return left
        if self.is_top(right) or self.is_bottom(left):
            return right
        if left == right:
            return left
        return self.bottom()

    def top(self) -> TypeElement:
        return OBJECT

    def is_top(self, elem: TypeElement) -> bool:
        return elem == OBJECT

    def is_bottom(self, elem: TypeElement) -> bool:
        return elem == BOTTOM

    @classmethod
    def bottom(cls) -> TypeElement:
        return BOTTOM

    def resolve(self, ref: TypeElement) -> TypeElement:
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

    def is_match(self, args: list[TypeElement], params: ParamsType):
        if len(args) < len(params.params):
            return False
        if len(args) > len(params.params) and not params.varargs:
            return False
        return all(self.is_subtype(arg, param)
                   for arg, param in zip(args, params.params))

    def join_all(self, types: Iterable[TypeElement]) -> TypeElement:
        result = TypeLattice.bottom()
        for t in types:
            result = self.join(result, t)
        return result

    def narrow_overload(self, overload: OverloadedFunctionType, args: list[TypeElement]) -> OverloadedFunctionType:
        return OverloadedFunctionType(tuple(func for func in overload.types
                                            if self.is_match(args, func.params)))

    def call(self, function: TypeElement, args: list[TypeElement]) -> TypeElement:
        if self.is_top(function):
            return self.top()
        if self.is_bottom(function) or any(self.is_bottom(arg) for arg in args):
            return self.bottom()
        if isinstance(function, ObjectType):
            if function.fields.get('__call__'):
                return self.call(function.fields['__call__'], args)
        if not isinstance(function, OverloadedFunctionType):
            print(f'{function} is not an overloaded function')
            return self.bottom()
        return self.join_all(self.resolve(func.return_type)
                             for func in self.narrow_overload(function, args).types)

    def binary(self, left: TypeElement, right: TypeElement, op: str) -> TypeElement:
        overloaded_function = BINARY.get(op)
        result = self.call(overloaded_function, [left, right])
        # Shorthand for: "we assume that there is an implementation"
        if self.is_bottom(result):
            return self.top()
        return result

    def get_unary_attribute(self, value: TypeElement, op: UnOp) -> TypeElement:
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

    def unary(self, value: TypeElement, op: UnOp) -> TypeElement:
        if op == UnOp.NOT:
            return BOOL
        f = self.get_unary_attribute(value, op)
        return self.call(f, [value])

    def predefined(self, name: Predefined) -> Optional[TypeElement]:
        match name:
            case Predefined.LIST: return LIST_CONSTRUCTOR
            case Predefined.TUPLE: return TUPLE_CONSTRUCTOR
            case Predefined.GLOBALS: return self.globals
            case Predefined.NONLOCALS: return NONLOCALS_OBJECT
            case Predefined.LOCALS: return LOCALS_OBJECT
            case Predefined.SLICE: return SLICE_CONSTRUCTOR
            case Predefined.CONST_KEY_MAP: return DICT_CONSTRUCTOR
        return None

    def const(self, value: object) -> TypeElement:
        if value is None:
            return NONE
        return self.annotation(type(value).__name__)

    def _attribute(self, var: TypeElement, attr: tac.Var) -> TypeElement:
        if self.is_top(var):
            return self.top()
        if self.is_bottom(var):
            return self.bottom()
        match var:
            case ObjectType() as obj:
                name = attr.name
                if name in obj.fields:
                    return self.resolve(obj.fields[name])
                else:
                    unseen[obj.name].add(name)
                    return self.top()
        return self.top()

    def attribute(self, var: TypeElement, attr: tac.Var) -> TypeElement:
        assert isinstance(attr, tac.Var)
        res = self._attribute(var, attr)
        if isinstance(res, Property):
            return res.return_type
        return res

    def subscr(self, array: TypeElement, index: TypeElement) -> TypeElement:
        if self.is_bottom(array):
            return self.bottom()
        if self.is_top(array):
            return self.top()
        f = array.fields.get('__getitem__')
        if f is None:
            print(f'{array} is not subscriptable')
            return self.bottom()
        return self.call(f, [index])

    def annotation(self, code: str) -> TypeElement:
        return self.resolve(Ref(code, static=False))
        return ObjectType(type(code).__name__, {})

    def imported(self, modname: tac.Var) -> TypeElement:
        if isinstance(modname, tac.Var):
            return resolve_module(modname.name)
        elif isinstance(modname, tac.Attribute):
            return self.attribute(modname.var, modname.field)

    def is_subtype(self, left: TypeElement, right: TypeElement) -> bool:
        left = self.resolve(left)
        right = self.resolve(right)
        return self.join(left, right) == right

    def is_supertype(self, left: TypeElement, right: TypeElement) -> bool:
        return self.join(left, right) == left

    def allocation_type_function(self, overloaded_function: TypeElement) -> tac.AllocationType:
        if isinstance(overloaded_function, OverloadedFunctionType):
            if all(function.new for function in overloaded_function.types):
                return tac.AllocationType.STACK
            if not any(function.new for function in overloaded_function.types):
                return tac.AllocationType.NONE
        return tac.AllocationType.UNKNOWN

    def allocation_type_binary(self, left: TypeElement, right: TypeElement, op: str) -> tac.AllocationType:
        overloaded_function = BINARY.get(op)
        narrowed = self.narrow_overload(overloaded_function, [left, right])
        if self.allocation_type_function(narrowed):
            return tac.AllocationType.STACK
        return tac.AllocationType.NONE

    def allocation_type_unary(self, value: TypeElement, op: tac.UnOp) -> tac.AllocationType:
        if op is UnOp.NOT:
            return tac.AllocationType.NONE
        f = self.get_unary_attribute(value, op)
        if f.types[0].new:
            return tac.AllocationType.STACK
        return tac.AllocationType.NONE

    def allocation_type_attribute(self, var: TypeElement, attr: tac.Var) -> tac.AllocationType:
        p = self._attribute(var, attr)
        if isinstance(p, Property) and p.new:
            return tac.AllocationType.STACK
        return tac.AllocationType.NONE


unseen = defaultdict(set)
