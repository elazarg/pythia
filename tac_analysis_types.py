# Data flow analysis and stuff.

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Type, TypeVar, Optional, ClassVar, Final

import numpy as np

import tac
from tac import Const, Var
from tac_analysis_domain import AbstractDomain, IterationStrategy, ForwardIterationStrategy, Bottom, Top, Lattice

import graph_utils as gu

T = TypeVar('T')


@dataclass()
class ObjectType:
    name: str
    fields: dict[str, FunctionType | ObjectType]

    @staticmethod
    def translate(t: type):
        if t is int: return INT
        if t is float: return FLOAT
        if t is str: return STRING
        if t is bool: return BOOL
        if t is np.ndarray: return NDARRAY
        return ObjectType(type(t).__name__, {})

    @staticmethod
    def typeof(const: tac.Const):
        return ObjectType.translate(type(const.value)) if const.value is not None else NONE

    def __repr__(self):
        return self.name


@dataclass
class FunctionType:
    return_type: ObjectType
    new: bool = True

    def __repr__(self):
        return f'() -> {self.return_type}'


FLOAT = ObjectType('float', {})
INT = ObjectType('int', {})
STRING = ObjectType('str', {})
BOOL = ObjectType('bool', {})
NONE = ObjectType('None', {})


def make_tuple(t: ObjectType) -> ObjectType:
    return ObjectType('tuple', {
        '__getitem__': FunctionType(t),
    })


def iter_method(element_type: ObjectType) -> FunctionType:
    return FunctionType(ObjectType(f'iterator[{element_type}]', {
        '__next__': FunctionType(element_type),
    }))


NDARRAY = ObjectType('ndarray', {
    'mean': FunctionType(FLOAT),
    'std': FunctionType(FLOAT),
    'shape': make_tuple(INT),
    'size': INT,
    '__getitem__': FunctionType(FLOAT),
    '__iter__': iter_method(FLOAT),  # inaccurate
})

NDARRAY.fields['T'] = NDARRAY

ARRAY_GEN = FunctionType(NDARRAY)

NDARRAY.fields['astype'] = ARRAY_GEN


DATAFRAME = ObjectType('DataFrame', {

})

TIME_MODULE = ObjectType('/time', {

})

NUMPY_MODULE = ObjectType('/numpy', {
    'array': ARRAY_GEN,
    'dot': FunctionType(FLOAT),
    'zeros': ARRAY_GEN,
    'ones': ARRAY_GEN,
    'concatenate': ARRAY_GEN,
    'empty': ARRAY_GEN,
    'empty_like': ARRAY_GEN,
    'full': ARRAY_GEN,
    'full_like': ARRAY_GEN,
    'arange': ARRAY_GEN,
    'linspace': ARRAY_GEN,
    'logspace': ARRAY_GEN,
    'geomspace': ARRAY_GEN,
    'meshgrid': ARRAY_GEN,
    'max': FunctionType(FLOAT),
    'min': FunctionType(FLOAT),
    'sum': FunctionType(FLOAT),
    'setdiff1d': ARRAY_GEN,
    'unique': ARRAY_GEN,
    'append': ARRAY_GEN,
    'random': ARRAY_GEN,
    'argmax': FunctionType(INT),
    'c_': ObjectType('slice_trick', {
        '__getitem__': FunctionType(NDARRAY),
    }),
    'r_': ObjectType('slice_trick', {
        '__getitem__': FunctionType(NDARRAY),
    }),

})

PANDAS_MODULE = ObjectType('/pandas', {
    'DataFrame': FunctionType(DATAFRAME),
})

BUILTINS_MODULE = ObjectType('/builtins', {
    'range': FunctionType(ObjectType('range', {
        '__iter__': iter_method(INT),
    })),
    'len': FunctionType(INT),
    'print': FunctionType(NONE),
    'abs': FunctionType(FLOAT),
    'round': FunctionType(FLOAT),
    'min': FunctionType(FLOAT, new=False),
    'max': FunctionType(FLOAT, new=False),
    'sum': FunctionType(FLOAT),
    'all': FunctionType(BOOL),
    'any': FunctionType(BOOL),
    'int': FunctionType(INT),
    'float': FunctionType(FLOAT),
    'str': FunctionType(STRING),
    'bool': FunctionType(BOOL),
})

LOCALS_MODULE = ObjectType('LOCALS', { })
GLOBALS_MODULE = ObjectType('LOCALS', { })
NONLOCALS_MODULE = ObjectType('NONLOCALS', { })

modules = {
    'builtins': BUILTINS_MODULE,
    'numpy': NUMPY_MODULE,
    'pandas': PANDAS_MODULE,
    'time': TIME_MODULE,
}

BINARY = {
    (INT.name, INT.name): {
        '/': FLOAT,
        **{op: INT for op in ['+', '-', '*', '**', '//', '%']},
        **{op: INT for op in ['&', '|', '^', '<<', '>>']},
        **{op: BOOL for op in ['<', '>', '<=', '>=', '==', '!=']},
    },
    (INT.name, FLOAT.name): {
        **{op: FLOAT for op in ['+', '-', '*', '/', '**', '//', '%']},
        **{op: BOOL for op in ['<', '>', '<=', '>=', '==', '!=']},
    },
    (FLOAT.name, INT.name): {
        **{op: FLOAT for op in ['+', '-', '*', '/', '**', '//', '%']},
        **{op: BOOL for op in ['<', '>', '<=', '>=', '==', '!=']},
    },
    (FLOAT.name, FLOAT.name): {
        **{op: FLOAT for op in ['+', '-', '*', '/', '**', '//', '%']},
        **{op: BOOL for op in ['<', '>', '<=', '>=', '==', '!=']},
    },
    (NDARRAY.name, NDARRAY.name): {op: NDARRAY for op in ['+', '-', '*', '/', '**', '//', '%', '@', '==', '!=', '<', '<=', '>', '>=']},
    (NDARRAY.name, INT.name): {op: NDARRAY for op in ['+', '-', '*', '/', '**', '//', '%', '@', '==', '!=', '<', '<=', '>', '>=']},
    (INT.name, NDARRAY.name): {op: NDARRAY for op in ['+', '-', '*', '/', '**', '//', '%', '@', '==', '!=', '<', '<=', '>', '>=']},
    (NDARRAY.name, FLOAT.name): {op: NDARRAY for op in ['+', '-', '*', '/', '**', '//', '%', '@', '==', '!=', '<', '<=', '>', '>=']},
    (FLOAT.name, NDARRAY.name): {op: NDARRAY for op in ['+', '-', '*', '/', '**', '//', '%', '@', '==', '!=', '<', '<=', '>', '>=']},
}


@dataclass(frozen=True)
class TypeLattice(Lattice):
    """
    Abstract domain for type analysis with lattice operations.
    For now, it is essentially constant domain
    """

    @staticmethod
    def name() -> str:
        return "Type"

    value: ObjectType | FunctionType | Bottom | Top

    BOTTOM: Final[ClassVar[Bottom]] = Bottom()
    TOP: Final[ClassVar[Top]] = Top()

    def join(self, other: TypeLattice) -> TypeLattice:
        if self.is_bottom or other.is_top:
            return other
        if other.is_bottom or self.is_top:
            return self
        if self.value == other.value:
            return self
        return TypeLattice(TypeLattice.TOP)

    def meet(self, other: TypeLattice) -> TypeLattice:
        if self.is_top or other.is_bottom:
            return other
        if other.is_top or self.is_bottom:
            return self
        if self.value == other.value:
            return self
        return TypeLattice(TypeLattice.BOTTOM)

    @classmethod
    def top(cls) -> TypeLattice:
        return TypeLattice(TypeLattice.TOP)

    def is_top(self) -> bool:
        return self.value == TypeLattice.TOP

    def is_bottom(self) -> bool:
        return self.value == TypeLattice.BOTTOM

    @classmethod
    def bottom(cls) -> TypeLattice:
        return TypeLattice(TypeLattice.BOTTOM)

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)


class TypeDomain(AbstractDomain):
    types: defaultdict[Var, TypeLattice] | Bottom

    BOTTOM: ClassVar[Bottom] = Bottom()

    @staticmethod
    def name() -> str:
        return "Type"

    @classmethod
    def view(cls, cfg: gu.Cfg[T]) -> IterationStrategy[T]:
        return ForwardIterationStrategy(cfg)

    def __init__(self, types: defaultdict[Var, TypeLattice] | Bottom) -> None:
        super().__init__()
        if types is TypeDomain.BOTTOM:
            self.types = TypeDomain.BOTTOM
        else:
            self.types = types.copy()

    def __le__(self, other):
        return self.join(other).types == other.types

    def __eq__(self, other):
        return self.types == other.types

    def copy(self: T) -> T:
        return TypeDomain(self.types)

    @classmethod
    def initial(cls: Type[T]) -> T:
        return cls.top()

    @classmethod
    def top(cls: Type[T]) -> T:
        return TypeDomain(defaultdict(TypeLattice.top))

    @classmethod
    def bottom(cls: Type[T]) -> T:
        return TypeDomain(TypeDomain.BOTTOM)

    @property
    def is_bottom(self) -> bool:
        return self.types is TypeDomain.BOTTOM

    def join(self: T, other: T) -> T:
        if self.is_bottom:
            return other.copy()
        if other.is_bottom:
            return self.copy()
        res = TypeDomain.top()
        for k in self.types.keys() | other.types.keys():
            if k in self.types.keys() and k in other.types.keys():
                res.types[k] = self.types[k].join(other.types[k])
            else:
                res.types[k] = TypeLattice.top()
        res.normalize()
        return res

    def transfer(self, ins: tac.Tac, location: str) -> None:
        if self.is_bottom:
            return
        types = self.types.copy()
        for var in tac.gens(ins):
            if var in self.types:
                del self.types[var]
        if isinstance(ins, tac.Assign):
            rhs = eval(types, ins.expr)
            match ins.lhs:
                case tac.Attribute(var=tac.Scope.LOCALS): LOCALS_MODULE.fields[ins.lhs.attr] = rhs.value
                case tac.Attribute(var=tac.Scope.GLOBALS): GLOBALS_MODULE.fields[ins.lhs.attr] = rhs.value
                case tac.Attribute(var=tac.Scope.NONLOCALS): NONLOCALS_MODULE.fields[ins.lhs.attr] = rhs.value
                case tac.Var():
                    self.types[ins.lhs] = eval(types, ins.expr)
        elif isinstance(ins, tac.For):
            self.transfer(ins.as_call(), location)
        self.normalize()

    def normalize(self) -> None:
        for k, v in list(self.types.items()):
            if v == TypeLattice.BOTTOM:
                self.types = TypeDomain.BOTTOM
                return
            if v == TypeLattice.top():
                del self.types[k]

    def __str__(self) -> str:
        if self.is_bottom:
            return f'Types({TypeDomain.BOTTOM})'
        return 'Types({})'.format(", ".join(f'{k}: {v.value}' for k, v in self.types.items()))

    def __repr__(self) -> str:
        return self.types.__repr__()

    def keep_only_live_vars(self, alive_vars: set[tac.Var]) -> None:
        for var in set(self.types.keys()) - alive_vars:
            del self.types[var]

    @classmethod
    def read_initial(cls, annotations: dict[str, type]) -> TypeDomain:
        result = TypeDomain.top()
        result.types.update({
            tac.make_local(name): TypeLattice(ObjectType.translate(t))
            for name, t in annotations.items()
        })
        return result


unseen = defaultdict(set)


def eval(types: dict[Var, TypeLattice], expr: tac.Expr) -> TypeLattice:
    TOP = TypeLattice.top()
    match expr:
        case tac.Scope():
            if expr == tac.Scope.GLOBALS:
                return TypeLattice(BUILTINS_MODULE)
            elif expr == tac.Scope.LOCALS:
                return TypeLattice(LOCALS_MODULE)
            elif expr == tac.Scope.NONLOCALS:
                return TypeLattice(NONLOCALS_MODULE)
            else:
                assert False
        case tac.Const(): return TypeLattice(ObjectType.typeof(expr))
        case tac.Var():
            return types.get(expr, TOP)
        case tac.Attribute():
            t = eval(types, expr.var)
            if t.is_top() or t.is_bottom():
                return t
            match t.value:
                case ObjectType() as obj:
                    if expr.attr in obj.fields:
                        return TypeLattice(obj.fields[expr.attr])
                    else:
                        unseen[obj.name].add(expr.attr)
                    return TOP
                case FunctionType():
                    return TOP
            return types.get(expr, TOP)
        case tac.Subscr():
            array = eval(types, expr.var)
            if array.is_bottom():
                return TypeLattice.bottom()
            if array.is_top():
                return TOP
            f = array.value.fields.get(Var('__getitem__'))
            if f is None:
                print(f'eval({expr.var}) == {array} which is not an array')
                return TypeLattice.bottom()
            return TypeLattice(f.return_type)
        case tac.Call():
            function_signature = eval(types, expr.function)
            if function_signature == TOP:
                return TOP
            if not isinstance(function_signature.value, FunctionType):
                print(f'eval({expr.function}) == {function_signature} which is not a function')
                return TypeLattice.bottom()
            return TypeLattice(function_signature.value.return_type)
        case tac.Yield(): return TOP
        case tac.Import():
            return TypeLattice(modules[expr.modname])
        case tac.Binary():
            left = eval(types, expr.left)
            right = eval(types, expr.right)
            if left.is_top() or right.is_top() or left.is_bottom() or right.is_bottom():
                return TypeLattice.meet(left, right)
            if (left.value.name, right.value.name) in BINARY:
                return TypeLattice(BINARY[(left.value.name, right.value.name)][expr.op])
            return TOP
        case _: return TOP
