# Data flow analysis and stuff.

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Type, TypeVar, Optional, ClassVar, Final

import tac
from tac import Const, Var
from tac_analysis_domain import AbstractDomain, IterationStrategy, ForwardIterationStrategy, Bottom, Top, Lattice

import graph_utils as gu

T = TypeVar('T')


@dataclass()
class ObjectType:
    name: str
    fields: dict[Var, FunctionType | ObjectType]

    @staticmethod
    def translate(t: str):
        match t:
            case "np.ndarray" | "numpy.ndarray" | "ndarray": return NDARRAY
            case "int": return INT
            case "float": return FLOAT
            case "str": return STRING
            case "bool": return BOOL
        return ObjectType(type(t).__name__, {})

    @staticmethod
    def typeof(const: tac.Const):
        return ObjectType.translate(type(const.value).__name__) if const.value is not None else NONE

    def __repr__(self):
        return self.name


@dataclass
class FunctionType:
    return_type: ObjectType
    new: bool = True

    def __repr__(self):
        return f'() -> {self.return_type}'


FLOAT = ObjectType('float', {})
OBJECT = ObjectType('object', {})
INT = ObjectType('int', {})
STRING = ObjectType('str', {})
BOOL = ObjectType('bool', {})
NONE = ObjectType('None', {})


def make_tuple(t: ObjectType) -> ObjectType:
    return ObjectType('tuple', {
        Var('__getitem__'): FunctionType(t),
    })


def iter_method(element_type: ObjectType) -> FunctionType:
    return FunctionType(ObjectType(f'iterator[{element_type}]', {
        Var('__next__'): FunctionType(element_type),
    }))


NDARRAY = ObjectType('ndarray', {
    Var('mean'): FunctionType(FLOAT, new=False),
    Var('std'): FunctionType(FLOAT, new=False),
    Var('shape'): make_tuple(INT),
    Var('size'): INT,
    Var('__getitem__'): FunctionType(FLOAT),
    Var('__iter__'): iter_method(FLOAT),  # inaccurate
})


LIST = ObjectType('list', {
    Var('__getitem__'): FunctionType(OBJECT, new=False),
    Var('__iter__'): iter_method(OBJECT),
    Var('__len__'): FunctionType(INT, new=False),
    Var('__contains__'): FunctionType(BOOL, new=False),
    Var('clear'): FunctionType(NONE, new=False),
    Var('copy'): FunctionType(NONE, new=False),
    Var('count'): FunctionType(INT, new=False),
    Var('extend'): FunctionType(NONE),
    Var('index'): FunctionType(INT, new=False),
    Var('insert'): FunctionType(NONE, new=False),
    Var('pop'): FunctionType(OBJECT),
    Var('remove'): FunctionType(NONE, new=False),
    Var('reverse'): FunctionType(NONE, new=False),
    Var('sort'): FunctionType(NONE, new=False),
})

NDARRAY.fields[Var('T')] = NDARRAY

ARRAY_GEN = FunctionType(NDARRAY)

NDARRAY.fields[Var('astype')] = ARRAY_GEN


DATAFRAME = ObjectType('DataFrame', {

})

TIME_MODULE = ObjectType('/time', {

})

NUMPY_MODULE = ObjectType('/numpy', {
    Var('array'): ARRAY_GEN,
    Var('dot'): FunctionType(FLOAT, new=False),
    Var('zeros'): ARRAY_GEN,
    Var('ones'): ARRAY_GEN,
    Var('concatenate'): ARRAY_GEN,
    Var('empty'): ARRAY_GEN,
    Var('empty_like'): ARRAY_GEN,
    Var('full'): ARRAY_GEN,
    Var('full_like'): ARRAY_GEN,
    Var('arange'): ARRAY_GEN,
    Var('linspace'): ARRAY_GEN,
    Var('logspace'): ARRAY_GEN,
    Var('geomspace'): ARRAY_GEN,
    Var('meshgrid'): ARRAY_GEN,
    Var('max'): FunctionType(FLOAT, new=False),
    Var('min'): FunctionType(FLOAT, new=False),
    Var('sum'): FunctionType(FLOAT, new=False),
    Var('setdiff1d'): ARRAY_GEN,
    Var('unique'): ARRAY_GEN,
    Var('append'): ARRAY_GEN,
    Var('random'): ARRAY_GEN,
    Var('argmax'): FunctionType(INT, new=False),
    Var('c_'): ObjectType('slice_trick', {
        Var('__getitem__'): FunctionType(NDARRAY),
    }),
    Var('r_'): ObjectType('slice_trick', {
        Var('__getitem__'): FunctionType(NDARRAY),
    }),

})

PANDAS_MODULE = ObjectType('/pandas', {
    Var('DataFrame'): FunctionType(DATAFRAME),
})

BUILTINS_MODULE = ObjectType('/builtins', {
    Var('range'): FunctionType(ObjectType('range', {
        Var('__iter__'): iter_method(INT),
    })),
    Var('len'): FunctionType(INT, new=False),
    Var('print'): FunctionType(NONE, new=False),
    Var('abs'): FunctionType(FLOAT, new=False),
    Var('round'): FunctionType(FLOAT, new=False),
    Var('min'): FunctionType(FLOAT, new=False),
    Var('max'): FunctionType(FLOAT, new=False),
    Var('sum'): FunctionType(FLOAT, new=False),
    Var('all'): FunctionType(BOOL, new=False),
    Var('any'): FunctionType(BOOL, new=False),
    Var('int'): FunctionType(INT, new=False),
    Var('float'): FunctionType(FLOAT, new=False),
    Var('str'): FunctionType(STRING, new=False),
    Var('bool'): FunctionType(BOOL, new=False),
})

modules = {
    'builtins': BUILTINS_MODULE,
    'numpy': NUMPY_MODULE,
    'pandas': PANDAS_MODULE,
    'time': TIME_MODULE,
}

BINARY = {
    (INT.name, INT.name): {
        '/': FunctionType(FLOAT, new=False),
        **{op: FunctionType(INT, new=False) for op in ['+', '-', '*', '**', '//', '%']},
        **{op: FunctionType(INT, new=False) for op in ['&', '|', '^', '<<', '>>']},
        **{op: FunctionType(BOOL, new=False) for op in ['<', '>', '<=', '>=', '==', '!=']},
    },
    (INT.name, FLOAT.name): {
        **{op: FunctionType(FLOAT, new=False) for op in ['+', '-', '*', '/', '**', '//', '%']},
        **{op: FunctionType(BOOL, new=False) for op in ['<', '>', '<=', '>=', '==', '!=']},
    },
    (FLOAT.name, INT.name): {
        **{op: FunctionType(FLOAT, new=False) for op in ['+', '-', '*', '/', '**', '//', '%']},
        **{op: FunctionType(BOOL, new=False) for op in ['<', '>', '<=', '>=', '==', '!=']},
    },
    (FLOAT.name, FLOAT.name): {
        **{op: FunctionType(FLOAT, new=False) for op in ['+', '-', '*', '/', '**', '//', '%']},
        **{op: FunctionType(BOOL, new=False) for op in ['<', '>', '<=', '>=', '==', '!=']},
    },
    (NDARRAY.name, NDARRAY.name): {op: FunctionType(NDARRAY, new=True) for op in ['+', '-', '*', '/', '**', '//', '%', '@', '==', '!=', '<', '<=', '>', '>=']},
    (NDARRAY.name, INT.name): {op: FunctionType(NDARRAY, new=True) for op in ['+', '-', '*', '/', '**', '//', '%', '@', '==', '!=', '<', '<=', '>', '>=']},
    (INT.name, NDARRAY.name): {op: FunctionType(NDARRAY, new=True) for op in ['+', '-', '*', '/', '**', '//', '%', '@', '==', '!=', '<', '<=', '>', '>=']},
    (NDARRAY.name, FLOAT.name): {op: FunctionType(NDARRAY, new=True) for op in ['+', '-', '*', '/', '**', '//', '%', '@', '==', '!=', '<', '<=', '>', '>=']},
    (FLOAT.name, NDARRAY.name): {op: FunctionType(NDARRAY, new=True) for op in ['+', '-', '*', '/', '**', '//', '%', '@', '==', '!=', '<', '<=', '>', '>=']},
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
    def read_initial(cls, annotations: dict[str, str]) -> TypeDomain:
        result = TypeDomain.top()
        result.types.update({
            tac.Var(name): TypeLattice(ObjectType.translate(t))
            for name, t in annotations.items()
        })
        return result


unseen = defaultdict(set)


def eval(types: dict[Var, TypeLattice], expr: tac.Expr) -> TypeLattice:
    TOP = TypeLattice.top()
    match expr:
        case tac.Const(): return TypeLattice(ObjectType.typeof(expr))
        case tac.Var():
            if expr.name == 'GLOBALS':
                return TypeLattice(modules['builtins'])
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
            if expr.function == Var('LIST'):
                return TypeLattice(LIST)
            function_signature = eval(types, expr.function)
            if function_signature == TOP:
                return TOP
            if not isinstance(function_signature.value, FunctionType):
                print(f'eval({expr.function}) == {function_signature} which is not a function')
                return TypeLattice.bottom()
            expr.is_allocation = function_signature.value.new
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
                ftype = BINARY[(left.value.name, right.value.name)][expr.op]
                expr.is_allocation = ftype.new
                return TypeLattice(ftype.return_type)
            return TOP
        case _: return TOP
