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
    methods: dict[Var, FunctionType]

    @staticmethod
    def typeof(const: tac.Const):
        match const.value:
            case int(): return INT
            case float(): return FLOAT
            case str(): return STRING
            case bool(): return BOOL
            case None: return NONE
            case _:return ObjectType(type(const.value).__name__, {})

    def __repr__(self):
        return self.name


@dataclass
class FunctionType:
    return_type: ObjectType

    def __repr__(self):
        return f'() -> {self.return_type}'


@dataclass
class ModuleType:
    name: str
    functions: dict[Var, FunctionType]

    def __repr__(self):
        return f'Module({self.name})'

    def __eq__(self, other):
        return isinstance(other, ModuleType) and self.name == other.name


FLOAT = ObjectType('float', {})
INT = ObjectType('int', {})
STRING = ObjectType('str', {})
BOOL = ObjectType('bool', {})
NONE = ObjectType('None', {})

NDARRAY = ObjectType('ndarray', {
    Var('mean'): FunctionType(FLOAT),
    Var('std'): FunctionType(FLOAT),
    Var('size'): INT,
})

ARRAY_GEN = FunctionType(NDARRAY)

NUMPY_MODULE = ModuleType('numpy', {
    Var('array'): ARRAY_GEN,
    Var('dot'): FunctionType(FLOAT),
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
    Var('random'): ARRAY_GEN,
    Var('rand'): ARRAY_GEN,
    Var('randn'): ARRAY_GEN,
})


@dataclass(frozen=True)
class TypeLattice(Lattice):
    """
    Abstract domain for type analysis with lattice operations.
    For now, it is essentially constant domain
    """

    @staticmethod
    def name() -> str:
        return "Type"

    value: ObjectType | FunctionType | ModuleType | Bottom | Top

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
        if isinstance(ins, tac.Mov):
            self.types[ins.lhs] = eval(types, ins.rhs)
        elif isinstance(ins, tac.Assign):
            self.types[ins.lhs] = eval(types, ins.expr)
        elif isinstance(ins, tac.Import):
            if ins.modname == "numpy":
                self.types[ins.lhs] = TypeLattice(NUMPY_MODULE)
            else:
                self.types[ins.lhs] = TypeLattice(ObjectType('Module', {}))
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


def eval(types: dict[Var, TypeLattice], expr: tac.Expr) -> TypeLattice:
    TOP = TypeLattice.top()
    match expr:
        case tac.Const(): return TypeLattice(ObjectType.typeof(expr))
        case tac.Var(): return types.get(expr, TOP)
        case tac.Attribute():
            t = eval(types, expr.var)
            if t == TOP or t == Bottom:
                return t
            match t.value:
                case ModuleType() as m:
                    if expr.attr in m.functions:
                        return TypeLattice(m.functions[expr.attr])
                    return TOP
                case ObjectType() as obj:
                    if expr.attr in obj.methods:
                        return TypeLattice(obj.methods[expr.attr])
                    else:
                        print(f'{expr.attr} not in {obj.methods}')
                    return TOP
                case FunctionType():
                    return TOP
            return types.get(expr, TOP)
        case tac.Subscr(): return types.get(expr, TOP)
        case tac.Yield(): return TOP
        case tac.Call():
            function_signature = eval(types, expr.function)
            if function_signature == TOP:
                return TOP
            if not isinstance(function_signature.value, FunctionType):
                print(f'eval({expr.function}) == {function_signature} which is not a function')
                return TypeLattice.bottom()
            return TypeLattice(function_signature.value.return_type)
        case _: return TOP
