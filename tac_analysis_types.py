# Data flow analysis and stuff.

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Type, TypeVar, Optional, ClassVar, Final

import tac
from tac import Const, Var
from tac_analysis_domain import AbstractDomain, IterationStrategy, ForwardIterationStrategy, Bottom, Top

import graph_utils as gu

T = TypeVar('T')


@dataclass(frozen=True)
class ObjectType:
    type: str

    @staticmethod
    def typeof(const: tac.Const):
        return ObjectType(type(const.value).__name__)


@dataclass(frozen=True)
class TypeLattice(AbstractDomain):
    """
    Abstract domain for type analysis with lattice operations.
    For now, it is essentially constant domain
    """
    value: ObjectType | Bottom | Top

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
        res.types.update(dict(self.types.items() & other.types.items()))
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
            if isinstance(ins.lhs, tac.Var):
                self.types[ins.lhs] = eval(types, ins.expr)
        elif isinstance(ins, tac.Import):
            self.types[ins.lhs] = TypeLattice(ObjectType('Module'))
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
    match expr:
        case tac.Const(): return TypeLattice(ObjectType.typeof(expr))
        case tac.Var(): return types.get(expr, TypeLattice.top())
        case _: return TypeLattice.top()
