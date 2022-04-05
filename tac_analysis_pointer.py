# Data flow analysis and stuff.

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Type, TypeVar, Optional, ClassVar, Final

import networkx as nx

import tac
from tac_analysis_domain import AbstractDomain, IterationStrategy, ForwardIterationStrategy

import graph_utils as gu

T = TypeVar('T')


@dataclass(frozen=True)
class Top:
    pass


@dataclass(frozen=True)
class Bottom:
    pass


@dataclass(frozen=True)
class Object:
    location: int


@dataclass
class PointerDomain(AbstractDomain):
    pointers: dict[Object, dict[tac.Var, set[Object]]] | Bottom | Top

    BOTTOM: Final[ClassVar[Bottom]] = Bottom()
    TOP: Final[ClassVar[Top]] = Top()
    LOCALS: Final[ClassVar[Object]] = Object(0)

    @staticmethod
    def name() -> str:
        return "Pointer"

    @classmethod
    def view(cls, cfg: gu.Cfg[T]) -> IterationStrategy[T]:
        return ForwardIterationStrategy(cfg)

    def __init__(self, pointers: dict[Object, dict[tac.Var, set[Object]]] | Bottom | Top) -> None:
        super().__init__()
        if pointers == PointerDomain.BOTTOM:
            self.pointers = PointerDomain.BOTTOM
        elif pointers == PointerDomain.TOP:
            self.pointers = PointerDomain.TOP
        else:
            self.pointers = pointers.copy()

    def __le__(self, other):
        return self.join(other).pointers == other.pointers

    def __eq__(self, other):
        return self.pointers == other.pointers

    def copy(self: T) -> T:
        return PointerDomain(deepcopy(self.pointers))

    @classmethod
    def initial(cls: Type[T]) -> T:
        return PointerDomain({PointerDomain.LOCALS: {}})

    @classmethod
    def top(cls: Type[T]) -> T:
        return PointerDomain(PointerDomain.TOP)

    @classmethod
    def bottom(cls: Type[T]) -> T:
        return PointerDomain(PointerDomain.BOTTOM)

    @property
    def is_bottom(self) -> bool:
        return self.pointers == PointerDomain.BOTTOM

    @property
    def is_top(self) -> bool:
        return self.pointers == PointerDomain.TOP

    def join(self, other: PointerDomain) -> PointerDomain:
        if self.is_bottom or other.is_top:
            return other.copy()
        if other.is_bottom or self.is_top:
            return self.copy()
        pointers = deepcopy(self.pointers)
        for obj, fields in other.pointers.items():
            if obj in pointers:
                for field, values in fields.items():
                    pointers[obj][field] = pointers[obj].get(field, set()) | values
            else:
                pointers[obj] = deepcopy(fields)
        return PointerDomain(pointers)

    def transfer(self, ins: tac.Tac) -> None:
        if self.is_bottom or self.is_top:
            return
        activation = self.pointers[self.LOCALS]
        if isinstance(ins, tac.Mov):
            activation[ins.lhs] = self.eval(ins.rhs)
        elif isinstance(ins, tac.Assign):
            if isinstance(ins.lhs, tac.Var) and (val := self.eval(ins.expr)) is not None:
                activation[ins.lhs] = val
            else:
                for var in tac.gens(ins):
                    if var in activation:
                        del activation[var]

    def eval(self, expr: tac.Expr) -> set[Object]:
        match expr:
            case tac.Const(): return set()
            case tac.Var(): return self.pointers[self.LOCALS].get(expr, set()).copy()
            case tac.Attribute(): return set()
            case tac.Call():
                if expr.function.name[0].isupper():
                    return {Object(id(expr))}
                return set()
            case tac.Subscr(): return set()
            case tac.Yield(): return set()
            case tac.Binary(): return set()
            case _: raise Exception(f"Unsupported expression {expr}")

    def __str__(self) -> str:
        return 'PointerDomain({})'.format(self.pointers)

    def __repr__(self) -> str:
        return self.pointers.__repr__()

    def keep_only_live_vars(self, alive_vars: set[tac.Var]) -> None:
        if self.is_bottom or self.is_top:
            return
        for var in self.pointers[self.LOCALS].keys() - alive_vars:
            del self.pointers[self.LOCALS][var]
