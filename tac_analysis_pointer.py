# Data flow analysis and stuff.

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import chain
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
    LOCALS: Final[ClassVar[Object]] = Object(-1)
    GLOBALS: Final[ClassVar[Object]] = Object(-2)

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
        return PointerDomain({PointerDomain.LOCALS: {}, PointerDomain.GLOBALS: {}})

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
        state = self.copy().pointers
        activation = self.pointers[self.LOCALS]

        for var in tac.gens(ins):
            if var in activation:
                del activation[var]

        if isinstance(ins, tac.Mov):
            activation[ins.lhs] = eval(ins.rhs, state)
        elif isinstance(ins, tac.Assign):
            val = eval(ins.expr, state)
            if isinstance(ins.lhs, tac.Var):
                activation[ins.lhs] = val
            elif isinstance(ins.lhs, tac.Attribute):
                for obj in eval(ins.lhs.var, state):
                    self.pointers[obj][ins.lhs.attr] = val

    def __str__(self) -> str:
        return 'PointerDomain({})'.format(self.pointers)

    def __repr__(self) -> str:
        return self.pointers.__repr__()

    def keep_only_live_vars(self, alive_vars: set[tac.Var]) -> None:
        if self.is_bottom or self.is_top:
            return
        for var in self.pointers[self.LOCALS].keys() - alive_vars:
            del self.pointers[self.LOCALS][var]


def eval(expr: tac.Expr, state: dict[Object, dict[tac.Var, set[Object]]]) -> set[Object]:
    match expr:
        case tac.Const(): return set()
        case tac.Var(): return state[PointerDomain.LOCALS].get(expr, set()).copy()
        case tac.Attribute():
            if expr.var.name == 'GLOBALS':
                return state[PointerDomain.GLOBALS].get(expr.attr, set()).copy()
            else:
                return set(chain.from_iterable(state[obj].get(expr.attr, set()).copy()
                                               for obj in eval(expr.var, state)))
        case tac.Call():
            # if not expr.function.name[0].isupper():
            #     return set()
            return {Object(id(expr))}
        case tac.Subscr(): return set()
        case tac.Yield(): return set()
        case tac.Binary(): return set()
        case _: raise Exception(f"Unsupported expression {expr}")
