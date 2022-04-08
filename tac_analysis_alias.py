# Data flow analysis and stuff.

from __future__ import annotations

import dataclasses
from copy import deepcopy

from networkx.utils import UnionFind
from dataclasses import dataclass
from typing import Type, TypeVar, Optional, ClassVar

import graph_utils
import tac
from tac_analysis_domain import AbstractDomain, IterationStrategy, ForwardIterationStrategy

import graph_utils as gu

T = TypeVar('T')


@dataclass
class Bottom:
    pass


class AliasDomain(AbstractDomain):
    # map stack variable to program variable or constant
    alias: dict[tac.Var, tac.Var | tac.Attribute | tac.Expr] | Bottom

    BOTTOM: ClassVar[Bottom] = Bottom()

    @staticmethod
    def name() -> str:
        return "Alias"

    @classmethod
    def view(cls, cfg: gu.Cfg[T]) -> IterationStrategy[T]:
        return ForwardIterationStrategy(cfg)

    def __init__(self, alias: dict[tac.Var, tac.Var] | Bottom) -> None:
        super().__init__()
        if alias is AliasDomain.BOTTOM:
            self.alias = AliasDomain.BOTTOM
        else:
            self.alias = alias.copy()

    def __le__(self, other):
        return self.join(other).alias == other.alias

    def __eq__(self, other):
        return self.alias == other.alias

    def copy(self: T) -> T:
        return AliasDomain(self.alias)

    @classmethod
    def initial(cls: Type[T]) -> T:
        return cls.top()

    @classmethod
    def top(cls: Type[T]) -> T:
        return AliasDomain({})

    @classmethod
    def bottom(cls: Type[T]) -> T:
        return AliasDomain(AliasDomain.BOTTOM)

    @property
    def is_bottom(self) -> bool:
        return self.alias == AliasDomain.BOTTOM

    def join(self: T, other: T) -> T:
        if self.is_bottom:
            return other.copy()
        if other.is_bottom:
            return self.copy()
        return AliasDomain(dict(self.alias.items() & other.alias.items()))

    def transfer(self, ins: tac.Tac, location: str) -> None:
        if self.is_bottom:
            return
        self.alias = {stackvar: v for stackvar, v in self.alias.items()
                      if stackvar not in tac.gens(ins)
                      and stackvar not in tac.free_vars(ins)}
        if isinstance(ins, tac.Mov):
            left, right = (ins.lhs, ins.rhs) if ins.lhs.is_stackvar else (ins.rhs, ins.lhs)
            self.alias[left] = right
        elif isinstance(ins, tac.Assign) and isinstance(ins.expr, (tac.Var, tac.Attribute)) and isinstance(ins.lhs, tac.Var) and ins.lhs not in tac.free_vars_expr(ins.expr):
            left, right = ins.lhs, ins.expr
            self.alias[left] = right

    def __str__(self) -> str:
        return 'AliasDomain({})'.format(", ".join(f'{k}={v}' for k, v in self.alias.items()))

    def __repr__(self) -> str:
        return 'AliasDomain({})'.format(", ".join(f'{k}={v}' for k, v in self.alias.items()))

    def keep_only_live_vars(self, alive_vars: set[tac.Var]) -> None:
        self.alias = {k: v for k, v in self.alias.items() if k in alive_vars}


def rewrite_ins(ins: tac.Tac, pre: AliasDomain) -> tac.Tac:
    match ins:
        case tac.Mov():
            if ins.rhs in pre.alias:
                return dataclasses.replace(ins, rhs=pre.alias[ins.rhs])
            return ins
        case tac.Assign():
            lhs = ins.lhs
            if isinstance(ins.lhs, tac.Attribute) and ins.lhs.var in pre.alias:
                lhs = dataclasses.replace(ins.lhs, var=pre.alias[ins.lhs.var])
            ins = dataclasses.replace(ins, lhs=lhs)
            if not (tac.free_vars_expr(ins.expr) & pre.alias.keys()):
                return ins
            expr = ins.expr
            match ins.expr:
                case tac.Var():
                    expr = pre.alias[ins.expr]
                case tac.Attribute():
                    expr = dataclasses.replace(ins.expr, var=pre.alias[ins.expr.var])
                case tac.Binary():
                    expr = dataclasses.replace(ins.expr,
                                       left=pre.alias.get(ins.expr.left, ins.expr.left),
                                       right=pre.alias.get(ins.expr.right, ins.expr.right))
                case tac.Call():
                    expr = dataclasses.replace(ins.expr,
                                               function=pre.alias.get(ins.expr.function, ins.expr.function),
                                               args=tuple(pre.alias.get(arg, arg) for arg in ins.expr.args))
            ins = dataclasses.replace(ins, expr=expr)
            return ins
        case _: pass
    return ins


def rewrite_aliases(block: graph_utils.Block, label: int) -> None:
    invariant: AliasDomain = block.pre[AliasDomain.name()]
    if invariant.is_bottom:
        return
    for i in range(len(block)):
        block[i] = rewrite_ins(block[i], invariant)
        invariant.transfer(block[i], f'{label}.{i}')
