from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import TypeVar, Protocol, Generic, TypeAlias, Callable, Iterable, Optional
import graph_utils as gu

import tac

T = TypeVar('T')
K = TypeVar('K')


@dataclass
class ForwardIterationStrategy(Generic[T]):
    cfg: gu.Cfg[T]

    @property
    def entry_label(self):
        return self.cfg.entry_label

    def successors(self, label):
        return self.cfg.successors(label)

    def __getitem__(self, label) -> gu.Block:
        return self.cfg[label]

    def order(self, pair):
        return pair


@dataclass
class BackwardIterationStrategy(Generic[T]):
    cfg: gu.Cfg[T]

    @property
    def entry_label(self):
        return self.cfg.exit_label

    def successors(self, label):
        return self.cfg.predecessors(label)

    def __getitem__(self, label) -> gu.Block:
        return gu.BackwardBlock(self.cfg[label])

    def order(self, pair):
        return pair[1], pair[0]


IterationStrategy: TypeAlias = ForwardIterationStrategy | BackwardIterationStrategy


class Lattice(Protocol[T]):
    backward: bool

    def name(self) -> str:
        raise NotImplementedError

    def top(self) -> T:
        raise NotImplementedError

    def bottom(self) -> T:
        raise NotImplementedError

    def is_top(self, elem: T) -> bool:
        raise NotImplementedError

    def is_bottom(self, elem: T) -> bool:
        raise NotImplementedError

    def join(self, left: T, right: T) -> T:
        raise NotImplementedError

    def is_less_than(self, left: T, right: T) -> bool:
        raise NotImplementedError

    def is_equivalent(self, left, right) -> bool:
        raise NotImplementedError

    def copy(self, values: T) -> T:
        raise NotImplementedError

    def initial(self, annotations: dict[tac.Var, str]) -> T:
        raise NotImplementedError


@dataclass(frozen=True)
class Top:
    def __str__(self):
        return '⊤'

    def copy(self: T) -> T:
        return self

    def __or__(self, other):
        return self

    def __ror__(self, other):
        return self

    def __and__(self, other):
        return other

    def __rand__(self, other):
        return other


@dataclass(frozen=True)
class Bottom:
    def __str__(self):
        return '⊥'

    def copy(self: T) -> T:
        return self

    def __or__(self, other):
        return other

    def __ror__(self, other):
        return other

    def __rand__(self, other):
        return self

    def __and__(self, other):
        return self


BOTTOM = Bottom()
TOP = Top()


class Map(Generic[T]):
    # Essentially a defaultdict, but a defaultdict makes values appear out of nowhere
    _map: dict[tac.Var, T]
    default: T

    def __init__(self, default: T, d: Optional[dict[tac.Var, T]] = None):
        self.default = default
        self._map = {}
        if d is not None:
            self.update(d)

    def __getitem__(self, key: tac.Var) -> T:
        if not isinstance(key, tac.Var):
            breakpoint()
        assert isinstance(key, tac.Var), key
        return self._map.get(key, self.default)

    def __setitem__(self, key: tac.Var, value: T):
        assert isinstance(key, tac.Var), key
        if value == self.default:
            if key in self._map:
                del self._map[key]
        else:
            assert not isinstance(value, dict)
            assert not isinstance(value, Map)
            self._map[key] = value

    def update(self, dictionary: dict[tac.Var, T] | Map) -> None:
        for k, v in dictionary.items():
            self[k] = v

    def __iter__(self):
        return iter(self._map)

    def __contains__(self, key: tac.Var):
        return key in self._map

    def __len__(self):
        return len(self._map)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Map) and self._map == other._map

    def __delitem__(self, key: tac.Var) -> None:
        del self._map[key]

    def __repr__(self) -> str:
        items = ', '.join(f'{k}={v}' for k, v in self._map.items())
        return f'Map({items})'

    def __str__(self) -> str:
        return repr(self)

    def items(self) -> Iterable[tuple[tac.Var, T]]:
        return self._map.items()

    def values(self) -> Iterable[T]:
        return self._map.values()

    def keys(self) -> set[tac.Var]:
        return set(self._map.keys())

    def copy(self) -> Map:
        return Map(self.default, self._map)


MapDomain: TypeAlias = Map[T] | Bottom


def normalize(values: MapDomain[T]) -> MapDomain[T]:
    if isinstance(values, Bottom):
        return BOTTOM
    if any(isinstance(v, Bottom) for v in values.values()):
        return BOTTOM
    return values


class InstructionLattice(Lattice[T], Generic[T]):
    def transfer(self, values: T, ins: tac.Tac, location: tuple[int, int]) -> T:
        raise NotImplementedError


class ValueLattice(Lattice[T], Generic[T]):
    def const(self, value: object) -> T:
        return self.top()

    def var(self, value: T) -> T:
        return value

    def attribute(self, var: T, attr: tac.Var) -> T:
        assert isinstance(attr, tac.Var)
        return self.top()

    def subscr(self, array: T, index: T) -> T:
        return self.top()

    def call(self, function: T, args: list[T]) -> T:
        return self.top()

    def binary(self, left: T, right: T, op: str) -> T:
        return self.top()

    def unary(self, value: T, op: tac.UnOp) -> T:
        return self.top()

    def predefined(self, name: tac.Predefined) -> T:
        return self.top()

    def imported(self, modname: str) -> T:
        return self.top()

    def annotation(self, name: tac.Var, t: str) -> T:
        return self.top()

    def assign_var(self, value: T) -> T:
        return value

    def default(self) -> T:
        return self.top()



InvariantMap: TypeAlias = dict[tuple[int, int], T]


class VarLattice(InstructionLattice[MapDomain[T]], Generic[T]):
    lattice: ValueLattice[T]
    liveness: InvariantMap[Map[Lattice[Top | Bottom]]]
    backward: bool = False

    def __init__(self, lattice: ValueLattice[T], liveness: InvariantMap[Map[Lattice[Top | Bottom]]]):
        super().__init__()
        self.lattice = lattice
        self.liveness = liveness

    def name(self) -> str:
        return f"{self.lattice.name()}"

    def is_less_than(self, left: MapDomain[T], right: MapDomain[T]) -> bool:
        return self.join(left, right) == right

    def is_equivalent(self, left: MapDomain[T], right: MapDomain[T]) -> bool:
        return self.is_less_than(left, right) and self.is_less_than(right, left)

    def copy(self, values: MapDomain[T]) -> MapDomain[T]:
        return values.copy()

    def is_bottom(self, values: MapDomain[T]) -> bool:
        return isinstance(values, Bottom)

    def make_map(self, d: Optional[dict[tac.Var, T]] = None) -> Map[T]:
        d = d or {}
        return Map(default=self.lattice.default(), d=d)

    def initial(self, annotations: dict[tac.Var, str]) -> MapDomain[T]:
        result = self.make_map()
        result.update({
            name: self.lattice.annotation(name, t)
            for name, t in annotations.items()
        })
        return result

    def top(self) -> Map[T]:
        return self.make_map()

    def bottom(self) -> MapDomain[T]:
        return BOTTOM

    def join(self, left: MapDomain[T], right: MapDomain[T]) -> MapDomain[T]:
        match left, right:
            case (Bottom(), _): return right
            case (_, Bottom()): return left
            case (Map() as left, Map() as right):
                res: Map[T] = self.top()
                for k in left.keys() | right.keys():
                    res[k] = self.lattice.join(left[k], right[k])
                return normalize(res)
        return self.top()

    def transformer_expr(self, values: Map[T], expr: tac.Expr) -> T:
        def eval(expr: tac.Var | tac.Predefined) -> T:
            res = self.transformer_expr(values, expr)
            return res

        match expr:
            case tac.Var():
                return self.lattice.var(values[expr])
            case tac.Attribute():
                val = eval(expr.var)
                return self.lattice.attribute(val, expr.field)
            case tac.Call(function=function, args=args):
                func = eval(function)
                return self.lattice.call(
                    function=func,
                    args=[eval(arg) for arg in args]
                )
            case tac.Unary(var=var, op=op):
                value = eval(var)
                assert value is not None
                return self.lattice.unary(value=value, op=op)
            case tac.Binary() as expr:
                left = eval(expr.left)
                right = eval(expr.right)
                return self.lattice.binary(left=left, right=right, op=expr.op)
            case tac.Predefined() as expr:
                return self.lattice.predefined(expr)
            case tac.Const(value=value):
                return self.lattice.const(value)
            case tac.Subscript(var=var, index=index):
                return self.lattice.subscr(eval(var), eval(index))
            case tac.Yield():
                return self.lattice.top()
            case tac.Import():
                print(f"{expr}: {type(expr)}; {expr.modname} {type(expr.modname)}")
                if isinstance(expr.modname, tac.Attribute):
                    val = eval(expr.modname.var)
                    return self.lattice.attribute(val, expr.modname.field)
                else:
                    return self.lattice.imported(expr.modname)
            case tac.MakeFunction():
                return self.lattice.top()
            case _:
                assert False, f'unexpected expr of type {type(expr)}: {expr}'

    def transformer_signature(self, value: T, signature: tac.Signature) -> Map[T]:
        match signature:
            case tuple() as signature:  # type: ignore
                assert all(isinstance(v, tac.Var) for v in signature)
                return self.make_map({var: self.lattice.subscr(value, self.lattice.const(i))
                                      for i, var in enumerate(signature)})
            case tac.Var():
                return self.make_map({signature: self.lattice.assign_var(value)})
            case tac.Attribute():
                return self.make_map()
            case tac.Subscript():
                return self.make_map()
            case _:
                assert False, f'unexpected signature {signature}'

    def forward_transfer(self, values: MapDomain[T], ins: tac.Tac, location: tuple[int, int]) -> MapDomain[T]:
        if isinstance(values, Bottom):
            return BOTTOM
        if isinstance(ins, tac.For):
            ins = ins.as_call()
        updated = self.make_map()
        match ins:
            case tac.Assign():
                assigned = self.transformer_expr(values, ins.expr)
                updated = self.transformer_signature(assigned, ins.lhs)
            case tac.Return():
                assigned = self.transformer_expr(values, ins.value)
                updated = self.make_map({
                    tac.Var('return'): assigned
                })
        return updated

    def transfer(self, values: MapDomain[T], ins: tac.Tac, location: tuple[int, int]) -> MapDomain[T]:
        if isinstance(values, Bottom):
            return BOTTOM
        values = values.copy()
        to_update = self.forward_transfer(values, ins, location)
        for var in tac.gens(ins):
            if var in values:
                del values[var]
        if isinstance(to_update, Map):
            values.update(to_update)

        for var in set(values.keys()):
            here = self.liveness[location]
            if var.is_stackvar and here[var] is BOTTOM:
                del values[var]
        return normalize(values)
