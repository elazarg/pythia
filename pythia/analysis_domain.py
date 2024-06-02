from __future__ import annotations as _

import typing
from copy import deepcopy
from dataclasses import dataclass

from pythia import graph_utils as gu
from pythia import tac
from pythia.graph_utils import Label, Location


@dataclass
class ForwardIterationStrategy[T]:
    cfg: gu.Cfg[T]

    @property
    def entry_label(self) -> Label:
        return self.cfg.entry_label

    def successors(self, label: Label) -> typing.Iterator[Label]:
        return self.cfg.successors(label)

    def predecessors(self, label: Label) -> typing.Iterator[Label]:
        return self.cfg.predecessors(label)

    def __getitem__(self, label: Label) -> gu.Block:
        return self.cfg[label]

    def order[K, Q](self, pair: tuple[K, Q]) -> tuple[K, Q]:
        return pair


@dataclass
class BackwardIterationStrategy[T]:
    cfg: gu.Cfg[T]

    @property
    def entry_label(self) -> Label:
        return self.cfg.exit_label

    def successors(self, label: Label) -> typing.Iterator[Label]:
        return self.cfg.predecessors(label)

    def predecessors(self, label: Label) -> typing.Iterator[Label]:
        return self.cfg.successors(label)

    def __getitem__(self, label: Label) -> gu.BackwardBlock:
        return typing.cast(gu.BackwardBlock, reversed(self.cfg[label]))

    def order[K, Q](self, pair: tuple[K, Q]) -> tuple[Q, K]:
        return pair[1], pair[0]


type IterationStrategy = ForwardIterationStrategy | BackwardIterationStrategy


def iteration_strategy(
    cfg: gu.Cfg[tac.Tac], backward: bool
) -> IterationStrategy[tac.Tac]:
    return BackwardIterationStrategy(cfg) if backward else ForwardIterationStrategy(cfg)


class Lattice[T](typing.Protocol):

    def name(self) -> str:
        raise NotImplementedError

    def top(self) -> T:
        raise NotImplementedError

    def bottom(self) -> T:
        raise NotImplementedError

    def is_bottom(self, elem: T) -> bool:
        raise NotImplementedError

    def join(self, left: T, right: T) -> T:
        raise NotImplementedError

    def join_all(self, inv: T, *invs: T) -> T:
        for arg in invs:
            inv = self.join(inv, arg)
        return inv

    def is_less_than(self, left: T, right: T) -> bool:
        raise NotImplementedError


@dataclass(frozen=True)
class Top:
    def __str__(self) -> str:
        return "⊤"

    def __or__(self, other: object) -> Top:
        return self

    def __ror__(self, other: object) -> Top:
        return self

    def __and__[T](self, other: T) -> T:
        return other

    def __rand__[T](self, other: T) -> T:
        return other

    def __contains__(self, item: object) -> bool:
        return True

    def __copy__(self):
        return self

    def __deepcopy__(self, memodict={}):
        return self


@dataclass(frozen=True)
class Bottom:
    def __str__(self) -> str:
        return "⊥"

    def __deepcopy__(self, memodict={}):
        memodict[id(self)] = self
        return self

    def __or__[T](self, other: T) -> T:
        return other

    def __ror__[T](self, other: T) -> T:
        return other

    def __rand__(self, other: object) -> Bottom:
        return self

    def __and__(self, other: object) -> Bottom:
        return self


BOTTOM = Bottom()
TOP = Top()


class Set[T]:
    _set: frozenset[T] | Top

    def __init__(self, s: typing.Optional[typing.Iterable[T] | Top] = None):
        if s is None:
            self._set = frozenset()
        elif s is TOP:
            self._set = TOP
        else:
            self._set = frozenset(s)

    def __repr__(self):
        if isinstance(self._set, Top):
            return "Set(TOP)"
        items = ", ".join(repr(x) for x in self._set)
        return f"Set({items})"

    def __deepcopy__(self, memodict={}):
        if isinstance(self._set, Top):
            result = Set(TOP)
        else:
            result = Set([deepcopy(x, memodict) for x in self._set])
        memodict[id(self)] = result
        return result

    @classmethod
    def top(cls: typing.Type[Set[T]]) -> Set[T]:
        return Set(TOP)

    def meet(self, other: Set[T]) -> Set[T]:
        if isinstance(self._set, Top):
            return other
        if isinstance(other._set, Top):
            return self
        return Set(self._set & other._set)

    def join(self, other: Set[T]) -> Set[T]:
        if isinstance(self._set, Top) or isinstance(other._set, Top):
            return Set(TOP)
        return Set(self._set | other._set)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Set) and self._set == other._set

    def __contains__(self, item: T) -> bool:
        if isinstance(self._set, Top):
            return True
        return item in self._set

    def __bool__(self) -> bool:
        if isinstance(self._set, Top):
            return True
        return bool(self._set)

    def is_subset(self, other: Set[T]) -> bool:
        if isinstance(other._set, Top):
            return True
        else:
            if isinstance(self._set, Top):
                return False
            return self._set <= other._set

    def __le__(self: Set[T], other: Set[T]) -> bool:
        return self.is_subset(other)

    def is_less_than(self, other: Set[T]) -> bool:
        return self.is_subset(other)

    def __or__(self: Set[T], other: Set[T]) -> Set[T]:
        return self.join(other)

    def __and__(self: Set[T], other: Set[T]) -> Set[T]:
        return self.meet(other)

    @classmethod
    def squeeze(cls: typing.Type[Set[T]], s: T | Set[T]) -> T | Set[T]:
        if isinstance(s, Set) and not isinstance(s._set, Top) and len(s._set) == 1:
            return next(iter(s._set))
        return s

    @classmethod
    def union_all(cls: typing.Type[Set[T]], xs: typing.Iterable[Set[T]]) -> Set[T]:
        result: Set[T] = Set()
        for x in xs:
            if isinstance(x._set, Top):
                result._set = TOP
                break
            result._set |= x._set
        return result

    @classmethod
    def singleton(cls: typing.Type[Set[T]], x: T) -> Set[T]:
        return Set({x})

    def as_set(self) -> frozenset[T]:
        assert not isinstance(self._set, Top)
        return self._set


class Map[K, T]:
    # Essentially a defaultdict, but a defaultdict makes values appear out of nowhere
    _map: dict[K, T]
    default: T

    def __init__(
        self,
        default: typing.Callable[[], T],
        d: typing.Optional[typing.Mapping[K, T]] = None,
    ):
        self.default = default
        self._map = {}
        if d:
            self.update(d)

    def __getitem__(self, key: K) -> T:
        m = self._map
        if key in m:
            return m[key]
        return self.default()

    def __setitem__(self, key: K, value: T) -> None:
        if value == self.default():
            if key in self._map:
                del self._map[key]
        else:
            # assert not isinstance(value, dict)
            # assert not isinstance(value, Map)
            self._map[key] = value

    def update(self, dictionary: typing.Mapping[K, T] | Map) -> None:
        default = self.default()
        m = self._map
        for key, value in dictionary.items():
            if value == default:
                if key in m:
                    del m[key]
            else:
                # assert not isinstance(value, dict)
                # assert not isinstance(value, Map)
                m[key] = deepcopy(value)

    def __iter__(self) -> typing.Iterator[K]:
        return iter(self._map)

    def __contains__(self, key: K) -> bool:
        return key in self._map

    def __bool__(self) -> bool:
        return bool(self._map)

    def __len__(self) -> int:
        return len(self._map)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Map) and self._map == other._map

    def __delitem__(self, key: K) -> None:
        del self._map[key]

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v}" for k, v in self._map.items())
        return f"Map({items})"

    def __str__(self) -> str:
        return repr(self)

    def items(self) -> typing.Iterable[tuple[K, T]]:
        return self._map.items()

    def values(self) -> typing.Iterable[T]:
        return self._map.values()

    def keys(self) -> set[K]:
        return set(self._map.keys())

    def __deepcopy__(self, memodict={}):
        # deepcopy is done in the constructor
        return Map(self.default, {k: v for k, v in self._map.items()})

    def print(self) -> None:
        print(str(self))

    def keep_keys(self, keys: typing.Iterable[K]) -> None:
        for k in list(self._map.keys()):
            if k not in keys:
                del self._map[k]

    def join(self, other: Map[K, T]) -> Map[K, T]:
        result: Map[K, T] = Map(self.default)
        for k in {*other.keys(), *self.keys()}:
            result[k] = self[k].join(other[k])
        return result

    def __or__(self, other: Map[K, T]) -> Map[K, T]:
        result: Map[K, T] = Map(self.default)
        for k in {*other.keys(), *self.keys()}:
            self[k] = self[k] | other[k]
        return result

    def is_less_than(self, other: Map[K, T]) -> bool:
        return all(v.is_less_than(other[k]) for k, v in self.items())


type MapDomain[K, T] = Map[K, T] | Bottom
type VarMapDomain[K, T] = MapDomain[tac.Var, T]


class InstructionLattice[T](Lattice[T], typing.Protocol):
    backward: bool

    def transfer(self, values: T, ins: tac.Tac, location: Location) -> T:
        raise NotImplementedError

    def initial(self) -> T:
        return self.top()


class ValueLattice[T](Lattice[T], typing.Protocol):
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

    def default(self) -> T:
        return self.top()


type InvariantMap[T] = dict[Label, T]


class VarLattice[T](InstructionLattice[VarMapDomain[T]]):
    lattice: ValueLattice[T]
    liveness: dict[Location, VarMapDomain[Top | Bottom]]
    backward: bool = False

    def __init__(
        self,
        lattice: ValueLattice[T],
        liveness: dict[Location, VarMapDomain[Top | Bottom]],
    ):
        super().__init__()
        self.lattice = lattice
        self.liveness = liveness

    def name(self) -> str:
        return f"{self.lattice.name()}"

    def is_less_than(self, left: VarMapDomain[T], right: VarMapDomain[T]) -> bool:
        if self.is_bottom(left):
            return True
        if self.is_bottom(right):
            return False
        return all(self.lattice.is_less_than(left[k], right[k]) for k in left.keys())
        # return self.join(left, right) == right

    def is_bottom(self, values: VarMapDomain[T]) -> bool:
        return isinstance(values, Bottom)

    def make_map(self, d: typing.Optional[dict[tac.Var, T]] = None) -> Map[tac.Var, T]:
        d = d or {}
        return Map(default=self.lattice.default, d=d)

    def top(self) -> Map[tac.Var, T]:
        return self.make_map()

    def bottom(self) -> VarMapDomain[T]:
        return BOTTOM

    def normalize(self, values: VarMapDomain[T]) -> VarMapDomain[T]:
        if isinstance(values, Bottom):
            return BOTTOM
        if any(self.lattice.is_bottom(v) for v in values.values()):
            return BOTTOM
        return values

    def join(self, left: VarMapDomain[T], right: VarMapDomain[T]) -> VarMapDomain[T]:
        match left, right:
            case (Bottom(), _):
                return right
            case (_, Bottom()):
                return left
            case (Map() as left, Map() as right):
                res: Map[tac.Var, T] = self.top()
                for k in left.keys() | right.keys():
                    res[k] = self.lattice.join(left[k], right[k])
                return self.normalize(res)
        return self.top()

    def transformer_expr(self, values: Map[tac.Var, T], expr: tac.Expr) -> T:
        def eval(expr: tac.Var | tac.Predefined) -> T:
            return self.transformer_expr(values, expr)

        match expr:
            case tac.Var():
                res = self.lattice.var(values[expr])
                # assert not self.lattice.is_bottom(res)
                return res
            case tac.Attribute():
                val = eval(expr.var)
                return self.lattice.attribute(val, expr.field)
            case tac.Call(function=function, args=args):
                func = eval(function)
                arg_values: list[T] = [eval(arg) for arg in args]
                return self.lattice.call(function=func, args=arg_values)
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
                if isinstance(expr.modname, tac.Attribute):
                    val = eval(expr.modname.var)
                    return self.lattice.attribute(val, expr.modname.field)
                else:
                    return self.lattice.imported(expr.modname.name)
            case tac.MakeFunction():
                return self.lattice.top()
            case _:
                assert False, f"unexpected expr of type {type(expr)}: {expr}"

    def transformer_signature(
        self, value: T, signature: tac.Signature
    ) -> Map[tac.Var, T]:
        match signature:
            case tuple() as signature:  # type: ignore
                assert all(isinstance(v, tac.Var) for v in signature)
                return self.make_map(
                    {
                        var: self.lattice.subscr(value, self.lattice.const(i))
                        for i, var in enumerate(signature)
                    }
                )
            case tac.Var():
                return self.make_map({signature: value})
            case tac.Attribute():
                return self.make_map()
            case tac.Subscript():
                return self.make_map()
            case _:
                assert False, f"unexpected signature {signature}"

    def forward_transfer(
        self, values: VarMapDomain[T], ins: tac.Tac
    ) -> VarMapDomain[T]:
        if isinstance(values, Bottom):
            return BOTTOM
        if isinstance(ins, tac.For):
            ins = ins.as_call()
        to_update = self.make_map()

        match ins:
            case tac.Assign():
                assigned = self.transformer_expr(values, ins.expr)
                to_update = self.transformer_signature(assigned, ins.lhs)
            case tac.Return():
                assigned = self.transformer_expr(values, ins.value)
                to_update = self.make_map({tac.Var("return"): assigned})
        # assert not any(self.lattice.is_bottom(v) for v in to_update.values())
        return to_update

    def transfer(
        self, values: VarMapDomain[T], ins: tac.Tac, location: Location
    ) -> VarMapDomain[T]:
        if isinstance(values, Bottom):
            return BOTTOM
        values = deepcopy(values)
        try:
            # print(f'values: {values}')
            # print(f'{location}: {ins}')
            if location == (215, 0):
                pass
            to_update = self.forward_transfer(values, ins)
            # print(f'updated: {to_update}')
            # print()
        except Exception as e:
            e.add_note(f"while processing {ins} at {location}")
            e.add_note(f"values: {values}")
            raise
        for var in tac.gens(ins):
            if var in values:
                del values[var]
        if isinstance(to_update, Map):
            values.update(to_update)

        here = self.liveness[location]
        if not isinstance(here, Bottom):
            for var in set(values.keys()):
                if var.is_stackvar:
                    if here[var] == BOTTOM:
                        del values[var]
        return self.normalize(values)
