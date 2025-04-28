from __future__ import annotations

import typing
from copy import deepcopy

from pythia.domains import Top, TOP, Bottom


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
        items = ", ".join(sorted([repr(x) for x in self._set]))
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

    def __sub__(self: Set[T], other: Set[T]) -> Set[T]:
        return Set(self._set - other._set)

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


type SetDomain[K] = Set[K] | Top | Bottom


class Map[K, T]:
    # Essentially a defaultdict, but a defaultdict makes values appear out of nowhere
    _map: dict[K, T]
    default: typing.Callable[[], T]

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
        items = ", ".join(sorted([f"{k}={v}" for k, v in self._map.items()]))
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
