from spyte.dom_concrete import Set, Map, TOP, Bottom
from spyte.domains import Top


def test_set_init():
    # Test empty set initialization
    s = Set()
    assert str(s) == "Set()"

    # Test initialization with values
    s = Set([1, 2, 3])
    assert str(s) == "Set(1, 2, 3)"

    # Test initialization with TOP
    s = Set(TOP)
    assert str(s) == "Set(TOP)"


def test_set_top():
    s = Set.top()
    assert str(s) == "Set(TOP)"
    assert isinstance(s._set, Top)


def test_set_meet():
    # Test meet with regular sets
    s1 = Set([1, 2, 3])
    s2 = Set([2, 3, 4])
    result = s1.meet(s2)
    assert str(result) == "Set(2, 3)"

    # Test meet with TOP
    s1 = Set([1, 2, 3])
    s2 = Set.top()
    assert s1.meet(s2) == s1
    assert s2.meet(s1) == s1


def test_set_join():
    # Test join with regular sets
    s1 = Set([1, 2, 3])
    s2 = Set([2, 3, 4])
    result = s1.join(s2)
    assert str(result) == "Set(1, 2, 3, 4)"

    # Test join with TOP
    s1 = Set([1, 2, 3])
    s2 = Set.top()
    assert s1.join(s2) == Set.top()
    assert s2.join(s1) == Set.top()


def test_set_contains():
    s = Set([1, 2, 3])
    assert 1 in s
    assert 4 not in s

    s = Set.top()
    assert 1 in s
    assert 100 in s  # Everything is in TOP


def test_set_bool():
    s = Set()
    assert not bool(s)

    s = Set([1])
    assert bool(s)

    s = Set.top()
    assert bool(s)


def test_set_subset():
    s1 = Set([1, 2])
    s2 = Set([1, 2, 3])
    s3 = Set.top()

    assert s1.is_subset(s2)
    assert not s2.is_subset(s1)
    assert s1.is_subset(s3)
    assert not s3.is_subset(s1)

    # Test operator versions
    assert s1 <= s2
    assert not s2 <= s1
    assert s1 <= s3
    assert not s3 <= s1


def test_set_operators():
    s1 = Set([1, 2, 3])
    s2 = Set([2, 3, 4])

    # Test union operator
    assert s1 | s2 == Set([1, 2, 3, 4])

    # Test intersection operator
    assert s1 & s2 == Set([2, 3])

    # Test difference operator
    assert s1 - s2 == Set([1])
    assert s2 - s1 == Set([4])


def test_set_squeeze():
    # Test squeezing a singleton set
    s = Set([42])
    assert Set.squeeze(s) == 42

    # Test squeezing a non-singleton set
    s = Set([1, 2])
    assert Set.squeeze(s) == s

    # Test squeezing a non-set
    assert Set.squeeze(42) == 42

    # Test squeezing TOP
    s = Set.top()
    assert Set.squeeze(s) == s


def test_set_union_all():
    s1 = Set([1, 2])
    s2 = Set([2, 3])
    s3 = Set([3, 4])

    result = Set.union_all([s1, s2, s3])
    assert str(result) == "Set(1, 2, 3, 4)"

    # Test with TOP
    s4 = Set.top()
    result = Set.union_all([s1, s2, s4])
    assert result == Set.top()


def test_set_singleton():
    s = Set.singleton(42)
    assert str(s) == "Set(42)"


def test_set_as_set():
    s = Set([1, 2, 3])
    fs = s.as_set()
    assert isinstance(fs, frozenset)
    assert fs == frozenset([1, 2, 3])


def test_map_init():
    # Test empty map initialization
    m = Map(lambda: 0)
    assert str(m) == "Map()"

    # Test initialization with values
    m = Map(lambda: 0, {"a": 1, "b": 2})
    assert "a=1" in str(m)
    assert "b=2" in str(m)


def test_map_getitem():
    m = Map(lambda: 0, {"a": 1, "b": 2})
    assert m["a"] == 1
    assert m["b"] == 2
    assert m["c"] == 0  # Default value


def test_map_setitem():
    m = Map(lambda: 0)
    m["a"] = 1
    assert m["a"] == 1

    # Setting to default value should remove the key
    m["a"] = 0
    assert "a" not in m


def test_map_update():
    m1 = Map(lambda: 0, {"a": 1, "b": 2})
    m2 = Map(lambda: 0, {"b": 3, "c": 4})

    m1.update(m2)
    assert m1["a"] == 1
    assert m1["b"] == 3
    assert m1["c"] == 4


def test_map_iter():
    m = Map(lambda: 0, {"a": 1, "b": 2, "c": 3})
    keys = list(m)
    assert set(keys) == {"a", "b", "c"}


def test_map_contains():
    m = Map(lambda: 0, {"a": 1, "b": 2})
    assert "a" in m
    assert "c" not in m


def test_map_bool():
    m = Map(lambda: 0)
    assert not bool(m)

    m["a"] = 1
    assert bool(m)


def test_map_len():
    m = Map(lambda: 0)
    assert len(m) == 0

    m["a"] = 1
    m["b"] = 2
    assert len(m) == 2


def test_map_delitem():
    m = Map(lambda: 0, {"a": 1, "b": 2})
    del m["a"]
    assert "a" not in m
    assert "b" in m


def test_map_items_values_keys():
    m = Map(lambda: 0, {"a": 1, "b": 2})

    assert set(m.keys()) == {"a", "b"}
    assert set(m.values()) == {1, 2}
    assert set(m.items()) == {("a", 1), ("b", 2)}


def test_map_deepcopy():
    m1 = Map(lambda: 0, {"a": 1, "b": 2})
    m2 = m1.__deepcopy__()

    assert m1 == m2
    assert m1 is not m2

    # Modify m2 and check that m1 is unchanged
    m2["c"] = 3
    assert "c" not in m1


def test_map_keep_keys():
    m = Map(lambda: 0, {"a": 1, "b": 2, "c": 3})
    m.keep_keys(["a", "c"])

    assert "a" in m
    assert "b" not in m
    assert "c" in m


def test_map_join():
    class Value:
        def __init__(self, val):
            self.val = val

        def join(self, other):
            return Value(max(self.val, other.val))

        def __eq__(self, other):
            return isinstance(other, Value) and self.val == other.val

    m1 = Map(lambda: Value(0), {"a": Value(1), "b": Value(2)})
    m2 = Map(lambda: Value(0), {"b": Value(3), "c": Value(4)})

    result = m1.join(m2)
    assert result["a"].val == 1
    assert result["b"].val == 3
    assert result["c"].val == 4


def test_map_is_less_than():
    class Value:
        def __init__(self, val):
            self.val = val

        def is_less_than(self, other):
            return self.val < other.val

    m1 = Map(lambda: Value(0), {"a": Value(1), "b": Value(2)})
    m2 = Map(lambda: Value(0), {"a": Value(2), "b": Value(3)})

    assert m1.is_less_than(m2)
    assert not m2.is_less_than(m1)
