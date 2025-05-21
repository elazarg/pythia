from pythia import type_system as ts
from pythia.type_system import INT, FLOAT, STR, LIST, SET, TUPLE, BOTTOM, TOP, ANY

# Define common type variables for testing
T = ts.TypeVar("T")
T1 = ts.TypeVar("T1")
T2 = ts.TypeVar("T2")
N = ts.TypeVar("N")
Args = ts.TypeVar("Args", is_args=True)


def test_unpack_star():
    # Test with no Star items
    items = (INT, FLOAT, STR)
    assert ts.unpack_star(items) == items

    # Test with Star items
    star_items = ts.Star((INT, FLOAT))
    items = (star_items, STR)
    assert ts.unpack_star(items) == (INT, FLOAT, STR)

    # Test with multiple Star items
    star_items1 = ts.Star((INT, FLOAT))
    star_items2 = ts.Star((STR, TUPLE))
    items = (star_items1, LIST, star_items2)
    assert ts.unpack_star(items) == (INT, FLOAT, LIST, STR, TUPLE)


def test_bind_typevars():
    # Test with empty context
    assert ts.bind_typevars(INT, {}) == INT

    # Test with Module and Ref
    module = ts.Module("test", ts.typed_dict([]))
    ref = ts.Ref("test.module")
    assert ts.bind_typevars(module, {T: INT}) == module
    assert ts.bind_typevars(ref, {T: INT}) == ref

    # Test with TypeVar
    assert ts.bind_typevars(T, {T: INT}) == INT
    assert ts.bind_typevars(T, {T1: INT}) == T  # T not in context

    # Test with Star
    star = ts.Star((T, FLOAT))
    bound_star = ts.Star((INT, FLOAT))
    assert ts.bind_typevars(star, {T: INT}) == bound_star

    # Test with Literal (non-tuple value)
    literal = ts.literal(42)
    assert ts.bind_typevars(literal, {T: INT}) == literal

    # Test with Literal (tuple value)
    tuple_literal = ts.literal((T, FLOAT))
    bound_tuple_literal = ts.literal((INT, FLOAT))
    assert ts.bind_typevars(tuple_literal, {T: INT}) == bound_tuple_literal

    # Test with TypedDict
    row = ts.make_row(0, None, T)
    typed_dict = ts.typed_dict([row])
    bound_row = ts.make_row(0, None, INT)
    bound_typed_dict = ts.typed_dict([bound_row])
    assert ts.bind_typevars(typed_dict, {T: INT}) == bound_typed_dict

    # Test with Overloaded
    f1 = ts.FunctionType(
        params=ts.typed_dict([]),
        return_type=T,
        side_effect=ts.SideEffect(False),
        is_property=False,
        type_params=(),
    )
    f2 = ts.FunctionType(
        params=ts.typed_dict([]),
        return_type=FLOAT,
        side_effect=ts.SideEffect(False),
        is_property=False,
        type_params=(),
    )
    overloaded = ts.overload([f1, f2])
    bound_f1 = ts.FunctionType(
        params=ts.typed_dict([]),
        return_type=INT,
        side_effect=ts.SideEffect(False),
        is_property=False,
        type_params=(),
    )
    bound_overloaded = ts.overload([bound_f1, f2])
    assert ts.bind_typevars(overloaded, {T: INT}) == bound_overloaded

    # Test with FunctionType (no type params in context)
    f = ts.FunctionType(ts.typed_dict([]), T, ts.SideEffect(False), False, (T,))
    assert ts.bind_typevars(f, {T: INT}) == f  # T is in f.type_params, so not bound

    # Test with FunctionType (with type params not in function's type_params)
    f = ts.FunctionType(ts.typed_dict([]), T1, ts.SideEffect(False), False, (T1,))
    assert ts.bind_typevars(f, {T2: INT}) == f  # T2 not in f.type_params

    # Test with Union
    union = ts.union([T, FLOAT])
    bound_union = ts.join_all([INT, FLOAT])
    assert ts.bind_typevars(union, {T: INT}) == bound_union

    # Test with Row
    row = ts.make_row(0, None, T)
    bound_row = ts.make_row(0, None, INT)
    assert ts.bind_typevars(row, {T: INT}) == bound_row

    # Test with SideEffect (no update)
    side_effect = ts.SideEffect(False)
    assert ts.bind_typevars(side_effect, {T: INT}) == side_effect

    # Test with SideEffect (with update)
    side_effect = ts.SideEffect(False, update=(T, ()))
    bound_side_effect = ts.SideEffect(False, update=(INT, ()))
    assert ts.bind_typevars(side_effect, {T: INT}) == bound_side_effect

    # Test with Class
    class_dict = ts.typed_dict([ts.make_row(0, "attr", T)])
    klass = ts.Class("TestClass", class_dict, (), False, ())
    bound_class_dict = ts.typed_dict([ts.make_row(0, "attr", INT)])
    bound_klass = ts.Class("TestClass", bound_class_dict, (), False, ())
    assert ts.bind_typevars(klass, {T: INT}) == bound_klass

    # Test with Class (type_params in context)
    klass = ts.Class("TestClass", class_dict, (), False, (T,))
    assert ts.bind_typevars(klass, {T: INT}) == klass  # T is in klass.type_params

    # Test with Instantiation
    inst = ts.Instantiation(LIST, (T,))
    bound_inst = ts.Instantiation(LIST, (INT,))
    assert ts.bind_typevars(inst, {T: INT}) == bound_inst

    # Test with Access
    access = ts.Access(ts.Star((INT, FLOAT, STR)), ts.literal(1))
    # When binding T to INT in an Access object, the result should be the same Access object
    # since neither the Star nor the Literal contain T
    assert ts.bind_typevars(access, {T: INT}) == FLOAT

    # Test with Access (with Star and Literal int)
    access = ts.Access(ts.Star((T, FLOAT, STR)), ts.literal(0))
    assert ts.bind_typevars(access, {T: INT}) == INT


def test_typed_dict():
    # Test with empty rows
    assert ts.typed_dict([]) == TOP

    # Test with single row
    row = ts.make_row(0, None, INT)
    assert ts.typed_dict([row]) == ts.TypedDict(frozenset([row]))

    # Test with multiple rows
    row1 = ts.make_row(0, None, INT)
    row2 = ts.make_row(1, None, FLOAT)
    assert ts.typed_dict([row1, row2]) == ts.TypedDict(frozenset([row1, row2]))


def test_overload():
    # Test with empty list
    assert ts.overload([]) == ts.Overloaded(())

    # Test with single FunctionType
    f = ts.FunctionType(ts.typed_dict([]), INT, ts.SideEffect(False), False, ())
    assert ts.overload([f]) == ts.Overloaded((f,))

    # Test with multiple FunctionTypes
    f1 = ts.FunctionType(ts.typed_dict([]), INT, ts.SideEffect(False), False, ())
    f2 = ts.FunctionType(ts.typed_dict([]), FLOAT, ts.SideEffect(False), False, ())
    assert ts.overload([f1, f2]) == ts.Overloaded((f1, f2))

    # Test with Overloaded
    overloaded = ts.overload([f1])
    assert ts.overload([overloaded, f2]) == ts.Overloaded((f1, f2))

    # Test with multiple Overloaded
    overloaded1 = ts.overload([f1])
    overloaded2 = ts.overload([f2])
    assert ts.overload([overloaded1, overloaded2]) == ts.Overloaded((f1, f2))


def test_union():
    # Test with empty list
    assert ts.union([]) == ts.Union(frozenset())

    # Test with single type
    assert ts.union([INT]) == INT

    # Test with multiple types
    union = ts.union([INT, FLOAT])
    assert isinstance(union, ts.Union)
    assert union.items == frozenset([INT, FLOAT])

    # Test with duplicate types
    assert ts.union([INT, INT]) == INT

    # Test with squeeze=False
    union = ts.union([INT], squeeze=False)
    assert isinstance(union, ts.Union)
    assert union.items == frozenset([INT])


def test_meet():
    # Test with identical types
    assert ts.meet(INT, INT) == INT

    # Test with TOP
    assert ts.meet(TOP, INT) == INT
    assert ts.meet(INT, TOP) == INT

    # Test with BOTTOM
    assert ts.meet(BOTTOM, INT) == BOTTOM
    assert ts.meet(INT, BOTTOM) == BOTTOM

    # Test with Literal and Ref
    literal = ts.literal(42)
    assert ts.meet(literal, INT) == literal
    assert ts.meet(INT, literal) == literal

    # Test with two Literals of same ref
    literal1 = ts.literal(42)
    literal2 = ts.literal(43)
    assert ts.meet(literal1, literal2) == INT

    # Test with Ref and Instantiation
    inst = ts.Instantiation(LIST, (INT,))
    assert ts.meet(LIST, inst) == inst
    assert ts.meet(inst, LIST) == inst

    # Test with TypedDict
    dict1 = ts.typed_dict([ts.make_row(0, None, INT)])
    dict2 = ts.typed_dict([ts.make_row(1, None, FLOAT)])
    combined = ts.typed_dict([ts.make_row(0, None, INT), ts.make_row(1, None, FLOAT)])
    assert ts.meet(dict1, dict2) == combined

    # Test with Union
    union = ts.union([INT, FLOAT])
    assert ts.meet(union, INT) == INT

    # Test with Row
    row1 = ts.make_row(0, None, INT)
    row2 = ts.make_row(0, None, FLOAT)
    assert ts.meet(row1, row2) == ts.make_row(0, None, ts.meet(INT, FLOAT))

    # Test with different index Rows
    row1 = ts.make_row(0, None, INT)
    row2 = ts.make_row(1, None, FLOAT)
    assert ts.meet(row1, row2) == TOP


def test_join_all():
    # Test with empty list
    assert ts.join_all([]) == BOTTOM

    # Test with single type
    assert ts.join_all([INT]) == INT

    # Test with multiple types
    assert ts.join_all([INT, FLOAT]) == ts.join(INT, FLOAT)

    # Test with Star
    star = ts.Star((INT, FLOAT))
    assert ts.join_all([star]) == ts.join_all([INT, FLOAT])


def test_meet_all():
    # Test with empty list
    assert ts.meet_all([]) == TOP

    # Test with single type
    assert ts.meet_all([INT]) == INT

    # Test with multiple types
    assert ts.meet_all([INT, FLOAT]) == ts.meet(INT, FLOAT)

    # Test with Star
    star = ts.Star((INT, FLOAT))
    assert ts.meet_all([star]) == ts.meet_all([INT, FLOAT])


def test_is_subtype():
    # Test with ANY
    assert ts.is_subtype(INT, ANY) == True
    assert ts.is_subtype(ANY, INT) == True

    # Test with same type
    assert ts.is_subtype(INT, INT) == True

    # Test with different types
    assert ts.is_subtype(ts.literal(42), INT) == True
    assert ts.is_subtype(INT, ts.literal(42)) == False

    # Test with Union
    union = ts.union([INT, FLOAT])
    assert ts.is_subtype(INT, union) == True
    assert ts.is_subtype(union, INT) == False


def test_subtract_indices():
    # Test with empty bound_params
    unbound = ts.typed_dict([ts.make_row(0, None, INT), ts.make_row(1, None, FLOAT)])
    bound = ts.typed_dict([])
    assert ts.subtract_indices(unbound, bound) == unbound

    # Test with bound_params
    unbound = ts.typed_dict([ts.make_row(1, None, FLOAT), ts.make_row(2, None, STR)])
    bound = ts.typed_dict([ts.make_row(0, None, INT)])
    expected = ts.typed_dict([ts.make_row(0, None, FLOAT), ts.make_row(1, None, STR)])
    assert ts.subtract_indices(unbound, bound) == expected

    # Test with multiple bound_params
    unbound = ts.typed_dict([ts.make_row(2, None, STR), ts.make_row(3, None, LIST)])
    bound = ts.typed_dict([ts.make_row(0, None, INT), ts.make_row(1, None, FLOAT)])
    expected = ts.typed_dict([ts.make_row(0, None, STR), ts.make_row(1, None, LIST)])
    assert ts.subtract_indices(unbound, bound) == expected

    # Test with named indices
    unbound = ts.typed_dict([ts.make_row(0, "x", INT), ts.make_row(1, "y", FLOAT)])
    bound = ts.typed_dict([])
    assert ts.subtract_indices(unbound, bound) == unbound


def test_access():
    # Test with Class and string literal
    class_dict = ts.typed_dict([ts.make_row(0, "method", INT)])
    klass = ts.Class("TestClass", class_dict, (), False, ())
    assert ts.access(klass, ts.literal("method")) == INT

    # Test with Module and string literal
    module_dict = ts.typed_dict([ts.make_row(0, "func", FLOAT)])
    module = ts.Module("test", module_dict)
    assert ts.access(module, ts.literal("func")) == FLOAT

    # Test with non-existent attribute in Module
    assert ts.access(module, ts.literal("nonexistent")) == BOTTOM

    # Test with TypedDict and matching Row
    # TODO: This is poorly defined: multiple functions are overload - all valid; multiple nonfunctions are join - the choice is unknown
    # dict1 = ts.typed_dict([ts.make_row(0, None, INT), ts.make_row(1, None, FLOAT)])
    # # When accessing a TypedDict with a literal index, the result should be the type at that index
    # assert ts.access(dict1, ts.literal(0)) == INT
    # assert ts.access(dict1, ts.literal(1)) == FLOAT

    # Test with Union
    union = ts.union([INT, FLOAT])
    # This should return the join of accessing each item in the union
    # Since neither INT nor FLOAT have attributes, this should return an empty overloaded
    assert isinstance(ts.access(union, ts.literal("attr")), ts.Overloaded)

    # Test with Literal
    literal = ts.literal(42)
    # This should access the attribute on the Ref of the literal
    assert ts.access(literal, ts.literal("__add__")) == ts.access(
        INT, ts.literal("__add__")
    )


def test_split_by_args():
    # Test with TOP and BOTTOM
    assert ts.split_by_args(TOP, ts.typed_dict([])) == TOP
    assert ts.split_by_args(BOTTOM, ts.typed_dict([])) == BOTTOM

    # Test with FunctionType
    f = ts.FunctionType(
        ts.typed_dict([ts.make_row(0, None, INT)]),
        FLOAT,
        ts.SideEffect(False),
        False,
        (),
    )
    args = ts.typed_dict([ts.make_row(0, None, INT)])
    result = ts.split_by_args(f, args)
    assert isinstance(result, ts.Overloaded)

    # Test with Overloaded
    f1 = ts.FunctionType(
        ts.typed_dict([ts.make_row(0, None, INT)]),
        FLOAT,
        ts.SideEffect(False),
        False,
        (),
    )
    f2 = ts.FunctionType(
        ts.typed_dict([ts.make_row(0, None, STR)]),
        LIST,
        ts.SideEffect(False),
        False,
        (),
    )
    overloaded = ts.overload([f1, f2])

    # With matching args for f1
    args = ts.typed_dict([ts.make_row(0, None, INT)])
    result = ts.split_by_args(overloaded, args)
    assert isinstance(result, ts.Overloaded)

    # With matching args for f2
    args = ts.typed_dict([ts.make_row(0, None, STR)])
    result = ts.split_by_args(overloaded, args)
    assert isinstance(result, ts.Overloaded)

    # Test with Union
    union = ts.union([f1, f2])
    args = ts.typed_dict([ts.make_row(0, None, INT)])
    result = ts.split_by_args(union, args)
    assert isinstance(result, ts.Union)


def test_partial():
    # Create a function that takes an INT and returns a FLOAT
    f = ts.FunctionType(
        ts.typed_dict([ts.make_row(0, None, INT)]),
        FLOAT,
        ts.SideEffect(False),
        False,
        (),
    )

    # Test with matching args
    args = ts.typed_dict([ts.make_row(0, None, INT)])
    result = ts.partial(f, args, only_callable_empty=False)
    # partial should return a partially applied function, not the return value
    assert isinstance(result, ts.Overloaded)
    # We can use get_return to get the return value
    assert ts.get_return(result) == FLOAT

    # Test with non-matching args
    args = ts.typed_dict([ts.make_row(0, None, STR)])
    result = ts.partial(f, args, only_callable_empty=False)
    # When args don't match, partial should return BOTTOM
    assert result == BOTTOM

    # Test with Overloaded
    f1 = ts.FunctionType(
        ts.typed_dict([ts.make_row(0, None, INT)]),
        FLOAT,
        ts.SideEffect(False),
        False,
        (),
    )
    f2 = ts.FunctionType(
        ts.typed_dict([ts.make_row(0, None, STR)]),
        LIST,
        ts.SideEffect(False),
        False,
        (),
    )
    overloaded = ts.overload([f1, f2])

    # With matching args for f1
    args = ts.typed_dict([ts.make_row(0, None, INT)])
    result = ts.partial(overloaded, args, only_callable_empty=False)
    # partial should return a partially applied function, not the return value
    assert isinstance(result, ts.Overloaded)
    # We can use get_return to get the return value
    assert ts.get_return(result) == FLOAT

    # With matching args for f2
    args = ts.typed_dict([ts.make_row(0, None, STR)])
    result = ts.partial(overloaded, args, only_callable_empty=False)
    # partial should return a partially applied function, not the return value
    assert isinstance(result, ts.Overloaded)
    # We can use get_return to get the return value
    assert ts.get_return(result) == LIST

    # Test with only_callable_empty=True
    # Create a function that takes no arguments
    f_empty = ts.FunctionType(ts.typed_dict([]), FLOAT, ts.SideEffect(False), False, ())
    args = ts.typed_dict([])
    result = ts.partial(f_empty, args, only_callable_empty=True)
    # partial should return a partially applied function, not the return value
    assert isinstance(result, ts.Overloaded)
    # We can use get_return to get the return value
    assert ts.get_return(result) == FLOAT
