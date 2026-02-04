from dataclasses import replace

from pythia import type_system as ts
from pythia.type_system import INT, FLOAT, STR, LIST, SET, TUPLE, BOTTOM, TOP, ANY

ARRAY = ts.Ref("numpy.ndarray")

T = ts.TypeVar("T")
T1 = ts.TypeVar("T1")
T2 = ts.TypeVar("T2")
N = ts.TypeVar("N")
Args = ts.TypeVar("Args", is_args=True)

FIRST = ts.literal(0)
SECOND = ts.literal(1)


def typeof(x):
    return ts.Instantiation(ts.Ref("builtins.type"), (x,))


def binop(left: ts.TypeExpr, right: ts.TypeExpr, op: str) -> ts.TypeExpr:
    return ts.get_return(ts.partial_binop(left, right, op))


def make_function(
    return_type: ts.TypeExpr,
    params: ts.TypedDict,
    type_params=(),
    update=(None, ()),
    new=None,
) -> ts.FunctionType:
    if new is None:
        new = not ts.is_immutable(return_type)
    return ts.FunctionType(
        params,
        return_type,
        is_property=False,
        type_params=type_params,
        side_effect=ts.SideEffect(new=new, update=update),
    )


def make_rows(*types) -> ts.TypedDict:
    return ts.typed_dict([ts.make_row(index, None, t) for index, t in enumerate(types)])


def test_join():
    t1 = INT
    t2 = FLOAT
    assert ts.join(t1, t2) == ts.union([t1, t2])

    t1 = INT
    t2 = ts.literal(0)
    assert ts.join(t1, t2) == t1

    t1 = INT
    t2 = ts.literal(1)
    assert ts.join(t1, t2) == t1


def test_join_typevar_and_int():
    t = ts.TypeVar("T")

    t1 = t
    t2 = INT
    expected = ts.union([t, t2])
    assert ts.join(t1, t2) == expected

    t1 = ts.Instantiation(LIST, (t,))
    t2 = ts.Instantiation(LIST, (INT,))
    expected = ts.Instantiation(LIST, (ts.union([t, INT]),))
    assert ts.join(t1, t2) == expected

    # t1 = make_function(t, make_rows(t), type_params=(t,), new=False)
    # t2 = make_function(INT, make_rows(t), type_params=(t,), new=False)
    # expected = make_function(
    #     ts.union([t, INT]), make_rows(t), type_params=(t,), new=False
    # )
    # assert ts.join(t1, t2) == expected


def test_join_literals():
    t1 = ts.literal(0)
    t2 = ts.literal(1)
    joined = INT
    assert ts.join(t1, t2) == joined


def test_overload():
    f1 = make_function(STR, make_rows(INT))
    f2 = make_function(FLOAT, make_rows(FLOAT))
    arg = INT
    args = make_rows(arg)
    overload = ts.overload([f1, f2])
    assert ts.call(overload, args) == STR

    f1 = make_function(STR, make_rows(INT))
    f2 = make_function(FLOAT, make_rows(FLOAT))
    arg = ts.literal(0)
    args = make_rows(arg)
    overload = ts.overload([f1, f2])
    assert ts.call(overload, args) == STR


def test_unification():
    assert ts.unify(
        type_params=(T,), params=make_rows(T), args=make_rows(INT)
    ).bound_typevars == {T: INT}

    assert ts.unify(
        type_params=(T1, T2), params=make_rows(T1, T2), args=make_rows(INT, FLOAT)
    ).bound_typevars == {T1: INT, T2: FLOAT}


def test_unification_args():
    args = make_rows(INT, FLOAT)
    assert ts.unify(
        type_params=(Args,), params=make_rows(Args), args=args
    ).bound_typevars == {Args: ts.Star((INT, FLOAT))}

    args = make_rows(FLOAT, INT, FLOAT)
    assert ts.unify(
        type_params=(T, Args), params=make_rows(T, Args, T), args=args
    ).bound_typevars == {T: FLOAT, Args: ts.Star((INT,))}


def test_unification_protocol():
    param = ts.Instantiation(ts.Ref("typing.Iterable"), (T,))
    arg = ts.Instantiation(ts.Ref("builtins.set"), (INT,))
    assert ts.unify_argument(type_params=(T,), param=param, arg=arg) == {T: INT}


def test_function_call():
    f = make_function(INT, make_rows())
    arg = make_rows()
    assert ts.call(f, arg) == INT

    f = make_function(INT, make_rows(INT, FLOAT))
    arg = make_rows(INT, FLOAT)
    assert ts.call(f, arg) == INT

    f = make_function(INT, make_rows(INT, FLOAT))
    arg = make_rows(INT, INT)
    assert ts.call(f, arg) == ts.BOTTOM

    f = make_function(INT, make_rows())
    arg = ts.typed_dict([])
    assert ts.call(f, arg) == INT

    f = make_function(INT, make_rows(ts.TOP))
    arg = make_rows(FLOAT)
    assert ts.call(f, arg) == INT


def test_function_call_builtins():
    f = ts.resolve_static_ref(ts.Ref("builtins.max"))

    arg = ts.Instantiation(ts.Ref("typing.Iterable"), (INT,))
    assert ts.call(f, make_rows(arg)) == INT

    arg = ts.Instantiation(ts.Ref("builtins.set"), (INT,))
    assert ts.call(f, make_rows(arg)) == INT


def test_call_union():
    f = make_function(INT, make_rows(ts.union([INT, FLOAT])))
    arg = make_rows(INT)
    assert ts.call(f, arg) == INT


def test_function_call_generic_project():
    f = make_function(T, make_rows(T), [T])
    arg = make_rows(INT)
    assert ts.call(f, arg) == INT

    f = make_function(T1, make_rows(T1, T2), [T1, T2])
    arg = make_rows(INT, FLOAT)
    assert ts.call(f, arg) == INT

    f = make_function(T2, make_rows(T1, T2), [T1, T2])
    arg = make_rows(INT, FLOAT)
    assert ts.call(f, arg) == FLOAT

    f = make_function(T, make_rows(T, T), [T])
    arg = make_rows(INT, FLOAT)
    assert ts.call(f, arg) == ts.join(INT, FLOAT)


def test_bind_self_tuple():
    tuple_star = ts.Star((INT, FLOAT))
    tuple_named = ts.Instantiation(TUPLE, (INT, FLOAT))
    tuple_param = ts.Instantiation(TUPLE, (Args,))
    assert ts.unify_argument((Args,), tuple_param, tuple_named) == {Args: tuple_star}
    f = ts.FunctionType(
        params=make_rows(tuple_param, N),
        return_type=ts.Access(Args, N),
        is_property=False,
        side_effect=ts.SideEffect(new=False, bound_method=True),
        type_params=(N, Args),
    )
    g = replace(
        f, params=make_rows(N), return_type=ts.Access(tuple_star, N), type_params=(N,)
    )
    assert ts.bind_self(ts.overload([f]), tuple_named) == ts.overload([g])


def test_tuple():
    f = make_function(T, make_rows(T, T), [T])
    arg = make_rows(INT, FLOAT)
    assert ts.call(f, arg) == ts.join(INT, FLOAT)

    tuple_named = ts.Instantiation(TUPLE, (INT, FLOAT))

    tuple_star = ts.Star((INT, FLOAT))
    tuple_structure = ts.literal((INT, FLOAT))

    g = ts.subscr(tuple_named, ts.literal("__getitem__"))
    f = ts.FunctionType(
        params=ts.typed_dict([ts.make_row(0, "item", N)]),
        return_type=ts.Access(tuple_star, N),
        is_property=False,
        side_effect=ts.SideEffect(new=False, bound_method=True),
        type_params=(N,),
    )
    assert g == ts.overload([f])
    x = ts.call(g, make_rows(FIRST))
    assert x == INT

    x = ts.subscr_get_property(tuple_named, FIRST)
    assert x == INT

    x = ts.subscr_get_property(tuple_named, SECOND)
    assert x == FLOAT

    x = ts.call(g, make_rows(SECOND))
    assert x == FLOAT

    left = ts.subscr_get_property(tuple_structure, FIRST)
    assert left == INT

    right = ts.subscr_get_property(tuple_structure, SECOND)
    assert right == FLOAT


def test_list_add():
    t_int = ts.Instantiation(LIST, (INT,))
    gt = ts.subscr_get_property(t_int, ts.literal("__add__"))
    x = ts.call(gt, make_rows(t_int))
    assert x == t_int


def test_list_getitem():
    t_int = ts.Instantiation(LIST, (INT,))
    gt = ts.subscr_get_property(t_int, ts.literal("__getitem__"))
    x = ts.call(gt, make_rows(ts.literal(0)))
    assert x == INT


def test_list_setitem():
    t_bottom = ts.Instantiation(LIST, (ts.BOTTOM,))

    st = ts.subscr_get_property(t_bottom, ts.literal("__setitem__"))
    applied = ts.partial(
        st, make_rows(ts.literal(0), ts.literal(0)), only_callable_empty=True
    )
    t_literal = ts.get_side_effect(applied).update[0]
    assert t_literal == ts.Instantiation(LIST, (ts.literal(0),))

    st = ts.subscr_get_property(t_literal, ts.literal("__setitem__"))
    applied = ts.partial(
        st, make_rows(ts.literal(1), ts.literal(1)), only_callable_empty=True
    )
    t_two_literals = ts.get_side_effect(applied).update[0]
    assert t_two_literals == ts.Instantiation(LIST, (INT,))


def test_list_join():
    non_empty1 = ts.literal([INT])
    non_empty2 = ts.literal([INT])
    assert ts.join(non_empty1, non_empty2) == non_empty1

    empty = ts.literal([])
    non_empty = ts.literal([INT])
    assert ts.join(empty, non_empty) == ts.Instantiation(LIST, (INT,))


def test_getitem_list():
    t = ts.Instantiation(LIST, (INT,))
    x = ts.subscr_get_property(t, ts.literal(0))
    assert x == INT


def test_getitem_numpy():
    x = ts.subscr_get_property(ARRAY, ts.literal(None))
    assert x == ts.BOTTOM

    x = ts.subscr_get_property(ARRAY, ts.literal(0))
    assert x == ARRAY

    x = ts.subscr_get_property(ARRAY, ARRAY)
    assert x == ARRAY

    x = ts.subscr_get_property(ARRAY, ts.Ref("builtins.slice"))
    assert x == ARRAY


def test_operator_numpy():
    x = binop(ARRAY, ARRAY, "+")
    assert x == ARRAY


def test_right_operator_numpy():
    x = binop(FLOAT, ARRAY, "+")
    assert x == ARRAY

    x = binop(FLOAT, ARRAY, "*")
    assert x == ARRAY


def test_set_empty_constructor():
    constructor = ts.make_set_constructor()
    args = make_rows()
    s = ts.call(constructor, args)
    assert s == ts.Instantiation(SET, (ts.BOTTOM,))

    add = ts.subscr_get_property(s, ts.literal("add"))
    x = ts.partial(add, make_rows(INT), only_callable_empty=True)
    assert isinstance(x, ts.Overloaded)
    assert len(x.items) == 1
    x = x.items[0]
    assert x.side_effect.update[0] == ts.Instantiation(SET, (INT,))


def test_set_constructor():
    constructor = ts.make_set_constructor()

    args = make_rows()
    s = ts.call(constructor, args)
    assert s == ts.Instantiation(SET, (ts.BOTTOM,))

    args = make_rows(INT)
    s = ts.call(constructor, args)
    assert s == ts.Instantiation(SET, (INT,))

    args = make_rows(INT, INT)
    s = ts.call(constructor, args)
    assert s == ts.Instantiation(SET, (INT,))


def test_list_constructor():
    constructor = ts.make_list_constructor()
    args = make_rows(INT)
    lst = ts.call(constructor, args)
    # Star is not general, and cannot comfortably join
    # Intersect is incorrect when empty - it is not TOP, but an empty list
    # Solution: Literal(Start((INT,)))
    assert lst == ts.literal([INT])

    args = make_rows()
    lst = ts.call(constructor, args)
    assert lst == ts.literal([])


def test_list_init():
    args = make_rows(ts.Instantiation(ts.Ref("typing.Iterable"), (FLOAT,)))
    lst = ts.call(typeof(LIST), args)
    assert lst == ts.Instantiation(LIST, (FLOAT,))

    tt = ts.Instantiation(LIST, (FLOAT,))
    lst = ts.call(typeof(tt), make_rows())
    assert lst == tt


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
    union = ts.union([INT], should_squeeze=False)
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


def test_side_effect_equality():
    """Verify frozen dataclass equality works correctly for SideEffect."""
    # Same SideEffect should be equal
    s1 = ts.SideEffect(new=True, bound_method=False)
    s2 = ts.SideEffect(new=True, bound_method=False)
    assert s1 == s2

    # Different SideEffects should not be equal
    s3 = ts.SideEffect(new=False, bound_method=False)
    assert s1 != s3

    # Test with update field
    s4 = ts.SideEffect(new=True, update=(INT, (1,)))
    s5 = ts.SideEffect(new=True, update=(INT, (1,)))
    assert s4 == s5

    s6 = ts.SideEffect(new=True, update=(FLOAT, (1,)))
    assert s4 != s6

    # Test with alias field
    s7 = ts.SideEffect(new=True, alias=("_buffer",))
    s8 = ts.SideEffect(new=True, alias=("_buffer",))
    assert s7 == s8

    s9 = ts.SideEffect(new=True, alias=("_other",))
    assert s7 != s9


def test_side_effect_join_basic():
    """Join two SideEffects with different new/bound_method flags."""
    # Test OR semantics for boolean fields
    s1 = ts.SideEffect(new=True, bound_method=False, points_to_args=False)
    s2 = ts.SideEffect(new=False, bound_method=True, points_to_args=True)

    result = ts.join(s1, s2)
    assert isinstance(result, ts.SideEffect)
    assert result.new == True  # True | False = True
    assert result.bound_method == True  # False | True = True
    assert result.points_to_args == True  # False | True = True

    # Test with both False
    s3 = ts.SideEffect(new=False, bound_method=False)
    s4 = ts.SideEffect(new=False, bound_method=False)
    result = ts.join(s3, s4)
    assert result.new == False
    assert result.bound_method == False


def test_side_effect_join_alias():
    """Join with overlapping/non-overlapping alias tuples."""
    # Non-overlapping aliases: union
    s1 = ts.SideEffect(new=False, alias=("_buffer",))
    s2 = ts.SideEffect(new=False, alias=("_data",))
    result = ts.join(s1, s2)
    assert "_buffer" in result.alias
    assert "_data" in result.alias
    assert len(result.alias) == 2

    # Overlapping aliases: no duplicates
    s3 = ts.SideEffect(new=False, alias=("_buffer", "_data"))
    s4 = ts.SideEffect(new=False, alias=("_buffer", "_other"))
    result = ts.join(s3, s4)
    assert result.alias.count("_buffer") == 1
    assert "_data" in result.alias
    assert "_other" in result.alias

    # Empty aliases
    s5 = ts.SideEffect(new=False, alias=())
    s6 = ts.SideEffect(new=False, alias=("_field",))
    result = ts.join(s5, s6)
    assert result.alias == ("_field",)


def test_side_effect_join_update():
    """Join with different update type expressions."""
    # Both have same update args tuple
    s1 = ts.SideEffect(new=False, update=(INT, (1,)))
    s2 = ts.SideEffect(new=False, update=(FLOAT, (1,)))

    result = ts.join(s1, s2)
    assert isinstance(result, ts.SideEffect)
    # Types should be joined
    assert result.update[0] == ts.join(INT, FLOAT)
    # Arg indices should remain the same
    assert result.update[1] == (1,)

    # One has None update
    s3 = ts.SideEffect(new=False, update=(None, ()))
    s4 = ts.SideEffect(new=False, update=(INT, ()))
    result = ts.join(s3, s4)
    # join(None, INT) via join function - None isn't a TypeExpr, should be handled
    assert result.update[1] == ()


def test_side_effect_meet_basic():
    """Test the meet() implementation for SideEffect."""
    # Test AND semantics for boolean fields
    s1 = ts.SideEffect(new=True, bound_method=True, points_to_args=True)
    s2 = ts.SideEffect(new=True, bound_method=False, points_to_args=True)

    result = ts.meet(s1, s2)
    assert isinstance(result, ts.SideEffect)
    assert result.new == True  # True & True = True
    assert result.bound_method == False  # True & False = False
    assert result.points_to_args == True  # True & True = True

    # Test with both False - should stay False
    s3 = ts.SideEffect(new=False, bound_method=False)
    s4 = ts.SideEffect(new=True, bound_method=True)
    result = ts.meet(s3, s4)
    assert result.new == False  # False & True = False
    assert result.bound_method == False  # False & True = False


def test_side_effect_meet_alias():
    """Test meet for alias tuples (should be intersection)."""
    # Overlapping aliases: intersection
    s1 = ts.SideEffect(new=False, alias=("_buffer", "_data"))
    s2 = ts.SideEffect(new=False, alias=("_buffer", "_other"))
    result = ts.meet(s1, s2)
    assert result.alias == ("_buffer",)  # Only the common field

    # Non-overlapping aliases: empty
    s3 = ts.SideEffect(new=False, alias=("_field1",))
    s4 = ts.SideEffect(new=False, alias=("_field2",))
    result = ts.meet(s3, s4)
    assert result.alias == ()

    # One empty: result is empty
    s5 = ts.SideEffect(new=False, alias=())
    s6 = ts.SideEffect(new=False, alias=("_field",))
    result = ts.meet(s5, s6)
    assert result.alias == ()


def test_access_type_operations():
    """Test Access type handling in join, meet, and squeeze operations."""
    # Create Access types with TypeVars (used in generic contexts)
    tv1 = ts.TypeVar("T")
    tv2 = ts.TypeVar("U")
    access1 = ts.Access(tv1, ts.literal(0))
    access2 = ts.Access(tv2, ts.literal(0))

    # Test join of two Access types with same arg
    result = ts.join(access1, access2)
    assert isinstance(result, ts.Access)
    # The items should be joined (results in Union of T and U)
    assert result.arg == ts.literal(0)

    # Test join of Access types with different args
    access3 = ts.Access(tv1, ts.literal(1))
    result = ts.join(access1, access3)
    assert isinstance(result, ts.Access)

    # Test meet of two Access types with compatible inner types
    # Use types that meet() can handle (like INT and INT)
    access_int1 = ts.Access(INT, ts.literal(0))
    access_int2 = ts.Access(INT, ts.literal(0))
    result = ts.meet(access_int1, access_int2)
    assert isinstance(result, ts.Access)
    assert result.items == INT
    assert result.arg == ts.literal(0)

    # Test squeeze of Access type
    inner_union = ts.union([INT, FLOAT])
    access_union = ts.Access(inner_union, ts.literal(0))
    squeezed = ts.squeeze(access_union)
    assert isinstance(squeezed, ts.Access)

    # Test is_immutable for Access
    immutable_access = ts.Access(INT, INT)
    assert ts.is_immutable(immutable_access) == True

    mutable_access = ts.Access(LIST, INT)
    assert ts.is_immutable(mutable_access) == False


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
