from dataclasses import replace

from pythia import type_system as ts

INT = ts.Ref('builtins.int')
FLOAT = ts.Ref('builtins.float')
STR = ts.Ref('builtins.str')
ARRAY = ts.Ref('numpy.ndarray')
LIST = ts.Ref('builtins.list')
TUPLE = ts.Ref('builtins.tuple')

T = ts.TypeVar('T')
T1 = ts.TypeVar('T1')
T2 = ts.TypeVar('T2')
N = ts.TypeVar('N')
Args = ts.TypeVar('Args', is_args=True)

FIRST = ts.literal(0)
SECOND = ts.literal(1)


def make_function(return_type: ts.TypeExpr, params: ts.TypedDict, type_params=(), update=None) -> ts.FunctionType:
    return ts.FunctionType(params, return_type,
                           is_property=False,
                           type_params=type_params,
                           side_effect=ts.SideEffect(
                               new=not ts.is_immutable(return_type),
                               update=update))


def make_rows(*types) -> ts.TypedDict:
    return ts.typed_dict([ts.make_row(index, None, t)
                          for index, t in enumerate(types)])


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
        type_params=(T,),
        params=make_rows(T),
        args=make_rows(INT)).bound_typevars == {T: INT}

    assert ts.unify(
        type_params=(T1, T2),
        params=make_rows(T1, T2),
        args=make_rows(INT, FLOAT)).bound_typevars == {T1: INT, T2: FLOAT}


def test_unification_args():
    args = make_rows(INT, FLOAT)
    assert ts.unify(
        type_params=(Args,),
        params=make_rows(Args),
        args=args).bound_typevars == {Args: ts.Star((INT, FLOAT))}

    args = make_rows(FLOAT, INT, FLOAT)
    assert ts.unify(
        type_params=(T, Args),
        params=make_rows(T, Args, T),
        args=args).bound_typevars == {T: FLOAT, Args: ts.Star((INT,))}


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
    f = ts.FunctionType(params=make_rows(tuple_param, N),
                        return_type=ts.Access(Args, N),
                        is_property=False,
                        side_effect=ts.SideEffect(new=False),
                        type_params=(N, Args))
    g = replace(f,
                params=make_rows(N),
                return_type=ts.Access(tuple_star, N),
                type_params=(N,))
    assert ts.bind_self(ts.overload([f]), tuple_named) == ts.overload([g])


def test_tuple():
    f = make_function(T, make_rows(T, T), [T])
    arg = make_rows(INT, FLOAT)
    assert ts.call(f, arg) == ts.join(INT, FLOAT)

    tuple_named = ts.Instantiation(TUPLE, (INT, FLOAT))

    tuple_star = ts.Star((INT, FLOAT))
    tuple_structure = ts.literal((INT, FLOAT))

    g = ts.subscr(tuple_named, ts.literal('__getitem__'))
    f = ts.FunctionType(params=ts.typed_dict([ts.make_row(0, 'item', N)]),
                        return_type=ts.Access(tuple_star, N),
                        is_property=False,
                        side_effect=ts.SideEffect(new=False),
                        type_params=(N,))
    assert g == ts.overload([f])
    x = ts.call(g, make_rows(FIRST))
    assert x == INT

    x = ts.subscr(tuple_named, FIRST)
    assert x == INT

    x = ts.subscr(tuple_named, SECOND)
    assert x == FLOAT

    x = ts.call(g, make_rows(SECOND))
    assert x == FLOAT

    left = ts.subscr(tuple_structure, FIRST)
    assert left == INT

    right = ts.subscr(tuple_structure, SECOND)
    assert right == FLOAT


def test_list():
    t1 = ts.Instantiation(LIST, (INT,))
    gt = ts.subscr(t1, ts.literal('__getitem__'))
    x = ts.call(gt, make_rows(ts.literal(0)))
    assert x == INT

    gt = ts.subscr(t1, ts.literal('__add__'))
    x = ts.call(gt, make_rows(t1))
    assert x == t1


def test_list_join():
    non_empty1 = ts.literal([INT])
    non_empty2 = ts.literal([INT])
    assert ts.join(non_empty1, non_empty2) == non_empty1

    empty = ts.literal([])
    non_empty = ts.literal([INT])
    assert ts.join(empty, non_empty) == ts.Instantiation(LIST, (INT,))


def test_getitem_list():
    t = ts.Instantiation(LIST, (INT,))
    x = ts.subscr(t, ts.literal(0))
    assert x == INT


def test_getitem_numpy():
    x = ts.subscr(ARRAY, ts.literal(0))
    assert x == FLOAT

    x = ts.subscr(ARRAY, ARRAY)
    assert x == ARRAY

    x = ts.subscr(ARRAY, ts.Ref('builtins.slice'))
    assert x == ARRAY

    x = ts.subscr(ARRAY, ts.Ref('builtins.None'))
    assert x == ts.BOTTOM


def test_operator_numpy():
    x = ts.binop(ARRAY, ARRAY, '+')
    assert x == ARRAY


def test_right_operator_numpy():
    x = ts.binop(FLOAT, ARRAY, '+')
    assert x == ARRAY

    x = ts.binop(FLOAT, ARRAY, '*')
    assert x == ARRAY


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
