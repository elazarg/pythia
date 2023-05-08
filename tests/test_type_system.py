from pythia import type_system as ts

INT = ts.Ref('builtins.int')
FLOAT = ts.Ref('builtins.float')
STR = ts.Ref('builtins.str')
ARRAY = ts.Ref('numpy.ndarray')

T = ts.TypeVar('T')
T1 = ts.TypeVar('T1')
T2 = ts.TypeVar('T2')
N = ts.TypeVar('N')
Args = ts.TypeVar('Args', is_args=True)

FIRST = ts.Literal(0)
SECOND = ts.Literal(1)


def make_function(return_type: ts.TypeExpr, params: ts.Intersection, type_params=()) -> ts.FunctionType:
    return ts.FunctionType(params, return_type,
                           property=False,
                           type_params=type_params,
                           side_effect=ts.SideEffect(new=not ts.is_immutable(return_type), instructions=()))


def make_rows(*types) -> ts.Intersection:
    return ts.intersect([ts.make_row(index, None, t)
                         for index, t in enumerate(types)])


def test_join():
    t1 = INT
    t2 = FLOAT
    assert ts.join(t1, t2) == ts.union([t1, t2])

    t1 = INT
    t2 = ts.Literal(0)
    assert ts.join(t1, t2) == t1

    t1 = INT
    t2 = ts.constant(1)
    assert ts.join(t1, t2) == t1


def test_join_literals():
    t1 = ts.constant(0)
    t2 = ts.constant(1)
    joined = INT
    assert ts.join(t1, t2) == joined


def test_overload():
    f1 = make_function(STR, make_rows(INT))
    f2 = make_function(FLOAT, make_rows(FLOAT))
    arg = INT
    args = make_rows(arg)
    overload = ts.intersect([f1, f2])
    assert ts.call(overload, args) == STR

    f1 = make_function(STR, make_rows(INT))
    f2 = make_function(FLOAT, make_rows(FLOAT))
    arg = ts.constant(0)
    args = make_rows(arg)
    overload = ts.intersect([f1, f2])
    assert ts.call(overload, args) == STR


def test_unification():
    assert ts.unify(
        type_params=(T,),
        params=make_rows(T),
        args=make_rows(INT)) == {T: INT}

    assert ts.unify(
        type_params=(T1, T2),
        params=make_rows(T1, T2),
        args=make_rows(INT, FLOAT)) == {T1: INT, T2: FLOAT}

    args = make_rows(INT, FLOAT)
    assert ts.unify(
        type_params=(Args,),
        params=make_rows(Args),
        args=args) == {Args: args}

    args = make_rows(FLOAT, INT, FLOAT)
    assert ts.unify(
        type_params=(T, Args),
        params=make_rows(T, Args, T),
        args=args) == {T: FLOAT, Args: make_rows(INT)}


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
    arg = ts.BOTTOM
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

#
# def test_function_call_variadic():
#     f = make_function(Args, make_rows(Args), [Args])
#     arg = make_rows(INT, FLOAT)
#     assert ts.call(f, arg) == arg
#
#     f = make_function(ts.Instantiation(Args, (N,)), make_rows(N, Args), [N, Args])
#     arg = make_rows(ts.Literal(0), INT, FLOAT)
#     assert ts.call(f, arg) == INT
#     arg = make_rows(ts.Literal(1), INT, FLOAT)
#     assert ts.call(f, arg) == FLOAT


def test_tuple():
    f = make_function(T, make_rows(T, T), [T])
    arg = make_rows(INT, FLOAT)
    assert ts.call(f, arg) == ts.join(INT, FLOAT)

    tuple_named = ts.Instantiation(ts.Ref('builtins.tuple'), (INT, FLOAT))

    tuple_structure = make_rows(INT, FLOAT)

    gt = ts.subscr(tuple_named, ts.Literal('__getitem__'))
    f = ts.FunctionType(params=ts.intersect([ts.make_row(0, 'item', N)]),
                        return_type=ts.Instantiation(tuple_structure, (N,)),
                        property=False,
                        side_effect=ts.SideEffect(new=True, instructions=()),
                        type_params=(N,))
    assert gt == f
    x = ts.call(gt, make_rows(FIRST))
    assert x == INT

    x = ts.subscr(tuple_named, FIRST)
    assert x == INT

    x = ts.subscr(tuple_named, SECOND)
    assert x == FLOAT

    x = ts.call(gt, make_rows(SECOND))
    assert x == FLOAT

    left = ts.subscr(tuple_structure, FIRST)
    assert left == INT

    right = ts.subscr(tuple_structure, SECOND)
    assert right == FLOAT

    both = ts.intersect([tuple_structure, tuple_named])

    left = ts.subscr(both, FIRST)
    assert left == INT

    right = ts.subscr(both, SECOND)
    assert right == FLOAT


def test_list():
    # Hash:
    # - 125 Fails with empty
    # - 126 Fails with recursion
    # - 127 Pass

    t = ts.Ref('builtins.list')
    t1 = ts.simplify_generic(ts.Instantiation(t, (INT,)), {})
    gt = ts.subscr(t1, ts.Literal('__getitem__'))
    x = ts.call(gt, make_rows(ts.Literal(0)))
    assert x == INT

    gt = ts.subscr(t1, ts.Literal('__add__'))
    x = ts.call(gt, make_rows(t1))
    assert x == t1


def test_getitem_list():
    t = ts.Instantiation(ts.Ref('builtins.list'), (INT,))
    x = ts.subscr(t, ts.Literal(0))
    assert x == INT


def test_getitem_numpy():
    x = ts.subscr(ARRAY, ts.constant(0))
    assert x == FLOAT

    x = ts.subscr(ARRAY, ARRAY)
    assert x == ARRAY

    x = ts.subscr(ARRAY, ts.Ref('builtins.slice'))
    assert x == ARRAY

    x = ts.subscr(ARRAY, ts.Ref('builtins.None'))
    assert x == ts.BOTTOM


def test_list_constructor():
    t = ts.Ref('builtins.list')
    constructor = ts.make_constructor(t)
    args = make_rows(INT)
    lst = ts.call(constructor, args)
    assert lst == ts.intersect([args, ts.Instantiation(t, (INT,))])
