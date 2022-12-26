from frozendict import frozendict

import type_system as ts


INT = ts.Ref('builtins.int')
FLOAT = ts.Ref('builtins.float')
STR = ts.Ref('builtins.str')
ARRAY = ts.Ref('numpy.ndarray')

T = ts.TypeVar('T')
T1 = ts.TypeVar('T1')
T2 = ts.TypeVar('T2')
N = ts.TypeVar('N')
Args = ts.TypeVar('Args', is_args=True)


def make_function(return_type: ts.TypeExpr, params: ts.Intersection[ts.Row], type_params=()) -> ts.FunctionType:
    return ts.FunctionType(params, return_type,
                           new=ts.Literal(True),
                           property=ts.Literal(False),
                           type_params=frozendict({x: None for x in type_params}))


def make_rows(*types) -> ts.Intersection[ts.Row]:
    return ts.intersect([ts.Row(ts.Index(ts.Literal(index), None), t)
                         for index, t in enumerate(types)])


def test_join():
    t1 = ts.Ref('builtins.int')
    t2 = ts.Ref('builtins.float')
    assert ts.join(t1, t2) == ts.Union({t1, t2})


def test_function_call():
    f = make_function(INT, make_rows())
    arg = make_rows()
    assert ts.call(f, arg) == INT

    f = make_function(INT, make_rows(INT, FLOAT))
    arg = make_rows(INT, FLOAT)
    assert ts.call(f, arg) == INT

    f = make_function(INT, make_rows(INT, FLOAT))
    arg = make_rows(INT, INT)
    assert ts.call(f, arg) == ts.TOP

    f = make_function(INT, make_rows())
    arg = ts.BOTTOM
    assert ts.call(f, arg) == INT


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


def test_function_call_variadic():
    f = make_function(Args, make_rows(Args), [Args])
    arg = make_rows(INT, FLOAT)
    assert ts.call(f, arg) == arg

    f = make_function(ts.Instantiation(Args, (N,)), make_rows(N, Args), [N, Args])
    arg = make_rows(ts.Literal(0), INT, FLOAT)
    assert ts.call(f, arg) == INT
    arg = make_rows(ts.Literal(1), INT, FLOAT)
    assert ts.call(f, arg) == FLOAT


def test_tuple():
    f = make_function(T, make_rows(T, T), [T])
    arg = make_rows(INT, FLOAT)
    assert ts.call(f, arg) == ts.join(INT, FLOAT)

    tuple_named = ts.Instantiation(ts.Ref('builtins.tuple'), (INT, FLOAT))

    tuple_structure = make_rows(INT, FLOAT)

    first = ts.Literal(0)
    second = ts.Literal(1)
    first_intersect = ts.intersect([first, ts.Ref('builtins.int')])
    second_intersect = ts.intersect([second, ts.Ref('builtins.int')])

    gt = ts.subscr(tuple_named, ts.Literal('__getitem__'))
    f = ts.FunctionType(params=ts.intersect([ts.Row(ts.Index(ts.Literal(0), ts.Literal('item')), N)]),
                        return_type=ts.Instantiation(tuple_structure, (N,)),
                        new=ts.Literal(True),
                        property=ts.Literal(False),
                        type_params=(N,))
    assert gt == f
    x = ts.call(gt, make_rows(first))
    assert x == INT
    x = ts.call(gt, make_rows(first_intersect))
    assert x == INT

    x = ts.subscr(tuple_named, first)
    assert x == INT
    x = ts.subscr(tuple_named, first_intersect)
    assert x == INT

    x = ts.subscr(tuple_named, second)
    assert x == FLOAT
    x = ts.subscr(tuple_named, second_intersect)
    assert x == FLOAT

    x = ts.call(gt, make_rows(second))
    assert x == FLOAT
    x = ts.call(gt, make_rows(second_intersect))
    assert x == FLOAT

    left = ts.subscr(tuple_structure, first)
    assert left == INT
    left = ts.subscr(tuple_structure, first_intersect)
    assert left == INT

    right = ts.subscr(tuple_structure, second)
    assert right == FLOAT
    right = ts.subscr(tuple_structure, second_intersect)
    assert right == FLOAT

    both = ts.intersect([tuple_structure, tuple_named])

    left = ts.subscr(both, first)
    assert left == INT
    left = ts.subscr(both, first_intersect)
    assert left == INT

    right = ts.subscr(both, second)
    assert right == FLOAT
    right = ts.subscr(both, second_intersect)
    assert right == FLOAT

    tuple_named = ts.Instantiation(ts.Ref('builtins.tuple'),
                                   (ts.intersect([ts.Literal(1), INT]),
                                    ts.intersect([ts.Literal('x'), STR])))
    left = ts.subscr(tuple_named, first_intersect)
    assert left == ts.intersect([ts.Literal(1), INT])
    right = ts.subscr(tuple_named, second_intersect)
    assert right == ts.intersect([ts.Literal('x'), STR])


def test_list():
    t = ts.Ref('builtins.list')
    t1 = ts.simplify_generic(ts.Instantiation(t, (INT,)), {})
    gt = ts.subscr(t1, ts.Literal('__getitem__'))
    x = ts.call(gt, make_rows(ts.Literal(0)))
    assert x == INT

def test_getitem_list():
    t = ts.Instantiation(ts.Ref('builtins.list'), (INT,))
    x = ts.subscr(t, ts.Literal(0))
    assert x == INT

def test_getitem_numpy():
    x = ts.subscr(ARRAY, ts.intersect([ts.Literal(0), INT]))
    assert x == FLOAT

    x = ts.subscr(ARRAY, ARRAY)
    assert x == ARRAY

    x = ts.subscr(ARRAY, ts.Ref('builtins.slice'))
    assert x == ARRAY

    x = ts.subscr(ARRAY, ts.Ref('builtins.None'))
    assert x == ts.TOP


def test_list_constructor():
    t = ts.Ref('builtins.list')
    constructor = ts.make_constructor(t)
    args = make_rows(INT)
    lst = ts.call(constructor, args)
    assert lst == ts.intersect([args, ts.Instantiation(t, (INT,))])
