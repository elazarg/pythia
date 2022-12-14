import type_system as ts


INT = ts.Ref('builtins.int')
FLOAT = ts.Ref('builtins.float')

T = ts.TypeVar('T')
T1 = ts.TypeVar('T1')
T2 = ts.TypeVar('T2')
Args = ts.TypeVar('Args', is_args=True)

def make_function(return_type: ts.TypeExpr, params: ts.Intersection[ts.Row]) -> ts.FunctionType:
    return ts.FunctionType(params, return_type,
                           new=ts.Literal(False),
                           property=ts.Literal(False))


def make_rows(*types) -> ts.Intersection[ts.Row]:
    return ts.intersect([ts.Row(index, None, t)
                         for index, t in enumerate(types)])


def test_function_call_empty():
    f = make_function(INT, make_rows())
    arg = make_rows()
    assert ts.call(f, arg) == INT

def test_function_call_arg():
    f = make_function(INT, make_rows(INT, INT))
    arg = make_rows(INT)
    assert ts.call(f, arg) == INT

def test_unification():
    assert ts.unify(
        type_params=(T,),
        params=make_rows(T),
        args=make_rows(INT)) == (INT,)

    assert ts.unify(
        type_params=(T1, T2),
        params=make_rows(T1, T2),
        args=make_rows(INT, FLOAT)) == (INT, FLOAT)

    args = make_rows(INT, FLOAT)
    assert ts.unify(
        type_params=(Args,),
        params=make_rows(Args),
        args=args) == (args,)

    args = make_rows(FLOAT, INT, FLOAT)
    assert ts.unify(
        type_params=(T, Args),
        params=make_rows(T, Args, T),
        args=args) == (FLOAT, make_rows(INT))

def test_function_call_generic_project():
    f = ts.Generic((T,), make_function(T, make_rows(T)))
    arg = make_rows(INT)
    assert ts.call(f, arg) == INT

    f = ts.Generic((T1, T2), make_function(T1, make_rows(T1, T2)))
    arg = make_rows(INT, FLOAT)
    assert ts.call(f, arg) == INT

    f = ts.Generic((T1, T2), make_function(T2, make_rows(T1, T2)))
    arg = make_rows(INT, FLOAT)
    assert ts.call(f, arg) == FLOAT

    f = ts.Generic((T,), make_function(T, make_rows(T, T)))
    arg = make_rows(INT, FLOAT)
    assert ts.call(f, arg) == ts.join(INT, FLOAT)

def test_tuple():
    t = ts.resolve_static_ref(ts.Ref('builtins.tuple'))
    t1 = ts.simplify_generic(ts.Instantiation(t, (INT, FLOAT)))
    # assert t1 == None
