from pythia.tac import (
    Const,
    Var,
    Attribute,
    Subscript,
    Binary,
    Unary,
    Call,
    Yield,
    Import,
    MakeFunction,
    Nop,
    Assign,
    Jump,
    For,
    Return,
    Raise,
    PredefinedFunction,
    UnOp,
    free_vars_expr,
    free_vars,
    gens,
    stackvar,
    is_stackvar,
)

X = Var("x", False)
Y = Var("y", False)
A = Var("a", False)
B = Var("b", False)


def test_const():
    # Test Const creation and string representation
    c = Const(42)
    assert c.value == 42
    assert str(c) == "42"
    assert repr(c) == "42"

    c = Const("hello")
    assert c.value == "hello"
    assert str(c) == "'hello'"
    assert repr(c) == "'hello'"


def test_var():
    # Test Var creation and string representation
    v = X
    assert v.name == "x"
    assert not v.is_stackvar
    assert str(v) == "x"
    assert repr(v) == "x"

    # Test stackvar
    v = Var("1", True)
    assert v.name == "1"
    assert v.is_stackvar
    assert str(v) == "$1"
    assert repr(v) == "$1"


def test_stackvar_function():
    # Test stackvar function
    v = stackvar(1)
    assert v.name == "1"
    assert v.is_stackvar
    assert str(v) == "$1"


def test_is_stackvar():
    # Test is_stackvar function
    v1 = X
    v2 = stackvar(1)
    c = Const(42)

    assert not is_stackvar(v1)
    assert is_stackvar(v2)
    assert not is_stackvar(c)


def test_attribute():
    # Test Attribute creation and string representation
    v = Var("obj")
    f = Var("attr")
    attr = Attribute(v, f)

    assert attr.var == v
    assert attr.field == f
    assert str(attr) == "obj.attr"

    # Test with PredefinedScope
    scope = Var("builtins")
    attr = Attribute(scope, f)
    assert attr.var == scope
    assert attr.field == f
    assert str(attr) == "builtins.attr"


def test_subscript():
    # Test Subscript creation and string representation
    v = Var("arr")
    idx = Var("i")
    sub = Subscript(v, idx)

    assert sub.var == v
    assert sub.index == idx
    assert str(sub) == "arr[i]"


def test_binary():
    # Test Binary creation and string representation
    left = A
    right = B
    op = "+"
    bin_op = Binary(left, op, right, False)

    assert bin_op.left == left
    assert bin_op.op == op
    assert bin_op.right == right
    assert not bin_op.inplace
    assert str(bin_op) == "a + b"

    # Test with inplace=True
    bin_op = Binary(left, "+=", right, True)
    assert bin_op.inplace
    assert str(bin_op) == "a += b"


def test_unary():
    # Test Unary creation and string representation
    v = X
    op = UnOp.NOT
    unary_op = Unary(op, v)

    assert unary_op.op == op
    assert unary_op.var == v
    assert str(unary_op) == "NOT x"


def test_call():
    # Test Call creation and string representation
    func = Var("func")
    args = (A, B)
    call = Call(func, args)

    assert call.function == func
    assert call.args == args
    assert call.kwargs is None
    assert str(call) == "func(a, b)"

    # Test with kwargs
    kwargs = Var("kwargs")
    call = Call(func, args, kwargs)
    assert call.kwargs == kwargs
    assert str(call) == "func(a, b), kwargs=kwargs"

    # Test with PredefinedFunction
    func = PredefinedFunction.SET
    call = Call(func, args)
    assert call.function == func
    assert str(call) == "SET(a, b)"


def test_yield():
    # Test Yield creation
    v = X
    y = Yield(v)

    assert y.value == v


def test_import():
    # Test Import creation and string representation
    mod = Var("module")
    imp = Import(mod)

    assert imp.modname == mod
    assert imp.feature is None
    assert str(imp) == "IMPORT module"

    # Test with feature
    imp = Import(mod, "feature")
    assert imp.feature == "feature"
    assert str(imp) == "IMPORT module.feature"

    # Test with Attribute
    attr = Attribute(Var("package"), Var("module"))
    imp = Import(attr)
    assert imp.modname == attr
    assert str(imp) == "IMPORT package.module"


def test_make_function():
    # Test MakeFunction creation
    code = Var("code")
    free_vars_set = frozenset([X, Y])
    func = MakeFunction(code, free_vars_set)

    assert func.code == code
    assert func.free_vars == free_vars_set


def test_nop():
    # Test Nop creation
    nop = Nop()
    assert isinstance(nop, Nop)


def test_assign():
    # Test Assign creation and string representation
    lhs = X
    expr = Binary(A, "+", B, False)
    assign = Assign(lhs, expr)

    assert assign.lhs == lhs
    assert assign.expr == expr
    assert not assign.or_null
    assert str(assign) == "x = a + b"

    # Test with or_null=True
    assign = Assign(lhs, expr, True)
    assert assign.or_null
    assert str(assign) == "x = a + b?"

    # Test with lhs=None
    assign = Assign(None, expr)
    assert assign.lhs is None
    assert str(assign) == "a + b"


def test_jump():
    # Test Jump creation and string representation
    target = 10
    jump = Jump(target)

    assert jump.jump_target == target
    assert jump.cond == Const(True)
    assert str(jump) == "GOTO 10"

    # Test with condition
    cond = Var("cond")
    jump = Jump(target, cond)
    assert jump.cond == cond
    assert str(jump) == "IF cond GOTO 10"


def test_for():
    # Test For creation and string representation
    lhs = X
    iterator = Var("iter")
    target = 10
    lineno = 5
    for_loop = For(lhs, iterator, target, lineno)

    assert for_loop.lhs == lhs
    assert for_loop.iterator == iterator
    assert for_loop.jump_target == target
    assert for_loop.original_lineno == lineno
    assert str(for_loop) == "x = next(iter) HANDLE: GOTO 10"

    # Test as_call method
    call = for_loop.as_call()
    assert isinstance(call, Assign)
    assert call.lhs == lhs
    assert isinstance(call.expr, Unary)
    assert call.expr.op == UnOp.NEXT
    assert call.expr.var == iterator


def test_free_vars_expr():
    # Test free_vars_expr with different expressions

    # Test Const
    assert free_vars_expr(Const(42)) == set()

    # Test Var
    v = X
    assert free_vars_expr(v) == {v}

    # Test Attribute
    attr = Attribute(Var("obj"), Var("attr"))
    assert free_vars_expr(attr) == {Var("obj")}

    # Test Subscript
    sub = Subscript(Var("arr"), Var("i"))
    assert free_vars_expr(sub) == {Var("arr"), Var("i")}

    # Test Binary
    bin_op = Binary(A, "+", B, False)
    assert free_vars_expr(bin_op) == {A, B}

    # Test Unary
    unary_op = Unary(UnOp.NOT, X)
    assert free_vars_expr(unary_op) == {X}

    # Test Call
    call = Call(Var("func"), (A, B))
    assert free_vars_expr(call) == {Var("func"), A, B}

    # Test Call with kwargs
    call = Call(Var("func"), (A, B), Var("kwargs"))
    assert free_vars_expr(call) == {Var("func"), A, B, Var("kwargs")}

    # Test Yield
    assert free_vars_expr(Yield(X)) == {X}

    # Test Import
    imp = Import(Var("module"))
    assert free_vars_expr(imp) == set()

    # Test MakeFunction
    func = MakeFunction(Var("code"), frozenset([X, Y]))
    assert (
        free_vars_expr(func) == set()
    )  # Note: This might change if MakeFunction.free_vars is included


def test_free_vars():
    # Test free_vars with different TAC instructions

    # Test Nop
    assert free_vars(Nop()) == set()

    # Test Assign
    assign = Assign(X, Binary(A, "+", B, False))
    assert free_vars(assign) == {A, B}

    # Test Jump
    jump = Jump(10, Var("cond"))
    assert free_vars(jump) == {Var("cond")}

    # Test For
    for_loop = For(X, Var("iter"), 10, 5)
    assert free_vars(for_loop) == {Var("iter")}

    # Test Return
    assert free_vars(Return(X)) == {X}

    # Test Raise
    raise_stmt = Raise(Var("error"))
    assert free_vars(raise_stmt) == {Var("error")}


def test_gens():
    # Test gens with different TAC instructions

    # Test Nop
    assert gens(Nop()) == set()

    # Test Assign with Var lhs
    assign = Assign(X, Binary(A, "+", B, False))
    assert gens(assign) == {X}

    # Test Assign with Attribute lhs
    assign = Assign(Attribute(Var("obj"), Var("attr")), Var("value"))
    assert gens(assign) == set()  # Attribute assignments don't generate variables

    # Test Assign with Subscript lhs
    assign = Assign(Subscript(Var("arr"), Var("i")), Var("value"))
    assert gens(assign) == set()  # Subscript assignments don't generate variables

    # Test Jump
    jump = Jump(10, Var("cond"))
    assert gens(jump) == set()

    # Test For
    for_loop = For(X, Var("iter"), 10, 5)
    assert gens(for_loop) == {X}

    # Test Return
    ret = Return(X)
    assert gens(ret) == set()

    # Test Raise
    raise_stmt = Raise(Var("error"))
    assert gens(raise_stmt) == set()
