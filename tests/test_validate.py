"""Tests for pythia.validate - supported subset validation pass."""

from __future__ import annotations as _

import warnings

import pytest

from pythia import tac, validate
from pythia.graph_utils import Cfg, Block


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(instructions: list[tac.Tac]) -> Cfg[tac.Tac]:
    """Build a minimal CFG with a single block containing the given instructions."""
    return Cfg(
        graph=[(0, 1)],
        blocks={0: instructions, 1: []},
    )


def _v(name: str) -> tac.Var:
    return tac.Var(name)


def _sv(i: int) -> tac.Var:
    return tac.Var(f"${i}", is_stackvar=True)


# ---------------------------------------------------------------------------
# Tests: Unsupported bytecodes
# ---------------------------------------------------------------------------


class TestUnsupportedBytecodes:
    def test_unsupported_instruction(self) -> None:
        cfg = _make_cfg([tac.Unsupported("BEFORE_WITH")])
        violations = validate.check_function(cfg)
        assert len(violations) == 1
        assert violations[0].category == "unsupported_bytecode"
        assert violations[0].severity == validate.Severity.ERROR
        assert "BEFORE_WITH" in violations[0].message

    def test_multiple_unsupported(self) -> None:
        cfg = _make_cfg([
            tac.Unsupported("BEFORE_WITH"),
            tac.Unsupported("SETUP_FINALLY"),
        ])
        violations = validate.check_function(cfg)
        assert len(violations) == 2


# ---------------------------------------------------------------------------
# Tests: Dynamic execution
# ---------------------------------------------------------------------------


class TestRejectedBuiltins:
    @pytest.mark.parametrize(
        "name",
        ["eval", "exec", "compile", "setattr", "delattr",
         "globals", "locals", "vars", "__import__", "breakpoint"],
    )
    def test_rejected_builtin(self, name: str) -> None:
        expr = tac.Attribute(tac.PredefinedScope.GLOBALS, _v(name))
        cfg = _make_cfg([tac.Assign(_sv(0), expr)])
        violations = validate.check_function(cfg)
        assert len(violations) == 1
        assert violations[0].category == "rejected_builtin"
        assert violations[0].severity == validate.Severity.ERROR
        assert name in violations[0].message

    def test_message_explains_why(self) -> None:
        expr = tac.Attribute(tac.PredefinedScope.GLOBALS, _v("eval"))
        cfg = _make_cfg([tac.Assign(_sv(0), expr)])
        [v] = validate.check_function(cfg)
        assert "arbitrary code" in v.message


# ---------------------------------------------------------------------------
# Tests: Object internals
# ---------------------------------------------------------------------------


class TestObjectInternals:
    @pytest.mark.parametrize(
        "field",
        [
            "__dict__",
            "__class__",
            "__bases__",
            "__code__",
            "__closure__",
            "__func__",
            "__defaults__",
            "__builtins__",
        ],
    )
    def test_rejected_field_read(self, field: str) -> None:
        expr = tac.Attribute(_v("obj"), _v(field))
        cfg = _make_cfg([tac.Assign(_sv(0), expr)])
        violations = validate.check_function(cfg)
        assert any(v.category == "rejected_field" for v in violations)
        assert all(v.severity == validate.Severity.ERROR for v in violations)

    @pytest.mark.parametrize("field", ["__dict__", "__class__", "__bases__"])
    def test_rejected_field_write(self, field: str) -> None:
        target = tac.Attribute(_v("obj"), _v(field))
        cfg = _make_cfg([tac.Assign(target, _v("val"))])
        violations = validate.check_function(cfg)
        assert any(v.category == "rejected_field" for v in violations)

    def test_normal_field_allowed(self) -> None:
        expr = tac.Attribute(_v("obj"), _v("x"))
        cfg = _make_cfg([tac.Assign(_sv(0), expr)])
        violations = validate.check_function(cfg)
        assert not violations

    def test_normal_dunder_allowed(self) -> None:
        # __init__, __len__, etc. are not in the dangerous set
        expr = tac.Attribute(_v("obj"), _v("__init__"))
        cfg = _make_cfg([tac.Assign(_sv(0), expr)])
        violations = validate.check_function(cfg)
        assert not violations


# ---------------------------------------------------------------------------
# Tests: Yield
# ---------------------------------------------------------------------------


class TestYield:
    def test_yield_rejected(self) -> None:
        cfg = _make_cfg([tac.Assign(_sv(0), tac.Yield(_v("x")))])
        violations = validate.check_function(cfg)
        assert any(v.category == "yield" for v in violations)
        assert all(v.severity == validate.Severity.ERROR for v in violations)


# ---------------------------------------------------------------------------
# Tests: Closures
# ---------------------------------------------------------------------------


class TestClosures:
    def test_nonlocal_read_allowed(self) -> None:
        expr = tac.Attribute(tac.PredefinedScope.NONLOCALS, _v("x"))
        cfg = _make_cfg([tac.Assign(_sv(0), expr)])
        violations = validate.check_function(cfg)
        assert not violations

    def test_nonlocal_write_rejected(self) -> None:
        target = tac.Attribute(tac.PredefinedScope.NONLOCALS, _v("x"))
        cfg = _make_cfg([tac.Assign(target, _v("val"))])
        violations = validate.check_function(cfg)
        assert any(v.category == "nonlocal_mutation" for v in violations)
        assert all(v.severity == validate.Severity.ERROR for v in violations)

    def test_closure_creation_warning(self) -> None:
        func = tac.MakeFunction(
            code=_v("code"),
            free_var_cells=_sv(3),
        )
        cfg = _make_cfg([tac.Assign(_sv(0), func)])
        violations = validate.check_function(cfg)
        assert any(v.category == "closure_creation" for v in violations)
        assert all(v.severity == validate.Severity.WARNING for v in violations)

    def test_function_without_closure_allowed(self) -> None:
        func = tac.MakeFunction(code=_v("code"))
        cfg = _make_cfg([tac.Assign(_sv(0), func)])
        violations = validate.check_function(cfg)
        assert not violations


# ---------------------------------------------------------------------------
# Tests: Imports
# ---------------------------------------------------------------------------


class TestImports:
    def test_rejected_import_in_function(self) -> None:
        expr = tac.Import(modname=_v("threading"))
        cfg = _make_cfg([tac.Assign(_sv(0), expr)])
        violations = validate.check_function(cfg)
        assert any(v.category == "rejected_import" for v in violations)
        assert all(v.severity == validate.Severity.WARNING for v in violations)

    def test_safe_import_allowed(self) -> None:
        expr = tac.Import(modname=_v("math"))
        cfg = _make_cfg([tac.Assign(_sv(0), expr)])
        violations = validate.check_function(cfg)
        assert not violations

    @pytest.mark.parametrize(
        "module",
        ["threading", "pickle", "ctypes", "sys", "asyncio", "signal", "weakref"],
    )
    def test_dangerous_modules(self, module: str) -> None:
        expr = tac.Import(modname=_v(module))
        cfg = _make_cfg([tac.Assign(_sv(0), expr)])
        violations = validate.check_function(cfg)
        assert any(v.category == "rejected_import" for v in violations)


class TestModuleImports:
    def test_dangerous_module_import(self) -> None:
        imports = {"threading": "threading"}
        violations = validate.check_module_imports(imports)
        assert len(violations) == 1
        assert violations[0].category == "rejected_import"
        assert violations[0].severity == validate.Severity.WARNING
        assert violations[0].location is None

    def test_submodule_import(self) -> None:
        imports = {"futures": "concurrent.futures"}
        violations = validate.check_module_imports(imports)
        assert len(violations) == 1  # root module "concurrent" is dangerous

    def test_safe_module_import(self) -> None:
        imports = {"np": "numpy", "math": "math"}
        violations = validate.check_module_imports(imports)
        assert not violations

    def test_multiple_imports(self) -> None:
        imports = {
            "np": "numpy",
            "threading": "threading",
            "pickle": "pickle",
            "math": "math",
        }
        violations = validate.check_module_imports(imports)
        assert len(violations) == 2


# ---------------------------------------------------------------------------
# Tests: Safe code
# ---------------------------------------------------------------------------


class TestSafeCode:
    """Verify that common safe patterns produce no violations."""

    def test_local_assignment(self) -> None:
        cfg = _make_cfg([tac.Assign(_v("x"), tac.Const(42))])
        assert not validate.check_function(cfg)

    def test_attribute_access(self) -> None:
        cfg = _make_cfg([tac.Assign(_sv(0), tac.Attribute(_v("obj"), _v("field")))])
        assert not validate.check_function(cfg)

    def test_subscript(self) -> None:
        cfg = _make_cfg([tac.Assign(_sv(0), tac.Subscript(_v("arr"), _v("i")))])
        assert not validate.check_function(cfg)

    def test_binary_op(self) -> None:
        cfg = _make_cfg(
            [tac.Assign(_sv(0), tac.Binary(_v("a"), "+", _v("b"), inplace=False))]
        )
        assert not validate.check_function(cfg)

    def test_unary_op(self) -> None:
        cfg = _make_cfg(
            [tac.Assign(_sv(0), tac.Unary(tac.UnOp.NEG, _v("x")))]
        )
        assert not validate.check_function(cfg)

    def test_function_call(self) -> None:
        cfg = _make_cfg([
            tac.Assign(_sv(0), tac.BoundCall(_v("f"), (_v("x"),))),
            tac.Assign(_sv(1), tac.Call(_sv(0), ())),
        ])
        assert not validate.check_function(cfg)

    def test_for_loop(self) -> None:
        cfg = _make_cfg([tac.For(_v("x"), _v("iter"), jump_target=1, original_lineno=5)])
        assert not validate.check_function(cfg)

    def test_return(self) -> None:
        cfg = _make_cfg([tac.Return(_v("x"))])
        assert not validate.check_function(cfg)

    def test_raise(self) -> None:
        cfg = _make_cfg([tac.Raise(_v("exc"))])
        assert not validate.check_function(cfg)

    def test_del(self) -> None:
        cfg = _make_cfg([tac.Del((_v("x"),))])
        assert not validate.check_function(cfg)

    def test_nop(self) -> None:
        cfg = _make_cfg([tac.Nop()])
        assert not validate.check_function(cfg)

    def test_jump(self) -> None:
        cfg = _make_cfg([tac.Jump(jump_target=1)])
        assert not validate.check_function(cfg)

    def test_global_read(self) -> None:
        expr = tac.Attribute(tac.PredefinedScope.GLOBALS, _v("print"))
        cfg = _make_cfg([tac.Assign(_sv(0), expr)])
        assert not validate.check_function(cfg)

    def test_safe_import(self) -> None:
        cfg = _make_cfg([tac.Assign(_sv(0), tac.Import(_v("numpy")))])
        assert not validate.check_function(cfg)

    def test_make_function_no_closure(self) -> None:
        cfg = _make_cfg([tac.Assign(_sv(0), tac.MakeFunction(code=_v("c")))])
        assert not validate.check_function(cfg)

    def test_container_construction(self) -> None:
        cfg = _make_cfg([
            tac.Assign(
                _sv(0), tac.Call(tac.PredefinedFunction.LIST, (_v("a"), _v("b")))
            ),
        ])
        assert not validate.check_function(cfg)


# ---------------------------------------------------------------------------
# Tests: Integration (validate function)
# ---------------------------------------------------------------------------


class TestValidate:
    def test_errors_raise(self) -> None:
        cfg = _make_cfg([tac.Unsupported("BEFORE_WITH")])
        with pytest.raises(validate.UnsupportedFeatureError) as exc_info:
            validate.validate(cfg, "my_func")
        assert len(exc_info.value.violations) == 1

    def test_warnings_only_do_not_raise(self) -> None:
        cfg = _make_cfg([tac.Assign(_sv(0), tac.Import(_v("threading")))])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate.validate(cfg, "my_func")  # should not raise
        assert len(w) == 1
        assert "threading" in str(w[0].message)

    def test_clean_code_passes(self) -> None:
        cfg = _make_cfg([tac.Assign(_v("x"), tac.Const(42))])
        validate.validate(cfg, "my_func")  # should not raise

    def test_module_imports_warned(self) -> None:
        cfg = _make_cfg([tac.Assign(_v("x"), tac.Const(42))])
        imports = {"pickle": "pickle"}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate.validate(cfg, "my_func", imports)
        assert len(w) == 1
        assert "pickle" in str(w[0].message)

    def test_errors_and_warnings_combined(self) -> None:
        """When both errors and warnings exist, warnings are emitted and errors raised."""
        cfg = _make_cfg([
            tac.Assign(_sv(0), tac.Import(_v("threading"))),
            tac.Unsupported("BEFORE_WITH"),
        ])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with pytest.raises(validate.UnsupportedFeatureError) as exc_info:
                validate.validate(cfg, "my_func")
        # Warning was still emitted before the error was raised
        assert any("threading" in str(warning.message) for warning in w)
        # Only the error is in the exception
        assert all(
            v.severity == validate.Severity.ERROR
            for v in exc_info.value.violations
        )

    def test_multiple_errors_collected(self) -> None:
        """All violations are collected, not just the first."""
        cfg = _make_cfg([
            tac.Assign(_sv(0), tac.Yield(_v("x"))),
            tac.Unsupported("BEFORE_WITH"),
        ])
        with pytest.raises(validate.UnsupportedFeatureError) as exc_info:
            validate.validate(cfg, "my_func")
        assert len(exc_info.value.violations) == 2
