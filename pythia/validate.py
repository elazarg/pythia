"""Validation pass: checks that analyzed code stays within the supported subset.

Supported Python subset
=======================

The analysis works on ordinary Python functions that use standard
operations on types it knows about.  Here is what you can write:

  Supported types     int, float, bool, str, None,
                      list, tuple, set, dict, slice,
                      numpy arrays and scalars,
                      and any type you add to typeshed_mini/.

  Expressions         arithmetic, comparisons, boolean logic,
                      attribute access (obj.field),
                      item access (obj[key], slicing),
                      function and method calls,
                      container literals ([...], {...}, (...)),
                      unpacking (a, b = x  and  a, *b = x).

  Statements          assignment, del, return, raise,
                      if/elif/else, for, while,
                      import (of modules with stubs),
                      function definitions,
                      closures that READ enclosing variables.

  Not supported       yield / generators,
                      async / await,
                      try / except / finally,
                      with (context managers),
                      *args/**kwargs forwarding (f(*a, **kw)),
                      closures that WRITE to enclosing variables.

  Rejected builtins   eval, exec, compile,
                      setattr, delattr,
                      globals, locals, vars.

  Rejected access     obj.__dict__, obj.__class__, obj.__bases__
                      and other internal-state dunders.

Type requirements
-----------------
Every type your function touches must have a stub in typeshed_mini/.
If a type is missing, the analysis will fail with an assertion error
when it tries to resolve a method or attribute.

Currently stubbed:  builtins, collections, numpy, typing.
To add support for a new library, add a .pyi file to typeshed_mini/.

Effect annotations
------------------
Function stubs must declare their heap effects (@new, @update, @alias,
@accessor).  A function without effect annotations is assumed to have
no side effects.  If that is wrong, the analysis will miss mutations
and report incorrect dirty sets.
"""

from __future__ import annotations as _

import enum
import warnings
from dataclasses import dataclass

from pythia import tac
from pythia.graph_utils import Cfg, Location

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class Severity(enum.Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class Violation:
    location: Location | None  # None for module-level warnings
    category: str
    message: str
    severity: Severity

    def __str__(self) -> str:
        prefix = self.severity.value.upper()
        loc = f" at {self.location}" if self.location else ""
        return f"{prefix}{loc}: {self.message} [{self.category}]"


class UnsupportedFeatureError(Exception):
    """Raised when the analyzed function uses features outside the supported subset."""

    def __init__(self, violations: list[Violation]) -> None:
        self.violations = violations
        super().__init__("\n".join(str(v) for v in violations))


# ---------------------------------------------------------------------------
# Rejected names
#
# Each set below lists names the analysis cannot model.  The comments
# say WHY in terms a user can act on, not which internal category they
# fall into.
# ---------------------------------------------------------------------------

# Builtins whose behaviour depends on runtime values that no stub can
# describe.  The analysis would silently ignore their effects.
REJECTED_BUILTINS = frozenset(
    {
        "eval",  # executes arbitrary code
        "exec",  # executes arbitrary code
        "compile",  # creates code objects
        "setattr",  # sets an attribute chosen at runtime
        "delattr",  # deletes an attribute chosen at runtime
        "globals",  # returns the live global namespace as a dict
        "locals",  # returns the live local namespace as a dict
        "vars",  # returns an object's live namespace as a dict
        "__import__",  # imports a module chosen at runtime
        "breakpoint",  # enters the debugger
    }
)

# Accessing these fields lets code inspect or replace object internals
# (the attribute dict, the class pointer, the function body, etc.)
# in ways the analysis does not track.
REJECTED_FIELDS = frozenset(
    {
        "__dict__",  # raw attribute dictionary
        "__class__",  # runtime type pointer
        "__bases__",  # base class tuple
        "__mro__",  # method resolution order
        "__code__",  # function bytecode
        "__closure__",  # closure cell tuple
        "__func__",  # underlying function of a bound method
        "__defaults__",  # default argument values
        "__kwdefaults__",  # default keyword argument values
        "__globals__",  # function's global namespace
        "__builtins__",  # builtins namespace
        "__subclasses__",  # live subclass list
    }
)

# Modules that introduce concurrency, native code, serialization, or
# interpreter introspection.  Importing one at module level is a WARNING
# (the module may not use it in the analyzed function); importing inside
# the function is also a WARNING since the import itself is not harmful
# but suggests the function may be doing something the analysis cannot model.
REJECTED_MODULES = frozenset(
    {
        "threading",  # threads
        "multiprocessing",  # processes
        "concurrent",  # thread/process pools
        "signal",  # signal handlers
        "asyncio",  # async event loop
        "pickle",  # arbitrary object reconstruction
        "shelve",  # pickle-based storage
        "copyreg",  # pickle customization
        "marshal",  # low-level serialization
        "ctypes",  # C function calls
        "cffi",  # C function calls
        "sys",  # interpreter internals
        "inspect",  # frame and code inspection
        "gc",  # garbage collector control
        "weakref",  # pointers with GC callbacks
        "importlib",  # import system mutation
    }
)

# ---------------------------------------------------------------------------
# Function-level checks (ERROR unless noted)
# ---------------------------------------------------------------------------


def check_function(cfg: Cfg[tac.Tac]) -> list[Violation]:
    """Check a function's TAC for features outside the supported subset."""
    violations: list[Violation] = []

    for label, block in cfg.items():
        for index, ins in block.items():
            location = (label, index)
            _check_instruction(ins, location, violations)

    return violations


def _check_instruction(
    ins: tac.Tac, location: Location, violations: list[Violation]
) -> None:
    match ins:
        case tac.Unsupported(name=name):
            violations.append(
                Violation(
                    location,
                    "unsupported_bytecode",
                    f"Unsupported bytecode: {name}",
                    Severity.ERROR,
                )
            )

        case tac.Assign(lhs=lhs, expr=expr):
            _check_expr(expr, location, violations)
            _check_signature(lhs, location, violations)

        case _:
            pass


def _check_expr(
    expr: tac.Expr, location: Location, violations: list[Violation]
) -> None:
    match expr:
        case tac.Attribute(
            var=tac.PredefinedScope.GLOBALS, field=tac.Var(name=name)
        ) if name in REJECTED_BUILTINS:
            violations.append(
                Violation(
                    location,
                    "rejected_builtin",
                    f"{name}() cannot be analyzed — "
                    f"{_why_rejected_builtin(name)}",
                    Severity.ERROR,
                )
            )

        case tac.Attribute(
            var=tac.Var(), field=tac.Var(name=name)
        ) if name in REJECTED_FIELDS:
            violations.append(
                Violation(
                    location,
                    "rejected_field",
                    f"Access to {name} is not supported — "
                    f"the analysis does not track object internals",
                    Severity.ERROR,
                )
            )

        case tac.Yield():
            violations.append(
                Violation(
                    location,
                    "yield",
                    "yield is not supported — "
                    "the analysis requires non-suspendable control flow",
                    Severity.ERROR,
                )
            )

        case tac.Import():
            modname = _extract_module_name(expr)
            if modname in REJECTED_MODULES:
                violations.append(
                    Violation(
                        location,
                        "rejected_import",
                        f"import {modname} — "
                        f"this module is outside the analysis model",
                        Severity.WARNING,
                    )
                )

        case tac.MakeFunction(free_var_cells=cells) if cells is not None:
            violations.append(
                Violation(
                    location,
                    "closure_creation",
                    "Nested function captures enclosing variables; "
                    "its body is not analyzed — "
                    "ensure it does not mutate shared state",
                    Severity.WARNING,
                )
            )

        case _:
            pass


def _check_signature(
    sig: tac.Signature, location: Location, violations: list[Violation]
) -> None:
    match sig:
        case tac.Attribute(
            var=tac.PredefinedScope.NONLOCALS, field=tac.Var(name=name)
        ):
            violations.append(
                Violation(
                    location,
                    "nonlocal_mutation",
                    f"Writing to nonlocal variable '{name}' is not supported — "
                    f"closures must be read-only",
                    Severity.ERROR,
                )
            )

        case tac.Attribute(
            var=tac.Var(), field=tac.Var(name=name)
        ) if name in REJECTED_FIELDS:
            violations.append(
                Violation(
                    location,
                    "rejected_field",
                    f"Store to {name} is not supported — "
                    f"the analysis does not track object internals",
                    Severity.ERROR,
                )
            )

        case tuple() as elements:
            for element in elements:
                _check_signature(element, location, violations)

        case _:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _why_rejected_builtin(name: str) -> str:
    reasons = {
        "eval": "it executes arbitrary code",
        "exec": "it executes arbitrary code",
        "compile": "it creates code objects at runtime",
        "setattr": "it sets an attribute chosen at runtime",
        "delattr": "it deletes an attribute chosen at runtime",
        "globals": "it exposes the live namespace as a mutable dict",
        "locals": "it exposes the live namespace as a mutable dict",
        "vars": "it exposes the live namespace as a mutable dict",
        "__import__": "it imports a module chosen at runtime",
        "breakpoint": "it enters the debugger",
    }
    return reasons.get(name, "its effects cannot be statically described")


def _extract_module_name(imp: tac.Import) -> str:
    match imp.modname:
        case tac.Var(name=name):
            return name
        case tac.Attribute(var=tac.Var(name=name)):
            return name
        case _:
            return ""


# ---------------------------------------------------------------------------
# Module-level checks (WARNING only)
# ---------------------------------------------------------------------------


def check_module_imports(imports: dict[str, str]) -> list[Violation]:
    """Check module-level imports for modules outside the analysis model.

    Args:
        imports: The imports dict from ParsedFile (alias -> qualified name).
    """
    violations: list[Violation] = []

    for alias, qualified_name in imports.items():
        root = qualified_name.split(".")[0]
        if root in REJECTED_MODULES:
            violations.append(
                Violation(
                    None,
                    "rejected_import",
                    f"Module imports {qualified_name} — "
                    f"this module is outside the analysis model",
                    Severity.WARNING,
                )
            )

    return violations


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def validate(
    cfg: Cfg[tac.Tac],
    function_name: str,
    module_imports: dict[str, str] | None = None,
) -> None:
    """Run all validation checks. Raises UnsupportedFeatureError on errors.

    Warnings are emitted via the warnings module.
    """
    violations: list[Violation] = []

    if module_imports is not None:
        violations.extend(check_module_imports(module_imports))

    violations.extend(check_function(cfg))

    for v in violations:
        if v.severity == Severity.WARNING:
            warnings.warn(f"[{function_name}] {v}", stacklevel=2)

    errors = [v for v in violations if v.severity == Severity.ERROR]
    if errors:
        raise UnsupportedFeatureError(errors)
