# Python dynamicity vs. Pythia's heap analysis

## Supported subset

The analysis works on ordinary Python functions that use standard
operations on types it knows about.

**Types:** `int`, `float`, `bool`, `str`, `None`, `list`, `tuple`,
`set`, `dict`, `slice`, numpy arrays and scalars, and any type you
add to `typeshed_mini/`.

**Expressions:** arithmetic, comparisons, boolean logic, attribute
access (`obj.field`), item access (`obj[key]`, slicing), function
and method calls, container literals, unpacking (`a, b = x` and
`a, *b = x`).

**Statements:** assignment, `del`, `return`, `raise`, `if`/`elif`/
`else`, `for`, `while`, `import` (of modules with stubs), function
definitions, closures that read enclosing variables.

**Not supported:** `yield`/generators, `async`/`await`,
`try`/`except`/`finally`, `with` (context managers),
`*args`/`**kwargs` forwarding, closures that write to enclosing
variables.

**Rejected builtins:** `eval`, `exec`, `compile`.

**Rejected field access:** `obj.__dict__`, `obj.__class__`,
`obj.__bases__`, and other internal-state dunders.

**Not yet supported (need stubs):** `setattr`/`getattr`/`delattr`
(equivalent to `obj.field` access when the name is a constant
string), `globals()`/`locals()` (safe for reading, not mutation).

### Type requirements

Every type your function touches must have a stub in
`typeshed_mini/`. If a type is missing, the analysis will fail with
an assertion error when it tries to resolve a method or attribute.

Currently stubbed: `builtins`, `collections`, `numpy`, `typing`.
To add support for a new library, add a `.pyi` file to
`typeshed_mini/`.

### Effect annotations

Function stubs must declare their heap effects (`@new`, `@update`,
`@alias`, `@accessor`). A function without effect annotations is
assumed to have no side effects. If that is wrong, the analysis
will miss mutations and report incorrect dirty sets.

### Enforcement

A validation pass (`pythia/validate.py`) runs after TAC translation
and before analysis. It rejects unsupported features with clear
error messages and warns about suspicious imports.

Features not covered by the validation pass (descriptor
interception, metaclasses, monkey patching from outside the
function) are the user's responsibility — the analysis assumes
stubs accurately describe the types in use.


---


## Detailed feature assessment

The rest of this document records, for each category of Python
dynamicity, how the analysis handles it. Each feature is one of:

- **Precise** — modeled correctly.
- **Sound** — over-approximates but never misses a mutation.
- **Rejected** — validation pass stops with a clear message.
- **Detected** — crashes with an assertion (traceable but not friendly).
- **Silent break** — completes but may miss mutations.


### A. Runtime code creation

| Feature | Status |
|---------|--------|
| `eval`, `exec`, `compile` | **Rejected** — unstubbable builtins. |
| `function.__code__` assignment | **Rejected** — object internal. |
| Indirect execution via parameters | **Silent break** — the call's effects are whatever the stub says. |


### B. Namespace / scope mutation

| Feature | Status |
|---------|--------|
| `globals()`, `locals()`, `vars()` | **Not yet supported** — needs stubs. Supportable: reading is safe; mutation of the returned dict is not. |
| `sys._getframe`, frame access | **Detected** — `sys` has no stub. |
| Dict mutation connected to namespace | **Silent break** — tracked as dict mutation, not as rebinding. |


### C. Function / closure mutation

| Feature | Status |
|---------|--------|
| `__defaults__`, `__kwdefaults__`, `__closure__` access | **Rejected** — object internals. |
| Nonlocal writes (`STORE_DEREF`) | **Rejected** — pure closures only. |
| Nonlocal reads (`LOAD_DEREF`) | Allowed (pure closure assumption). |
| `function.__dict__` mutation | **Silent break** — attribute store tracked, but semantics not understood. |
| Rebinding a function name | **Sound** — pointer set updated, but may fail to resolve new callable's type. |
| Decorators replacing identity | **Detected** — decorator return type must be in stubs. |


### D. Monkey patching

| Feature | Status |
|---------|--------|
| `instance.attr = ...` within function | **Precise** — pointer update, dirty marking, type tracking. |
| `module.attr = ...` or `Class.attr = ...` | **Silent break** — stubs are authoritative; runtime changes invisible. |
| Patching before the analyzed function runs | **Silent break** — same reason. |


### E. Object state / attribute-layout mutation

| Feature | Status |
|---------|--------|
| `setattr`, `getattr`, `delattr` | **Not yet supported** — needs stubs. Supportable: equivalent to `obj.field` access when the attribute name is a constant string. |
| `obj.__dict__` | **Rejected** — object internal. |
| `__slots__` interactions | **Silent break** — not modeled. |


### F. Attribute-access interception

| Feature | Status |
|---------|--------|
| `__getattribute__`, `__getattr__`, `__setattr__` | **Silent break** — attribute access resolved statically via stubs. |
| Custom descriptors (`__get__`, `__set__`) | **Silent break** — not modeled. |
| `property` (declared in stubs) | **Precise** — `is_property` flag, `@new`/`@accessor` effects applied. |
| `classmethod`, `staticmethod` | Handled if stub declares them correctly; otherwise **silent break**. |


### G. Callable / construction interception

| Feature | Status |
|---------|--------|
| `__init__` (declared in stubs) | **Precise** — `get_init_func()`, side effects, object creation. |
| `__new__` returning cached/different object | **Silent break** — not modeled. |
| `__call__` on arbitrary objects | **Silent break** — resolved by type, not by checking `__call__`. |
| Object factories returning cached objects | **Silent break** — `@new` assumes fresh allocation. |


### H. Protocol dispatch (dunders behind syntax)

| Feature | Status |
|---------|--------|
| `__getitem__`, `__setitem__` | **Precise** (stub-declared). |
| `__iter__`, `__next__` | **Precise** (stub-declared). |
| Arithmetic, comparison, containment dunders | **Precise** (stub-declared). |
| `__bool__` | **Precise** (stub-declared). |
| `yield` | **Rejected** — breaks sequential control flow. |
| `async`/`await`, `__aiter__`/`__anext__` | **Detected** — bytecodes not handled. |
| `with` / context managers | **Detected** — bytecodes not handled. |
| Pattern matching | **Detected** — bytecodes not handled. |
| User types without stubs | **Detected** — assertion on missing type. |


### I. Class / type metaprogramming

| Feature | Status |
|---------|--------|
| `__class__`, `__bases__` access | **Rejected** — object internals. |
| `type(name, bases, ns)` | **Detected** — stub expects `type[T]` syntax. |
| Class decorators | **Detected** — return type must be in stubs. |
| Metaclasses, `__prepare__` | **Silent break** — analysis uses `get_init_func()` directly. |
| `__mro_entries__`, ABC registration | **Silent break** — not modeled. |


### J. Module / import-system mutation

| Feature | Status |
|---------|--------|
| `import x` (stubbed module) | **Precise**. |
| `import x` (unstubbed module) | **Detected** — resolves to BOTTOM. |
| `__import__` | **Rejected** — unstubbable builtin. |
| `importlib` | **Detected** — no stub. |
| `threading`, `signal`, `asyncio`, `pickle`, etc. | **Warning** — imports flagged by validation pass. |
| `sys.modules`, module `__getattr__` | **Silent break** — not modeled. |


### K. Serialization

| Feature | Status |
|---------|--------|
| `pickle`, `shelve`, `copyreg` | **Detected** — no stubs. |
| `__reduce__`, `__getstate__`, `__setstate__` | **Silent break** — never invoked by analysis. |


### L. Finalization / GC

| Feature | Status |
|---------|--------|
| `__del__` | **Silent break** — never invoked. |
| Weakref callbacks | **Silent break** — `weakref` has no stub. |
| Generator/coroutine cleanup | **Silent break** — no state machine. |


### M. Async scheduling

| Feature | Status |
|---------|--------|
| `async def` / `await` | **Detected** — bytecodes not handled. |
| Callback registration | **Silent break** — scheduling semantics not modeled. |


### N. Threads / signals

| Feature | Status |
|---------|--------|
| `threading`, `signal` imports | **Warning** from validation pass. |
| Concurrent mutation | **Silent break** — single-threaded assumption. |


### O. Interpreter hooks

| Feature | Status |
|---------|--------|
| `sys.settrace`, `sys.setprofile` | **Detected** — `sys` has no stub. |
| `breakpoint` | **Rejected** — unstubbable builtin. |


### P. Builtins mutation

| Feature | Status |
|---------|--------|
| `__builtins__` access | **Rejected** — object internal. |
| Rebinding builtins in global scope | **Silent break** — stubs are authoritative. |


### Q. Native code

| Feature | Status |
|---------|--------|
| Extension modules with stubs (numpy) | **Precise**. |
| `ctypes`, `cffi` | **Warning** + **detected** (no stub). |
| Unstubbed extension modules | **Detected** — resolves to BOTTOM. |


### R. Implementation-dependent behavior

All **silent break**. The analysis operates on CPython 3.12-3.14
bytecode and does not model interpreter-specific behavior.


---


## Remaining silent breaks

These are features that the validation pass cannot catch because
they happen outside the analyzed function's bytecode:

- **Monkey patching** (D): module/class attributes changed before
  the function runs.
- **Descriptor interception** (F): `__getattr__`, `__setattr__`,
  custom descriptors on types the function uses.
- **Construction interception** (G): `__new__`, `__call__` on types
  the function constructs.
- **Metaclasses** (I): custom metaclass behavior during class
  creation.
- **Finalization** (L): `__del__`, weakref callbacks, generator
  cleanup.
- **Callback semantics** (M): event-loop callbacks invoked later.
- **Concurrent mutation** (N): thread/signal interleaving.
- **Builtin rebinding** (P): `int = str` at module level.

These are all cross-boundary effects. The analysis is sound under
the assumption that stubs accurately describe the types the function
uses and that no external mutation changes their behavior.


---


## Comparison with Numba

Both Pythia and Numba restrict Python to a static subset. The
approaches differ:

**Subset enforcement.** Both define a supported subset and reject
code outside it. Numba uses type inference failure; Pythia uses a
validation pass plus stub-miss assertions.

**Object model.** Numba replaces Python's object model with C
structs, making categories like descriptors, `__dict__`, and
metaclasses structurally impossible. Pythia analyses the Python
object model but trusts stubs to describe it accurately.

**Closures.** Numba captures by value (documented semantic change).
Pythia assumes pure closures: read-only nonlocal access, writes
rejected.

**Type coverage.** Numba infers types from runtime values at first
call. Pythia reads types from static stubs. A missing stub causes an
assertion crash (detected, but with an unhelpful error message).

**Native code.** Numba compiles to native code and has first-class
ctypes/cffi interop. Pythia treats native code as out of scope,
relying on stubs (currently only numpy).

**Threading.** Numba has `prange` and releases the GIL with a sound
memory model for shared arrays. Pythia assumes single-threaded
execution.

**Residual silent breaks.** Numba's main risk is monkey patching
between compilations. Pythia's remaining risks are monkey patching,
descriptor interception, construction interception, metaclasses,
finalization callbacks, and concurrent mutation — all cross-boundary
effects invisible from the function's bytecode.
