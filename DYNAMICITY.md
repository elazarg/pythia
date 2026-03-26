  Assessment: Python Features vs. Pythia's Heap Analysis

  Rating scale per feature:
  - Precise — handled correctly with useful precision
  - Sound/conservative — over-approximates (marks too much dirty) but never misses a real mutation
  - Rejected — explicitly rejected by the validation pass (validate.py) with a clear message
  - Detected — analysis crashes (assertion/NotImplementedError) with a traceable error
  - Silent break — unsound without warning; analysis may miss real heap mutations

  The analysis is intraprocedural (single function), relies on static type stubs (typeshed_mini/), models side effects via explicit annotations
  (@update, @new, @alias, @accessor), and targets annotated for-loops within function bodies.

  A validation pass (pythia/validate.py) runs after TAC translation and before analysis, checking that the function stays within the supported
  subset. See the module docstring for the programmer-facing description of what is and isn't supported.

  ---
  A. Runtime code creation / execution

  ┌───────────────────────────────┬───────────┬──────────────────────────────────────────────────────────────────────────────────────────────┐
  │            Feature            │  Verdict  │                                         Explanation                                          │
  ├───────────────────────────────┼───────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤
  │ eval / exec / compile         │ Rejected  │ Validation pass rejects these as unstubbable builtins: their effects depend on runtime values  │
  │                               │           │ that no stub can describe                                                                     │
  ├───────────────────────────────┼───────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Execution of dynamically      │ Silent    │ If a code object is produced and invoked through a chain the analysis can resolve (e.g., as  │
  │ produced code objects         │ break     │ a parameter), the call's side effects are whatever the type stub says — typically nothing    │
  ├───────────────────────────────┼───────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Assignment to                 │ Rejected  │ Validation pass rejects access to __code__ — it is an object internal whose mutation the     │
  │ function.__code__             │           │ analysis cannot track                                                                        │
  └───────────────────────────────┴───────────┴──────────────────────────────────────────────────────────────────────────────────────────────┘

  Status: eval/exec/compile are rejected by the validation pass. __code__ access is rejected as a dangerous field.

  ---
  B. Namespace / scope mutation

  ┌───────────────────────────────────┬───────────┬──────────────────────────────────────────────────────────────────────────────────────────┐
  │              Feature              │  Verdict  │                                       Explanation                                        │
  ├───────────────────────────────────┼───────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ globals() / locals() / vars() as  │ Rejected  │ Validation pass rejects these — they expose the live namespace as a mutable dict            │
  │ calls                             │           │                                                                                          │
  ├───────────────────────────────────┼───────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ Mutation of returned dict after   │ Silent    │ If the call were somehow resolved, the returned dict is a regular dict object; mutations │
  │ globals()                         │ break     │  to it are tracked as dict mutations but not connected back to the actual namespace      │
  ├───────────────────────────────────┼───────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ sys._getframe / frame access /    │ Detected  │ sys module not in typeshed_mini → attribute resolution fails                             │
  │ f_locals / f_globals              │           │                                                                                          │
  ├───────────────────────────────────┼───────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ Reflective rebinding through      │ Silent    │ The analysis doesn't connect dict mutations to name rebindings                           │
  │ mappings                          │ break     │                                                                                          │
  ├───────────────────────────────────┼───────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ Traceback-to-frame access         │ Silent    │ Same as frame access; if reachable without module lookup, unmodeled                      │
  │                                   │ break     │                                                                                          │
  └───────────────────────────────────┴───────────┴──────────────────────────────────────────────────────────────────────────────────────────┘

  Status: globals()/locals()/vars() rejected by validation pass. sys not in typeshed_mini, so frame access crashes at analysis time.

  ---
  C. Function / closure mutation

  ┌──────────────────────────────┬────────────────────┬───────────────────────────────────────────────────────────────────────────────────────┐
  │           Feature            │      Verdict       │                                      Explanation                                      │
  ├──────────────────────────────┼────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ Assignment to function       │ Silent break       │ STORE_ATTR recorded, but doesn't affect call resolution                               │
  │ attributes (f.x = ...)       │                    │                                                                                       │
  ├──────────────────────────────┼────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ Replacement of __defaults__  │ Rejected           │ Validation pass rejects access to __defaults__ and __kwdefaults__ as object internals │
  │ / __kwdefaults__             │                    │                                                                                       │
  ├──────────────────────────────┼────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ cell.cell_contents mutation  │ Silent break       │ LOAD_CLOSURE is hardcoded to return Const(None) (tac.py:797). Assumption: pure        │
  │                              │                    │ closures — read-only access to enclosing variables only. STORE_DEREF crashes at TAC   │
  │                              │                    │ translation; validation pass also rejects nonlocal writes as belt-and-suspenders       │
  ├──────────────────────────────┼────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ Rebinding function name to   │                    │ If done within the analyzed function, the variable now points to the new callable.    │
  │ different callable           │ Sound/conservative │ The analysis tracks the variable's pointer set but may fail to resolve the new        │
  │                              │                    │ callable's type → assertion or TOP result                                             │
  ├──────────────────────────────┼────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ Decorators replacing         │ Detected           │ At bytecode level, decorator application is CALL. If decorator return type isn't in   │
  │ function identity            │                    │ stubs → assertion failure                                                             │
  └──────────────────────────────┴────────────────────┴───────────────────────────────────────────────────────────────────────────────────────┘

  Status: __defaults__, __code__, __closure__ access rejected by validation pass. Nonlocal writes rejected.
  Closure assumption documented: pure closures (read-only nonlocal access) are the supported mode.

  ---
  D. Ordinary monkey patching

  ┌─────────────────────────────────┬───────────┬─────────────────────────────────────────────────────────────────────────────────────────────┐
  │             Feature             │  Verdict  │                                         Explanation                                         │
  ├─────────────────────────────────┼───────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤
  │ module.attr = ...               │ Silent    │ The analysis uses static type stubs from typeshed_mini/. Any runtime mutation to module     │
  │                                 │ break     │ attributes is invisible — the type system always returns the stub-declared type             │
  ├─────────────────────────────────┼───────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Class.attr = ...                │ Silent    │ Same: class types come from stubs. Runtime mutation not reflected                           │
  │                                 │ break     │                                                                                             │
  ├─────────────────────────────────┼───────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤
  │ instance.attr = ... (within     │ Precise   │ STORE_ATTR is fully handled. The analysis records the pointer update, marks the field       │
  │ analyzed function)              │           │ dirty, and tracks the new type. Strong update for singletons, weak update otherwise         │
  ├─────────────────────────────────┼───────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Patching after                  │ Silent    │ Happens before/outside the analyzed function; type stubs are the sole source of truth       │
  │ import/definition               │ break     │                                                                                             │
  └─────────────────────────────────┴───────────┴─────────────────────────────────────────────────────────────────────────────────────────────┘

  This is one of the most practically relevant categories. The analysis is sound only if type stubs accurately describe runtime behavior. Any
  monkey patching that changes attribute types or function signatures will silently violate that assumption.

  Detectability: Detecting monkey patching in general is undecidable, but patching of known modules/classes within the analyzed function's
  bytecode can be flagged.

  ---
  E. Object state / attribute-layout mutation

  ┌────────────────────┬────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │      Feature       │  Verdict   │                                              Explanation                                               │
  ├────────────────────┼────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ setattr(obj, name, │ Rejected   │ Validation pass rejects setattr — it sets an attribute chosen at runtime                                │
  │  val)              │            │                                                                                                         │
  ├────────────────────┼────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ getattr(obj, name) │ Detected   │ Not in builtins.pyi stubs → assertion on call resolution                                               │
  ├────────────────────┼────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ delattr(obj, name) │ Rejected   │ Validation pass rejects delattr — it deletes an attribute chosen at runtime                             │
  ├────────────────────┼────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Computed attribute │ N/A        │ LOAD_ATTR/STORE_ATTR always carry a constant attribute name from bytecode. Computed names go through   │
  │  names             │            │ getattr/setattr calls                                                                                  │
  ├────────────────────┼────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ obj.__dict__       │ Rejected   │ Validation pass rejects access to __dict__ — it bypasses the attribute model                            │
  │ access             │            │                                                                                                        │
  ├────────────────────┼────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ __slots__          │ Silent     │ The analysis doesn't model __slots__. Type stubs declare fields; whether __slots__ restricts them is   │
  │ interactions       │ break      │ not checked                                                                                            │
  └────────────────────┴────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  Status: setattr/delattr rejected by validation pass. __dict__ access rejected. getattr crashes at analysis time (no stub).

  ---
  F. Attribute-access interception and descriptor machinery

  ┌───────────────────────────┬───────────────────┬──────────────────────────────────────────────────────────────────────────────────────────┐
  │          Feature          │      Verdict      │                                       Explanation                                        │
  ├───────────────────────────┼───────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ __getattribute__ /        │ Silent break      │ Attribute loads go through the type system's static lookup, not the descriptor protocol. │
  │ __getattr__               │                   │  Custom interception methods are never invoked by the analysis                           │
  ├───────────────────────────┼───────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ __setattr__ / __delattr__ │ Silent break      │ Attribute stores go directly to pointer/type update. Custom setattr never invoked        │
  ├───────────────────────────┼───────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ Descriptors (__get__,     │ Silent break      │ Not modeled. The analysis treats all attributes as plain data fields                     │
  │ __set__, __delete__)      │                   │                                                                                          │
  ├───────────────────────────┼───────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │                           │ Precise           │ The type system checks is_property flag. Properties declared in stubs are handled: the   │
  │ property                  │ (stub-declared)   │ getter's return type is used, @new and @accessor side effects are applied. But only      │
  │                           │                   │ works for types in typeshed_mini                                                         │
  ├───────────────────────────┼───────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ classmethod /             │ Partially handled │ Method binding (bind_self_function) skips self for static methods if the stub declares   │
  │ staticmethod              │                   │ them correctly. Otherwise silent break                                                   │
  ├───────────────────────────┼───────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ __set_name__              │ Silent break      │ Not modeled                                                                              │
  └───────────────────────────┴───────────────────┴──────────────────────────────────────────────────────────────────────────────────────────┘

  Detectability: User-defined __getattr__/__setattr__ etc. in classes being analyzed could be detected by scanning class bodies for these dunder
  definitions. However, detection in third-party libraries requires stub annotation.

  ---
  G. Callable / construction interception

  ┌───────────────────────────────┬───────────────────┬───────────────────────────────────────────────────────────────────────────────────────┐
  │            Feature            │      Verdict      │                                      Explanation                                      │
  ├───────────────────────────────┼───────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │                               │                   │ If an object has __call__ but its type stub declares it as a regular class (not       │
  │ __call__                      │ Silent break      │ Overloaded), calling it will fail or produce wrong results. The analysis resolves     │
  │                               │                   │ calls by type, not by checking __call__ at runtime                                    │
  ├───────────────────────────────┼───────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ __new__                       │ Silent break      │ Not explicitly handled. get_init_func() extracts __init__ but not __new__. A custom   │
  │                               │                   │ __new__ returning a cached or different-class object is not modeled                   │
  ├───────────────────────────────┼───────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ __init__                      │ Precise           │ Handled via get_init_func(). The return type, side effects, and new-object creation   │
  │                               │ (stub-declared)   │ are modeled per stub annotations                                                      │
  ├───────────────────────────────┼───────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ __init_subclass__             │ Silent break      │ Not modeled; triggered during class creation, which the analysis doesn't deeply       │
  │                               │                   │ inspect                                                                               │
  ├───────────────────────────────┼───────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ Object factories returning    │                   │ The analysis assumes @new means a fresh object. A factory returning a cached object   │
  │ cached/unrelated objects      │ Silent break      │ would violate that assumption — the analysis would assign it a unique abstract        │
  │                               │                   │ location                                                                              │
  └───────────────────────────────┴───────────────────┴───────────────────────────────────────────────────────────────────────────────────────┘

  Detectability: Custom __new__ and __call__ can be detected in analyzed class bodies. Factory caching patterns are harder to detect statically.

  ---
  H. Protocol dispatch hidden behind ordinary syntax

  ┌──────────────────────────┬────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────┐
  │         Feature          │      Verdict       │                                       Explanation                                        │
  ├──────────────────────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ __getitem__ /            │ Precise            │ BINARY_SUBSCR and STORE_SUBSCR are fully translated. Type resolution goes through        │
  │ __setitem__ /            │ (stub-declared)    │ ts.subscr() which looks up the dunder in stubs. @accessor and @new side effects are      │
  │ __delitem__              │                    │ applied                                                                                  │
  ├──────────────────────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ __iter__ / __next__      │ Precise            │ GET_ITER → UnOp.ITER, FOR_ITER → For instruction. Iterator type resolved via stubs       │
  │                          │ (stub-declared)    │                                                                                          │
  ├──────────────────────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ __aiter__ / __anext__    │ Detected           │ Async bytecodes not handled → NotImplementedError                                        │
  ├──────────────────────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ __enter__ / __exit__     │ Detected           │ with statement bytecodes (BEFORE_WITH, SETUP_WITH) not in the match cases →              │
  │ (context managers)       │                    │ NotImplementedError                                                                      │
  ├──────────────────────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ __aenter__ / __aexit__   │ Detected           │ Async, not supported                                                                     │
  ├──────────────────────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ Arithmetic / comparison  │ Precise            │ BINARY_OP resolved via binop_to_dunder_method() → type system's get_binop(). Both        │
  │ dunders                  │ (stub-declared)    │ forward (__add__) and reverse (__radd__) methods checked                                 │
  ├──────────────────────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ Truthiness (__bool__)    │ Precise            │ TO_BOOL → UnOp.BOOL → resolved to __bool__ via stubs                                     │
  ├──────────────────────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ Containment              │ Precise            │ CONTAINS_OP → Binary with in op → resolved to __contains__                               │
  │ (__contains__)           │                    │                                                                                          │
  ├──────────────────────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ Hashing / equality       │ Precise            │ __eq__, __hash__ resolved through comparison/binary dunder dispatch                      │
  ├──────────────────────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ Pattern matching hooks   │ N/A                │ Pattern matching bytecodes not supported (would hit NotImplementedError)                 │
  └──────────────────────────┴────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────┘

  This is the strongest category. For types with stubs, protocol dispatch through syntax is handled well. The gap is user-defined types not in
  stubs — their dunders won't be in the type system, leading to assertion failures or BOTTOM.

  ---
  I. Class / type metaprogramming

  ┌─────────────────────────┬───────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │         Feature         │  Verdict  │                                             Explanation                                             │
  ├─────────────────────────┼───────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ type(name, bases, ns)   │ Detected  │ Three-argument type() would be resolved against builtins.pyi's type definition. The stub likely     │
  │                         │           │ expects type[T] constructor syntax, not 3-arg dynamic form → assertion or wrong result              │
  ├─────────────────────────┼───────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │                         │ Silent    │ The type system creates an implicit metaclass (type_system.py:1915) for class definitions, but this │
  │ Metaclasses             │ break     │  only affects stub parsing. At analysis time, metaclass __call__ is not invoked — the analysis uses │
  │                         │           │  get_init_func() directly. Custom metaclass behavior is invisible                                   │
  ├─────────────────────────┼───────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ __prepare__ / metaclass │ Silent    │ Not modeled                                                                                         │
  │  __new__ / __init__     │ break     │                                                                                                     │
  ├─────────────────────────┼───────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Class decorators        │ Detected  │ Decorator application is a CALL. If the decorator's return type isn't an Overloaded → assertion     │
  │                         │           │ failure                                                                                             │
  ├─────────────────────────┼───────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Mutation of __class__ / │ Rejected  │ Validation pass rejects access to __class__ and __bases__ as object internals                       │
  │  __bases__              │           │                                                                                                     │
  ├─────────────────────────┼───────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ __mro_entries__         │ Silent    │ Not modeled                                                                                         │
  │                         │ break     │                                                                                                     │
  ├─────────────────────────┼───────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ ABC virtual subclass    │ Silent    │ isinstance/issubclass not modeled in the analysis (they're just function calls)                     │
  │ registration            │ break     │                                                                                                     │
  └─────────────────────────┴───────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────┘

  Detectability: Three-arg type() call detectable by argument count. Class decorators already crash. Metaclass usage detectable in class AST.
  __class__ assignment detectable via STORE_ATTR("__class__").

  ---
  J. Module / import-system mutation

  ┌─────────────────────────────────────┬─────────────┬───────────────────────────────────────────────────────────────────────────────────────┐
  │               Feature               │   Verdict   │                                      Explanation                                      │
  ├─────────────────────────────────────┼─────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ import x (static, known module)     │ Precise     │ IMPORT_NAME → Import TAC. Module resolved via typeshed_mini stubs if available        │
  ├─────────────────────────────────────┼─────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ import x (unknown module)           │ Detected    │ Module not in typeshed_mini → resolution returns BOTTOM → subsequent attribute access │
  │                                     │             │  fails                                                                                │
  ├─────────────────────────────────────┼─────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ importlib.import_module(name)       │ Detected    │ importlib not in typeshed_mini → assertion failure                                    │
  ├─────────────────────────────────────┼─────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ __import__                          │ Rejected    │ Validation pass rejects __import__ — it imports a module chosen at runtime             │
  ├─────────────────────────────────────┼─────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ Custom loaders/finders,             │ Silent      │ sys not in stubs; if somehow reached, modifications not tracked                       │
  │ sys.meta_path                       │ break       │                                                                                       │
  ├─────────────────────────────────────┼─────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ sys.modules mutation                │ Silent      │ Same                                                                                  │
  │                                     │ break       │                                                                                       │
  ├─────────────────────────────────────┼─────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ Module __getattr__                  │ Silent      │ Module attribute access goes through type system's static lookup. Module-level        │
  │                                     │ break       │ __getattr__ is never invoked                                                          │
  ├─────────────────────────────────────┼─────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ module.__class__ mutation           │ Silent      │ Not modeled                                                                           │
  │                                     │ break       │                                                                                       │
  └─────────────────────────────────────┴─────────────┴───────────────────────────────────────────────────────────────────────────────────────┘

  The analysis's reliance on static stubs makes it inherently resistant to import-system games — but also unable to analyze code that uses
  non-stubbed modules.

  ---
  K. Serialization / reconstruction channels

  ┌───────────────────────────────────┬───────────┬──────────────────────────────────────────────────────────────────────────────────────────┐
  │              Feature              │  Verdict  │                                       Explanation                                        │
  ├───────────────────────────────────┼───────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ pickle.loads / pickle.dumps       │ Detected  │ pickle not in typeshed_mini → module resolution fails                                    │
  ├───────────────────────────────────┼───────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ __reduce__ / __reduce_ex__ /      │ Silent    │ These are dunder methods that the analysis never invokes. If they perform heap mutations │
  │ __getstate__ / __setstate__       │ break     │  during serialization/deserialization, those mutations are invisible                     │
  ├───────────────────────────────────┼───────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ copyreg / shelve                  │ Detected  │ Not in stubs                                                                             │
  └───────────────────────────────────┴───────────┴──────────────────────────────────────────────────────────────────────────────────────────┘

  Detectability: All serialization modules can be detected by scanning imports. The dunder methods could be detected in class bodies.

  ---
  L. Finalization / GC-triggered execution

  ┌───────────────────────────────────┬─────────────┬────────────────────────────────────────────────────────────────────────────────────────┐
  │              Feature              │   Verdict   │                                      Explanation                                       │
  ├───────────────────────────────────┼─────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ __del__                           │ Silent      │ Never invoked by the analysis. GC-triggered finalizer mutations are invisible          │
  │                                   │ break       │                                                                                        │
  ├───────────────────────────────────┼─────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ Weakref callbacks /               │ Silent      │ weakref not in stubs. Even if resolved, callbacks aren't modeled                       │
  │ weakref.finalize                  │ break       │                                                                                        │
  ├───────────────────────────────────┼─────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ Generator/coroutine cleanup       │ Silent      │ Generator state machines aren't modeled. Cleanup code in finally blocks of generators  │
  │                                   │ break       │ is invisible                                                                           │
  ├───────────────────────────────────┼─────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ Object resurrection               │ Silent      │ Not modeled                                                                            │
  │                                   │ break       │                                                                                        │
  └───────────────────────────────────┴─────────────┴────────────────────────────────────────────────────────────────────────────────────────┘

  Detectability: __del__ definitions detectable in class bodies. weakref imports detectable. Generator finally blocks detectable at AST level.

  ---
  M. Asynchronous scheduling / callback-driven execution

  ┌──────────────────────────┬────────────┬───────────────────────────────────────────────────────────────────────────────────────────────────┐
  │         Feature          │  Verdict   │                                            Explanation                                            │
  ├──────────────────────────┼────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ async def / await        │ Detected   │ Async bytecodes not in match cases → NotImplementedError                                          │
  ├──────────────────────────┼────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Event-loop scheduled     │ Silent     │ Callback registration is a regular function call; the analysis doesn't understand that the        │
  │ callbacks                │ break      │ callback will be invoked later with heap effects                                                  │
  ├──────────────────────────┼────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Task creation / future   │ Silent     │ Same — these are ordinary calls whose scheduling semantics are unmodeled                          │
  │ callbacks                │ break      │                                                                                                   │
  └──────────────────────────┴────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────┘

  Detectability: async/await already crashes. Callback registration patterns (loop.call_soon, add_done_callback) detectable by name.

  ---
  N. Threads / signals / concurrent mutation

  ┌──────────────────────────┬───────────┬────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │         Feature          │  Verdict  │                                            Explanation                                             │
  ├──────────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ threading / thread pools │ Silent    │ The analysis assumes single-threaded sequential execution. Concurrent mutations from other threads │
  │                          │ break     │  are invisible. Thread creation is a regular function call                                         │
  ├──────────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Signal handlers          │ Silent    │ signal.signal() registers a callback; the analysis doesn't model signal delivery or handler        │
  │                          │ break     │ invocation                                                                                         │
  ├──────────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Shared mutable state     │ Silent    │ Pointer/type/dirty state has no concept of concurrent access                                       │
  │ across threads           │ break     │                                                                                                    │
  └──────────────────────────┴───────────┴────────────────────────────────────────────────────────────────────────────────────────────────────┘

  Detectability: threading and signal imports detectable. Thread.start() calls detectable by type resolution.

  ---
  O. Interpreter hooks

  ┌──────────────────────────────┬─────────────┬─────────────────────────────────────────────────────────────────────────────────────────────┐
  │           Feature            │   Verdict   │                                         Explanation                                         │
  ├──────────────────────────────┼─────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤
  │ sys.settrace /               │ Detected    │ sys not in typeshed_mini → attribute access fails                                           │
  │ sys.setprofile               │             │                                                                                             │
  ├──────────────────────────────┼─────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Audit hooks                  │ Detected    │ Same                                                                                        │
  │ (sys.addaudithook)           │             │                                                                                             │
  ├──────────────────────────────┼─────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Frame-local trace hooks      │ Silent      │ If set through a path that doesn't go through sys, the analysis doesn't model trace         │
  │                              │ break       │ function invocation                                                                         │
  └──────────────────────────────┴─────────────┴─────────────────────────────────────────────────────────────────────────────────────────────┘

  Detectability: All sys.* calls detectable.

  ---
  P. Builtins / runtime-environment mutation

  ┌──────────────────────────┬───────────┬────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │         Feature          │  Verdict  │                                            Explanation                                             │
  ├──────────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Mutation of __builtins__ │ Rejected  │ Validation pass rejects access to __builtins__ as an object internal                                │
  │                          │           │                                                                                                    │
  ├──────────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Rebinding builtins in    │ Silent    │ STORE_GLOBAL to a builtin name would be tracked as a global attribute store, but the analysis      │
  │ global scope             │ break     │ resolves subsequent LOAD_GLOBAL of that name by checking the type system's module → builtins.pyi,  │
  │                          │           │ not the runtime state                                                                              │
  ├──────────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Replacing standard types │ Silent    │ Same — type stubs are authoritative                                                                │
  │  with user objects       │ break     │                                                                                                    │
  └──────────────────────────┴───────────┴────────────────────────────────────────────────────────────────────────────────────────────────────┘

  Detectability: STORE_GLOBAL with names matching builtins can be flagged. __builtins__ attribute access detectable.

  ---
  Q. Foreign-function / native-code escape

  ┌─────────────────────────────────┬──────────┬──────────────────────────────────────────────────────────────────────────────────────────────┐
  │             Feature             │ Verdict  │                                         Explanation                                          │
  ├─────────────────────────────────┼──────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤
  │ CPython extension modules       │ Detected │ Must have stubs in typeshed_mini. Currently only numpy.pyi is provided. Other extension      │
  │                                 │          │ modules → BOTTOM → assertion                                                                 │
  ├─────────────────────────────────┼──────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤
  │ ctypes / cffi                   │ Detected │ Not in typeshed_mini                                                                         │
  ├─────────────────────────────────┼──────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Cython / native wrappers        │ Detected │ Not in stubs                                                                                 │
  ├─────────────────────────────────┼──────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Raw memory access / pointer     │ N/A      │ Cannot be expressed in Python bytecode without ctypes                                        │
  │ arithmetic                      │          │                                                                                              │
  └─────────────────────────────────┴──────────┴──────────────────────────────────────────────────────────────────────────────────────────────┘

  Detectability: All native module imports detectable.

  ---
  R. Implementation-dependent runtime internals

  ┌─────────────────────────┬───────────┬────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │         Feature         │  Verdict  │                                            Explanation                                             │
  ├─────────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ CPython-only reflective │ Silent    │ The analysis doesn't distinguish CPython from other interpreters. If code relies on                │
  │  behavior               │ break     │ CPython-specific locals() behavior or GC ordering, the analysis doesn't know                       │
  ├─────────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ JIT/runtime differences │ Silent    │ The analysis operates on CPython 3.12-3.14 bytecode only (tac.py:568-571 asserts allowed           │
  │                         │ break     │ versions). Results are specific to those bytecode formats                                          │
  ├─────────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Internal attributes     │ Silent    │ If accessed via LOAD_ATTR, treated as regular attributes. Behavior differences across interpreters │
  │                         │ break     │  are invisible                                                                                     │
  └─────────────────────────┴───────────┴────────────────────────────────────────────────────────────────────────────────────────────────────┘

  Detectability: Version-specific bytecodes are already handled (version assertion). CPython-specific APIs like sys._getframe detectable by name.

  ---
  Summary Matrix

  ┌───────────────────┬──────────────────────────┬─────────────────────┬─────────────────────────────┬────────────────────────────┬────────────────────────────────┐
  │     Category      │         Precise          │ Sound/Conservative  │     Rejected (validate.py)  │        Detected (crash)    │          Silent Break          │
  ├───────────────────┼──────────────────────────┼─────────────────────┼─────────────────────────────┼────────────────────────────┼────────────────────────────────┤
  │ A. Runtime code   │ —                        │ —                   │ eval/exec/compile,          │ —                          │ indirect execution via         │
  │ creation          │                          │                     │ __code__ access             │                            │ parameters                     │
  ├───────────────────┼──────────────────────────┼─────────────────────┼─────────────────────────────┼────────────────────────────┼────────────────────────────────┤
  │ B. Namespace      │ —                        │ —                   │ globals()/locals()/vars()   │ sys._getframe              │ dict mutation connected to     │
  │ mutation          │                          │                     │                             │ (no stub)                  │ namespace                      │
  ├───────────────────┼──────────────────────────┼─────────────────────┼─────────────────────────────┼────────────────────────────┼────────────────────────────────┤
  │ C. Function       │ —                        │ name rebinding      │ __defaults__/__closure__    │ decorators                 │ function.__dict__              │
  │ mutation          │                          │ (partial)           │ access, nonlocal writes     │ (no stub)                  │                                │
  ├───────────────────┼──────────────────────────┼─────────────────────┼─────────────────────────────┼────────────────────────────┼────────────────────────────────┤
  │ D. Monkey         │ instance.attr= (within   │ —                   │ —                           │ —                          │ module/class patching          │
  │ patching          │ function)                │                     │                             │                            │                                │
  ├───────────────────┼──────────────────────────┼─────────────────────┼─────────────────────────────┼────────────────────────────┼────────────────────────────────┤
  │ E. Object state   │ —                        │ —                   │ setattr/delattr,            │ getattr                    │ __slots__                      │
  │ mutation          │                          │                     │ __dict__ access             │ (no stub)                  │                                │
  ├───────────────────┼──────────────────────────┼─────────────────────┼─────────────────────────────┼────────────────────────────┼────────────────────────────────┤
  │ F. Descriptor     │ property (stub-declared) │ —                   │ —                           │ —                          │ __getattr__, custom            │
  │ machinery         │                          │                     │                             │                            │ descriptors                    │
  ├───────────────────┼──────────────────────────┼─────────────────────┼─────────────────────────────┼────────────────────────────┼────────────────────────────────┤
  │ G. Callable       │ __init__ (stub-declared) │ —                   │ —                           │ —                          │ __new__, __call__, factories   │
  │ interception      │                          │                     │                             │                            │                                │
  ├───────────────────┼──────────────────────────┼─────────────────────┼─────────────────────────────┼────────────────────────────┼────────────────────────────────┤
  │ H. Protocol       │ subscript, iter,         │                     │ yield                       │ async iter, context        │                                │
  │ dispatch          │ arithmetic, bool,        │ —                   │                             │ managers, pattern matching  │ user types without stubs       │
  │                   │ containment              │                     │                             │                            │                                │
  ├───────────────────┼──────────────────────────┼─────────────────────┼─────────────────────────────┼────────────────────────────┼────────────────────────────────┤
  │ I.                │ —                        │ —                   │ __class__/__bases__          │ type(n,b,d), class         │ metaclasses,                   │
  │ Metaprogramming   │                          │                     │ access                      │ decorators                 │ __mro_entries__                 │
  ├───────────────────┼──────────────────────────┼─────────────────────┼─────────────────────────────┼────────────────────────────┼────────────────────────────────┤
  │ J. Import         │ static known imports     │ —                   │ __import__,                 │ unknown modules,           │ sys.modules, module            │
  │ mutation          │                          │                     │ dangerous modules (warn)    │ importlib (no stub)        │ __getattr__                    │
  ├───────────────────┼──────────────────────────┼─────────────────────┼─────────────────────────────┼────────────────────────────┼────────────────────────────────┤
  │ K. Serialization  │ —                        │ —                   │ —                           │ pickle/shelve imports      │ __reduce__/__getstate__        │
  │                   │                          │                     │                             │ (no stub)                  │ dunders                        │
  ├───────────────────┼──────────────────────────┼─────────────────────┼─────────────────────────────┼────────────────────────────┼────────────────────────────────┤
  │ L. Finalization   │ —                        │ —                   │ —                           │ —                          │ __del__, weakref callbacks,    │
  │                   │                          │                     │                             │                            │ generator cleanup              │
  ├───────────────────┼──────────────────────────┼─────────────────────┼─────────────────────────────┼────────────────────────────┼────────────────────────────────┤
  │ M. Async          │ —                        │ —                   │ —                           │ async/await syntax         │ callback registration          │
  │ scheduling        │                          │                     │                             │ (bad bytecode)             │ semantics                      │
  ├───────────────────┼──────────────────────────┼─────────────────────┼─────────────────────────────┼────────────────────────────┼────────────────────────────────┤
  │ N.                │ —                        │ —                   │ threading/signal imports    │ —                          │ concurrent mutation within     │
  │ Threads/signals   │                          │                     │ (warn)                      │                            │ analyzed code                  │
  ├───────────────────┼──────────────────────────┼─────────────────────┼─────────────────────────────┼────────────────────────────┼────────────────────────────────┤
  │ O. Interpreter    │ —                        │ —                   │ —                           │ sys.settrace (no stub)     │ indirect trace hooks           │
  │ hooks             │                          │                     │                             │                            │                                │
  ├───────────────────┼──────────────────────────┼─────────────────────┼─────────────────────────────┼────────────────────────────┼────────────────────────────────┤
  │ P. Builtins       │ —                        │ —                   │ __builtins__ access,        │ —                          │ rebinding builtins in          │
  │ mutation          │                          │                     │ breakpoint                  │                            │ global scope                   │
  ├───────────────────┼──────────────────────────┼─────────────────────┼─────────────────────────────┼────────────────────────────┼────────────────────────────────┤
  │ Q. Native code    │ —                        │ —                   │ ctypes/cffi imports (warn)  │ unstubbed native modules   │ —                              │
  ├───────────────────┼──────────────────────────┼─────────────────────┼─────────────────────────────┼────────────────────────────┼────────────────────────────────┤
  │ R. Impl-dependent │ —                        │ —                   │ —                           │ —                          │ all interpreter-specific       │
  │                   │                          │                     │                             │                            │ behavior                       │
  └───────────────────┴──────────────────────────┴─────────────────────┴─────────────────────────────┴────────────────────────────┴────────────────────────────────┘


---


Numba faces the same checklist but makes different engineering choices. Let me outline the key structural differences,
  then walk through category by category.

  Architectural contrast

  ┌────────────────────┬──────────────────────────────────────────────────┬───────────────────────────────────────────────────────────────────┐
  │                    │                      Pythia                      │                               Numba                               │
  ├────────────────────┼──────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ Goal               │ Determine dirty variables for checkpointing      │ JIT-compile to machine code                                       │
  ├────────────────────┼──────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ Failure mode       │ Unsoundness (misses a mutation → incorrect       │ Compilation failure or wrong results                              │
  │                    │ checkpoint)                                      │                                                                   │
  ├────────────────────┼──────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ Unsupported        │ Mostly crash accidentally (missing stub)         │ Mostly rejected explicitly (type inference failure with           │
  │ features           │                                                  │ diagnostic)                                                       │
  ├────────────────────┼──────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ Type source        │ Static stubs (typeshed_mini/)                    │ Runtime type inference at first call (specialization)             │
  ├────────────────────┼──────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ Scope              │ Intraprocedural, single function                 │ Intraprocedural in nopython mode; interprocedural within compiled │
  │                    │                                                  │  call graph                                                       │
  ├────────────────────┼──────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ Subset enforced?   │ Yes — validation pass (validate.py) rejects      │ Yes — nopython mode explicitly defines a supported subset         │
  │                    │ unsupported features before analysis             │                                                                   │
  └────────────────────┴──────────────────────────────────────────────────┴───────────────────────────────────────────────────────────────────┘

  Pythia now has a validation pass (pythia/validate.py) that defines and enforces a supported subset, similar to Numba's approach. Features
  outside the subset are rejected with clear messages before analysis begins. The remaining difference is scope: Numba's type inference covers
  everything it compiles, while Pythia's validation catches specific anti-patterns but relies on stub presence (detected by assertion) for
  type coverage.

  ---
  Category-by-category comparison

  A. Runtime code creation (eval/exec/compile)

  ┌────────────┬────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────┐
  │            │               Pythia               │                                          Numba                                          │
  ├────────────┼────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┤
  │ Handling   │ Explicitly rejected by validation   │ Explicitly rejected — eval/exec are not in the supported subset. Nopython mode cannot   │
  │            │ pass with clear message             │ represent arbitrary code objects                                                        │
  ├────────────┼────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┤
  │ Object     │ N/A                                │ Falls back to Python interpreter, so eval/exec work but no JIT benefit                  │
  │ mode       │                                    │                                                                                         │
  └────────────┴────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────┘

  Both now reject explicitly with clear messages.

  B. Namespace / scope mutation (globals(), locals(), frames)

  ┌──────────┬──────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────────┐
  │          │              Pythia              │                                            Numba                                            │
  ├──────────┼──────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Handling │ Explicitly rejected by            │ Explicitly unsupported — globals()/locals() are not compilable in nopython mode. No frame   │
  │          │ validation pass                   │ access at all                                                                               │
  └──────────┴──────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────────┘

  Same practical outcome and both give clear diagnostics.

  C. Function / closure mutation

  ┌──────────────────┬─────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────┐
  │                  │       Pythia        │                                              Numba                                               │
  ├──────────────────┼─────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────┤
  │                  │ Pure closures:      │ Supports closures with inferred types — but the closed-over variables are captured by value at   │
  │ Closures         │ reads allowed,      │ compile time, not by reference. Mutation of the outer variable from inside the closure is not    │
  │                  │ writes rejected     │ reflected, and vice versa. This is a documented, intentional semantic difference                 │
  ├──────────────────┼─────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Function         │                     │                                                                                                  │
  │ attribute        │ Silent break        │ Not supported in nopython mode — functions are opaque typed references, not Python objects       │
  │ mutation         │                     │                                                                                                  │
  ├──────────────────┼─────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ __code__         │ Silent break        │ Impossible — compiled functions are machine code, not bytecode                                   │
  │ replacement      │                     │                                                                                                  │
  └──────────────────┴─────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────┘

  Both systems restrict closures: Numba captures by value (documented semantic change). Pythia assumes pure closures — read-only nonlocal access;
  writes are rejected by the validation pass. Both are documented restrictions on Python semantics.

  D. Monkey patching

  ┌────────────┬─────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │            │         Pythia          │                                               Numba                                                │
  ├────────────┼─────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │            │ Silent break (stubs are │ Rejected at the type level — nopython mode resolves functions at compile time. If you monkey-patch │
  │ Handling   │  authoritative)         │  a function after JIT compilation, the compiled version keeps the old target. This is a known      │
  │            │                         │ gotcha but effectively the same silent break                                                       │
  ├────────────┼─────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Mitigation │ None                    │ Numba has @generated_jit and @overload for extending the compiler. Recompilation on type change is │
  │            │                         │  possible via recompile()                                                                          │
  └────────────┴─────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────┘

  Both systems are vulnerable to monkey patching. The difference is Numba's community is aware of it and documents it.

  E. Object state / attribute-layout mutation (setattr, __dict__)

  ┌─────────────────────────┬─────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────┐
  │                         │           Pythia            │                                       Numba                                       │
  ├─────────────────────────┼─────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ setattr/getattr/delattr │ setattr/delattr rejected by │ Explicitly rejected — not in nopython subset                                      │
  │                         │ validation pass; getattr    │                                                                                   │
  │                         │ crashes (no stub)           │                                                                                   │
  ├─────────────────────────┼─────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ __dict__ access         │ Rejected by validation pass │ Not available — nopython objects use struct layouts, not dicts                    │
  ├─────────────────────────┼─────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ __slots__               │ Silent break                │ Numba's @jitclass uses a struct layout similar to __slots__ — it's the only       │
  │                         │                             │ object model available. No dynamic attributes                                     │
  └─────────────────────────┴─────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────┘

  Numba's approach is fundamentally different: it replaces Python's attribute model entirely with a C-struct model. This eliminates the entire
  category rather than trying to analyze it.

  F. Descriptor machinery (__getattr__, properties, descriptors)

  ┌─────────────────────────┬──────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────┐
  │                         │        Pythia        │                                          Numba                                           │
  ├─────────────────────────┼──────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ __getattr__/__setattr__ │ Silent break         │ Not supported — attribute access on @jitclass uses direct struct field access. No        │
  │                         │                      │ interception                                                                             │
  ├─────────────────────────┼──────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ Properties              │ Precise              │ Supported on @jitclass via @property decorator, but compiled to direct getter/setter —   │
  │                         │ (stub-declared)      │ no descriptor protocol dispatch                                                          │
  ├─────────────────────────┼──────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ Custom descriptors      │ Silent break         │ Not supported — would fail type inference                                                │
  └─────────────────────────┴──────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────┘

  Numba eliminates descriptors by compiling attribute access to direct memory offsets. The descriptor protocol doesn't exist in compiled code.

  G. Callable / construction interception (__call__, __new__)

  ┌──────────┬─────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────┐
  │          │         Pythia          │                                          Numba                                          │
  ├──────────┼─────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┤
  │ __call__ │ Silent break            │ Not supported in nopython mode for arbitrary objects. Only recognized on specific types │
  ├──────────┼─────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┤
  │ __new__  │ Silent break            │ Not supported — @jitclass uses a fixed constructor model                                │
  ├──────────┼─────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┤
  │ __init__ │ Precise (stub-declared) │ Supported for @jitclass, compiled to direct initialization                              │
  └──────────┴─────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────┘

  H. Protocol dispatch (dunders behind syntax)

  ┌─────────────────────────┬──────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────┐
  │                         │        Pythia        │                                          Numba                                           │
  ├─────────────────────────┼──────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ Arithmetic dunders      │ Precise              │ Precise — Numba implements __add__ etc. for numeric types and NumPy arrays. User-defined │
  │                         │ (stub-declared)      │  dunders on @jitclass also supported                                                     │
  ├─────────────────────────┼──────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ __getitem__/__setitem__ │ Precise              │ Precise — supported for arrays, tuples, and @jitclass with __getitem__                   │
  │                         │ (stub-declared)      │                                                                                          │
  ├─────────────────────────┼──────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ __iter__/__next__       │ Precise              │ Supported for known iterables (range, arrays, enumerate, zip). Custom iterators via      │
  │                         │ (stub-declared)      │ @jitclass possible but limited                                                           │
  ├─────────────────────────┼──────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ Context managers        │ Detected (crashes)   │ Limited support — with objmode() is a Numba-specific construct. General with is not      │
  │                         │                      │ supported in nopython                                                                    │
  ├─────────────────────────┼──────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ Async                   │ Detected (crashes)   │ Explicitly unsupported                                                                   │
  └─────────────────────────┴──────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────┘

  Both handle the common numeric/array dunders well. Neither handles the full protocol dispatch model.

  I. Class / type metaprogramming

  ┌────────────────────────────┬────────────────┬────────────────────────────────────────────────────────────────────────────────────────────┐
  │                            │     Pythia     │                                           Numba                                            │
  ├────────────────────────────┼────────────────┼────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Metaclasses                │ Silent break   │ Explicitly unsupported — @jitclass is the only class mechanism, and it doesn't go through  │
  │                            │                │ type.__call__                                                                              │
  ├────────────────────────────┼────────────────┼────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Dynamic type() creation    │ Detected       │ Explicitly unsupported                                                                     │
  │                            │ (crash)        │                                                                                            │
  ├────────────────────────────┼────────────────┼────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Class decorators           │ Detected       │ Only @jitclass and related Numba decorators. Arbitrary class decorators unsupported        │
  │                            │ (crash)        │                                                                                            │
  ├────────────────────────────┼────────────────┼────────────────────────────────────────────────────────────────────────────────────────────┤
  │ __class__/__bases__        │ Silent break   │ Impossible — compiled classes have fixed struct layouts                                    │
  │ mutation                   │                │                                                                                            │
  └────────────────────────────┴────────────────┴────────────────────────────────────────────────────────────────────────────────────────────┘

  Numba eliminates this category entirely by providing its own class mechanism.

  J. Import-system mutation

  ┌─────────────────────────┬─────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────┐
  │                         │         Pythia          │                                        Numba                                         │
  ├─────────────────────────┼─────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
  │ Static imports          │ Precise (stubbed        │ Resolved at compile time. Supported modules have Numba overloads; others fail type   │
  │                         │ modules)                │ inference                                                                            │
  ├─────────────────────────┼─────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
  │ Dynamic imports         │ Detected (crash)        │ Unsupported in nopython                                                              │
  ├─────────────────────────┼─────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
  │ Import hooks,           │ Silent break            │ Irrelevant — Numba resolves at compile time, not at import time within compiled code │
  │ sys.modules             │                         │                                                                                      │
  └─────────────────────────┴─────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────┘

  K–L. Serialization, finalization, GC

  ┌───────────────────┬──────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────┐
  │                   │                Pythia                │                                     Numba                                      │
  ├───────────────────┼──────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ pickle/__reduce__ │ Detected (missing module) / Silent   │ Not supported in nopython mode                                                 │
  │                   │ break (dunders)                      │                                                                                │
  ├───────────────────┼──────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ __del__/weakrefs  │ Silent break                         │ Not supported — @jitclass instances don't have finalizers in compiled code. GC │
  │                   │                                      │  is handled outside compiled regions                                           │
  ├───────────────────┼──────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ Generator cleanup │ Silent break                         │ Limited generator support; cleanup semantics not guaranteed                    │
  └───────────────────┴──────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────┘

  M–N. Async, threads, signals

  ┌─────────────┬─────────────┬───────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │             │   Pythia    │                                                     Numba                                                     │
  ├─────────────┼─────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ async/await │ Detected    │ Explicitly unsupported                                                                                        │
  │             │ (crash)     │                                                                                                               │
  ├─────────────┼─────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │             │ Silent      │ Numba releases the GIL in nopython mode and supports prange for parallel loops. But it assumes no shared      │
  │ Threads     │ break       │ mutable Python state across threads — only shared NumPy arrays (which it handles correctly via its own memory │
  │             │             │  model)                                                                                                       │
  ├─────────────┼─────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Signals     │ Silent      │ Not supported in compiled code                                                                                │
  │             │ break       │                                                                                                               │
  └─────────────┴─────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  Numba is notably stronger here: it actually has a threading model (prange, nogil) that is designed to be sound for its use case.

  O–P. Interpreter hooks, builtins mutation

  ┌─────────────────────────────┬──────────────────┬─────────────────────────────────────────────────────────────────────────────────┐
  │                             │      Pythia      │                                      Numba                                      │
  ├─────────────────────────────┼──────────────────┼─────────────────────────────────────────────────────────────────────────────────┤
  │ sys.settrace/sys.setprofile │ Detected (crash) │ Irrelevant — compiled code doesn't go through the interpreter trace mechanism   │
  ├─────────────────────────────┼──────────────────┼─────────────────────────────────────────────────────────────────────────────────┤
  │ Builtins mutation           │ Silent break     │ Irrelevant — builtins are resolved at compile time and compiled to direct calls │
  └─────────────────────────────┴──────────────────┴─────────────────────────────────────────────────────────────────────────────────┘

  Numba's compilation model makes these categories irrelevant rather than dangerous.

  Q. Foreign-function / native code

  ┌─────────────┬──────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │             │      Pythia      │                                                  Numba                                                   │
  ├─────────────┼──────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ C           │ Detected         │ Numba compiles to native code. It has @cfunc for creating C-callable functions, ctypes/cffi interop for  │
  │ extensions  │ (missing stubs)  │ calling external C, and intrinsic support for LLVM IR. This is a core strength, not a weakness           │
  ├─────────────┼──────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ ctypes/cffi │ Detected (crash) │ Supported — Numba can call ctypes/cffi functions directly in nopython mode                               │
  └─────────────┴──────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  This is the biggest asymmetry. Numba is designed for native interop; Pythia treats it as out of scope.

  ---
  Key takeaway

  Pythia now has a validation pass (pythia/validate.py) that defines and enforces a supported subset, closing the gap with Numba
  on several categories. The remaining structural differences:

  1. Numba eliminates categories rather than analyzing them. By replacing Python's object model with structs, its attribute model with memory
  offsets, and its dispatch model with direct calls, Numba makes entire categories (descriptors, metaclasses, __dict__, monkey patching)
  structurally impossible. Pythia rejects some of these (e.g., __dict__ access) but cannot prevent all of them (e.g., external monkey patching).
  2. Both systems document their restrictions. Numba documents capture-by-value closures and compilation-time binding. Pythia documents pure
  closures, rejected builtins, and stub requirements in validate.py's module docstring.
  3. Residual silent breaks. Numba's main silent break is monkey patching between compilations. Pythia's remaining silent breaks are: monkey
  patching (D), descriptor interception (F), __new__/__call__ interception (G), metaclasses (I), __del__/weakref callbacks (L), callback
  scheduling semantics (M), concurrent mutation (N), and builtin rebinding in global scope (P). These are all external-to-the-function effects
  that cannot be detected from the function's bytecode alone.
  4. Type coverage. Numba's type inference covers everything it compiles, failing explicitly on unknown types. Pythia relies on type stubs; a
  missing stub causes an assertion crash (detected, but with an unhelpful message). Improving these assertion messages is the main remaining
  user-experience gap.
  