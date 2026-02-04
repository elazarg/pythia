# Pythia - Claude's Working Notes

## Project Overview

Pythia is a **static analysis framework for Python** that:
1. Translates Python bytecode to Three Address Code (TAC)
2. Builds and analyzes Control Flow Graphs (CFGs)
3. Performs data flow analysis (liveness, typed pointers)
4. Type inference
5. Code instrumentation for efficient checkpointing

**Purpose**: Determine which variables are "dirty" (modified) in loops to enable efficient checkpointing - only save what changed.

## Architecture

### Core Modules (`pythia/`)

| Module | Purpose |
|--------|---------|
| `tac.py` | Bytecode -> TAC translation. Defines TAC instructions (Assign, Jump, For, Return, etc.) |
| `analysis.py` | Abstract interpretation engine. Runs fixpoint computation over CFG |
| `domains.py` | Abstract domain interfaces (Lattice, InstructionLattice, ValueLattice) |
| `dom_typed_pointer.py` | **Main analysis** - tracks pointer/type info and dirty fields |
| `dom_liveness.py` | Liveness analysis for variables |
| `dom_concrete.py` | Concrete domain implementations (Set, Map) |
| `type_system.py` | Static type system, type expressions, joins, subscripting |
| `graph_utils.py` | CFG utilities |
| `instruction_cfg.py` | Builds CFGs from Python bytecode |
| `ast_transform.py` | AST transformation for instrumentation |
| `strategy.py` | Iteration strategies for fixpoint |
| `disassemble.py` | Bytecode disassembly utilities |

### Entry Points

- `analyze.py` - Analyze a Python file/function
- `instrument.py` - Instrument code for runtime analysis

### Test Coverage

- 259 tests, all passing
- Tests in `tests/` directory

## Current State

- Python 3.12, 3.13, and 3.14 supported
- All 259 tests pass
- Branch: `master`
- Key features:
  - **Phase 4: BoundCall + Call separation** - Separates function resolution from call execution
  - `pythia/tac.py` - Added BoundCall expression type, LIST_EXTEND/SET_UPDATE bytecode support
  - `pythia/dom_typed_pointer.py` - Added BoundCall analysis case, bound method tracking
  - `typeshed_mini/builtins.pyi` - Added max() overload with default parameter

### Recent Improvements

- **Bytecode support**: BUILD_CONST_KEY_MAP, BUILD_MAP (dict literals), UNPACK_EX (starred unpacking), CALL_KW (Python 3.13)
- **Type system**: Complete Access type handling in squeeze/join/meet/is_immutable
- **Code cleanup**: Fixed "builtings" typo, removed unreachable code, consolidated version checks
- **Documentation**: Documented debugging artifact in dom_typed_pointer.py

## TODOs Found in Code

1`dom_typed_pointer.py` - Multiple TODOs:
   - Line 989, 1062: `# TODO: class through type`
   - Line 1073: `# TODO: assert not direct_objs ??`
   - Line 1175: `# TODO: point from exact literal when possible`
   - Line 609: `# TODO: minus one only for self. Should be fixed on binding` - **Investigated**: This is a workaround for method binding not adjusting argument indices. The `-1` compensates for self being index 0 in type annotations but not in `arg_objects`. Proper fix would require changes to `bind_self_function` in `type_system.py`.
2`type_system.py` - Commented out TODOs about function compatibility

## Potential Improvements

### Code Quality
- [ ] The `-1` offset for bound methods in `dom_typed_pointer.py:609` should be fixed at binding time in `type_system.py:bind_self_function`
- [x] `dom_typed_pointer.py:657` has `if True or ...` - documented as debugging artifact that disables optimization

### Testing
- [ ] Add property-based testing?
- [ ] Test coverage for edge cases in type system

### Analysis Quality
- [x] Ran analysis with `--print-invariants` - output shows Liveness, TypedPointer, Types, and Dirty maps at each program point
- [x] BUILD_CONST_KEY_MAP and BUILD_MAP bytecode instructions implemented
- [x] UNPACK_EX bytecode instruction implemented (starred unpacking)
- [x] CALL_KW bytecode instruction implemented (Python 3.13)
- [ ] Type system doesn't know about some numpy functions (e.g., `np.random.rand`) - causes assertion error "Expected Overloaded type, got BOT"

### Key Insights from Paper Appendices

**Abstract Domains (Appendix: Pointer Analysis)**
- **Pointer Domain (P)**: O -> K -> 2^O (maps objects to field-keyed sets of objects)
- **Type Domain (T)**: O -> TypeExpr (maps objects to type expressions)
- **Dirty Domain (D)**: O -> 2^K (maps objects to sets of dirty fields)

**Strong vs Weak Updates**:
```
Upd(P, O_tgt, f, O_new) =
  if |O_tgt| = 1: P[o][f] <- O_new       (strong update)
  otherwise:      P[o][f] <- P[o][f] ∪ O_new  (weak update)
```

**Current Modification Analysis**:
The modified `TypeMap.__setitem__` always uses `ts.join` (weak update) instead of direct assignment for singleton cases. This is MORE CONSERVATIVE (sound) but LESS PRECISE than the paper's approach.

Interestingly, `Pointer.__setitem__` still uses strong updates (direct assignment) at lines 222 and 234. This asymmetry may be intentional or a bug.

**Analysis of TypeMap join behavior**:
- `signature()` at line 930 does `tp.types[pointed] = t` within the transfer function
- For most cases, `pointed` is a new location object, so join vs assign doesn't matter
- The change affects line 760: `new_tp.types[self_obj] = side_effect.update[0]` for mutating operations
- Join is more conservative (sound) but may lose precision for mutation tracking
- Tests still pass, suggesting this doesn't break existing behavior
- **Status**: Leaving as-is; further investigation needed to understand original motivation

**DirtyRoots Formula** (from paper):
```
DirtyRoots(Σ, R) = { x ∈ Live | ∃o ∈ Reach({P[LOCALS][x]}, P). D[o] ≠ ∅ }
```

**Assumptions**:
- No dynamic code evaluation (eval/exec)
- Statically-resolvable calls
- Explicit generic instantiations (e.g., `list[int]()`)
- No closures mutating globals
- No generators, coroutines, context managers

## Key Insight: What "Dirty" Means

The analysis determines which local variables are "dirty" at loop checkpoints. Comparing:

**Naive approach** (`naive.py`): Saves ALL local variables modified anywhere in the loop
```python
transaction.commit(counter, curr, neighbour, parent, r, root_to_leaf_path, world)  # 7 vars
```

**With analysis** (`instrumented.py`): Saves only variables that ESCAPE the loop (are live after)
```python
transaction.commit(counter, root_to_leaf_path)  # 2 vars!
```

Variables like `curr`, `neighbour`, `parent`, `r`, `world` are modified but their values don't persist meaningfully across iterations - they're recomputed each time.

## How Loops Are Marked

Loops are marked for checkpointing with a type comment:
```python
for r in range(10**100):  # type: int  <-- This marks the loop!
```

The analysis finds these loops via `annotated_for_labels()` in `ast_transform.py`.

## Experiments (`experiment/` folder)

The `experiment/` folder contains empirical evaluation benchmarks comparing different checkpointing strategies. Each experiment has multiple variants:

| File | Description |
|------|-------------|
| `main.py` | Original algorithm without checkpointing |
| `instrumented.py` | Uses Pythia's static analysis to checkpoint only dirty variables |
| `naive.py` | Naive checkpointing that saves all modified variables |
| `proc.py` | Process-level snapshotting |
| `vm.py` | VM-level snapshotting |

### Benchmarks

- **k_means**: K-means clustering algorithm on random data
- **omp**: Original OMP (Orthogonal Matching Pursuit) implementation
- **pivoter**: Graph pivoting algorithm for clique enumeration
- **trivial**: Simple test case
- **worst**: Worst-case scenario for analysis

These experiments demonstrate the effectiveness of static analysis for reducing checkpoint size compared to naive approaches.
