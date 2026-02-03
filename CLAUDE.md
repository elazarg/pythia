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

- Python 3.12 required (also supports 3.13)
- All 259 tests pass
- Branch: `claude-playground` (based on `master`)
- Key changes from master:
  - **Phase 4: BoundCall + Call separation** - Separates function resolution from call execution
  - `pythia/tac.py` - Added BoundCall expression type, LIST_EXTEND/SET_UPDATE bytecode support
  - `pythia/dom_typed_pointer.py` - Added BoundCall analysis case, bound method tracking
  - `typeshed_mini/builtins.pyi` - Added max() overload with default parameter

### Debug Print Statements
**FIXED**: Removed in commit c30c9e9:
- `dom_typed_pointer.py`: Removed print statements at lines 959-961, 985-986
- `analysis.py`: Made dirty_map print conditional on `print_invariants` flag

## TODOs Found in Code

1. ~~`tac.py:318` - `return set()  # TODO: fix this` in `free_vars_expr` for MakeFunction~~ **FIXED** (commit 1326c33)
2. `dom_typed_pointer.py` - Multiple TODOs:
   - ~~Line 334: `# TODO: check` in `is_less_than`~~ **VERIFIED CORRECT** (commit 6e80321) - only checking keys in self.map is correct because missing keys return BOTTOM
   - Line 989, 1062: `# TODO: class through type`
   - Line 1073: `# TODO: assert not direct_objs ??`
   - Line 1175: `# TODO: point from exact literal when possible`
   - Line 609: `# TODO: minus one only for self. Should be fixed on binding` - **Investigated**: This is a workaround for method binding not adjusting argument indices. The `-1` compensates for self being index 0 in type annotations but not in `arg_objects`. Proper fix would require changes to `bind_self_function` in `type_system.py`.
3. `type_system.py` - Commented out TODOs about function compatibility

## Potential Improvements

### Completed
- [x] Remove debug print statements from `dom_typed_pointer.py` and `analysis.py` (commit c30c9e9)
- [x] Fix the `free_vars_expr` TODO for MakeFunction (commit 1326c33)
- [x] Fix mutable default arguments in `__deepcopy__` methods (commit 7ed53d0)
- [x] Document why `is_less_than` only checks self.map keys (commit 6e80321)
- [x] Add test for `TypeMap.is_less_than` (commit d3d3d17)
- [x] Add test for `Pointer.is_less_than` (commit f53e38a)
- [x] **Keyword argument support** - Added kwnames field to Call, tracks KW_NAMES bytecode, passes keyword info to type analysis
- [x] **Phase 1-3**: Refactored expr() into helper methods, unified operator result creation, added explicit bound method tracking
- [x] **Phase 4**: Separated Call into BoundCall + Call (commit 946ff2c) - cleanly separates function resolution from execution
- [x] **LIST_EXTEND/SET_UPDATE bytecode** (commit 42b47b7) - supports list/set unpacking syntax

### Code Quality
- [ ] The `-1` offset for bound methods in `dom_typed_pointer.py:609` should be fixed at binding time in `type_system.py:bind_self_function`
- [ ] `dom_typed_pointer.py:591` has `if True or ...` - a debugging artifact that disables an optimization. The original condition `new_tp.types[self_obj] != side_effect.update[0]` would skip updates when types match. Currently always updates (more conservative).

### Testing
- [x] Added tests for `is_less_than` methods (TypeMap, Pointer) - verifies lattice ordering
- [ ] Add property-based testing?
- [ ] Test coverage for edge cases in type system

### Analysis Quality
- [x] Ran analysis with `--print-invariants` - output shows Liveness, TypedPointer, Types, and Dirty maps at each program point
- [ ] Missing bytecode instruction: `BUILD_CONST_KEY_MAP` (dict with const keys) - causes NotImplementedError
- [x] ~~Missing bytecode instruction: `LIST_EXTEND`~~ **FIXED** (commit 42b47b7)
- [ ] Type system doesn't know about some numpy functions (e.g., `np.random.rand`) - causes assertion error "Expected Overloaded type, got BOT"

## Related Paper

Located at `D:\workspace\Checkpointing-with-Static-Analysis`

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

**Assumptions** (Appendix 07):
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
- **omp_claud**: Claude's version of OMP
- **pivoter**: Graph pivoting algorithm for clique enumeration
- **trivial**: Simple test case
- **worst**: Worst-case scenario for analysis

These experiments demonstrate the effectiveness of static analysis for reducing checkpoint size compared to naive approaches.

## Session Log

### Review Started
- Read README.md - understood high-level purpose
- Read tac.py - bytecode to TAC translation
- Read analysis.py - abstract interpretation fixpoint algorithm
- Read dom_typed_pointer.py - main typed pointer analysis
- Read domains.py - abstract domain interfaces
- All 250 tests pass

### Deeper Understanding
- Read ast_transform.py - how code is instrumented
- Read persist.py - checkpoint/restore logic with pickle
- Read examples/pivoter.py - real example with loop annotation
- Compared instrumented.py vs naive.py - saw 7 vars -> 2 vars reduction!
