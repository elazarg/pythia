# Spyte

Spyte is a static analysis framework for Python that translates bytecode to Three Address Code (TAC) and performs various analyses on it, including type analysis, data flow analysis, and code instrumentation for efficient checkpointing.

## Overview

Spyte provides tools for:
- Translating Python [bytecode](https://docs.python.org/3.11/library/dis.html#python-bytecode-instructions) to IR (Spytecode)
- Building and analyzing Control Flow Graphs (CFGs)
- Performing data flow analysis
- Type inference and checking
- Code instrumentation for runtime analysis
- Checkpoint-based execution for debugging and analysis

## Installation

### Prerequisites
- Python 3.12 - EXACTLY.

### Setup
1. Clone the repository:
   ```
   git clone https://github.com/elazarg/pythia.git
   cd pythia
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Unix/MacOS
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Analyzing Python Code

Use `analyze.py` to analyze a Python file and function:

```
python analyze.py path/to/file.py function_name [--output output_file] [--print-invariants] [--simplify/--no-simplify]
```

Options:
- `--output`: Specify an output file (default: print to stdout)
- `--print-invariants`: Print analysis invariants
- `--simplify/--no-simplify`: Enable/disable simplification (default: enabled)

### Instrumenting Python Code

Use `instrument.py` to instrument a Python file for runtime analysis:

```
python instrument.py [options] pythonfile [args]
```

Options:
- `--kind {analyze,naive,vm,proc}`: Kind of instrumentation to use (default: analyze)
- `--function FUNCTION`: Function to instrument (default: run)
- `--fuel FUEL`: Number of iterations to run before inducing a crash (default: 1000000)
- `--no-generate`: Do not generate an instrumented file
- `--step STEP`: Number of iterations to run between checkpointing (default: 1)

### Core Modules

- **spytecode.py**: Translates basic block CFG into equivalent IR (Spytecode)
- **analysis.py**: Performs data flow analysis of TAC CFG
- **graph_utils.py**: Provides utilities for working with Control Flow Graphs
- **type_system.py**: Implements a static type system for Python
- **ast_transform.py**: Transforms Python AST for instrumentation
- **instruction_cfg.py**: Builds CFGs from Python bytecode instructions
- **domains.py**: Defines abstract domains for static analysis
- **dom_typed_pointer.py**: Implements pointer analysis with type information
- **dom_liveness.py**: Implements liveness analysis for variables
- **dom_concrete.py**: Provides concrete domain implementations
- **strategy.py**: Defines iteration strategies for fixed-point computations
- **disassemble.py**: Utilities for disassembling Python bytecode

## Examples

The `experiment` directory contains example applications of Spyte, including:
- k_means: K-means clustering algorithm
- omp: One More Proof algorithm
- pivoter: Graph analysis algorithm
- trivial: Simple example for testing

The `test_data` directory contains small Python files used for testing the analysis.

## Testing

Run the test suite with pytest:

```
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
