# pythia

Translating Python [bytecode](https://docs.python.org/3.5/library/dis.html#python-bytecode-instructions) to Three Address Code

## Pipeline operations by modules:
  
1. [tac.py](tac.py): Translate basic block CFG into equivalent Three Address Code (TAC)
2. [tac_analysis.py](tac_analysis.py): Data flow analysis of TAC CFG


## Working assumptions

1. There are two types of functions: constructors and heap-neutral.
   1. Constructors: create (concretely) an isolated sub-graph and returns a root
      to it.
   2. Heap neutral: do not change the heap object at all. In case the function
      returns an object, a contract (in the form of a function summary) is required
      to describe the identity of the returned object.

      Contracts may be describe as logical assersions but for economy reasons we
      will have them be the abstract transformers themselves.