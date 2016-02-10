# pythia

Translating Python to Three Address Code

## Pipeline operations by modules:

1. [bcode.py](bcode.py) Python code to bytecode (CPython functions)

  1. Fixing minor bytecode issues
  2. Turn into BCode
  
2. [bcode_cfg.py](bcode_cfg.py): Generate CFG of BCode

  1. Find stack depth at each point
  2. Build basic block CFG
  
3. [tac.py](tac.py): Translate basic block CFG into equivalent Three Address Code (TAC)
4. [tac_analysis.py](tac_analysis.py): Data flow analysis of TAC CFG
