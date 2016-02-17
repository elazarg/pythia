import z3

def sat(solver):
    res = solver.check()
    assert res == z3.sat or res == z3.unsat
    return res == z3.sat

def unsat(solver):
    return not sat(solver)

def sat_formula(formula):
    solver = z3.Solver()
    solver.add(formula)
    return sat(solver) 