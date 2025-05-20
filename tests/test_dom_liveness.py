import pytest
from pythia.dom_liveness import LivenessVarLattice, Liveness, Set, TOP, BOTTOM
from pythia.tac import Var, Assign, Const, Binary, Return, Del
from pythia.graph_utils import Location

X = Var("X", False)
Y = Var("Y", False)
Z = Var("Z", False)


def test_liveness_var_lattice_init():
    lattice = LivenessVarLattice()
    assert str(lattice.lattice) == "Set()"
    assert lattice.backward is True


def test_liveness_var_lattice_name():
    lattice = LivenessVarLattice()
    assert lattice.name() == "Liveness"


def test_liveness_var_lattice_initial():
    lattice = LivenessVarLattice()
    initial = lattice.initial()
    assert str(initial) == "Set()"


def test_liveness_var_lattice_is_bottom():
    assert LivenessVarLattice.is_bottom(BOTTOM)
    assert not LivenessVarLattice.is_bottom(Set())
    assert not LivenessVarLattice.is_bottom(TOP)


def test_liveness_var_lattice_top_bottom():
    lattice = LivenessVarLattice()
    assert lattice.top() is TOP
    assert lattice.bottom() is BOTTOM


def test_liveness_var_lattice_join():
    lattice = LivenessVarLattice()

    # Test joining with BOTTOM
    assert lattice.join(BOTTOM, Set()) == Set()
    assert lattice.join(Set(), BOTTOM) == Set()

    # Test joining with TOP
    assert lattice.join(TOP, Set()) is TOP
    assert lattice.join(Set(), TOP) is TOP

    # Test joining two sets
    set1 = Set([X, Y])
    set2 = Set([Y, Z])
    result = lattice.join(set1, set2)

    assert X in result
    assert Y in result
    assert Z in result


def test_liveness_var_lattice_is_less_than():
    lattice = LivenessVarLattice()

    set1 = Set([X])
    set2 = Set([X, Y])
    set3 = Set([Z])

    # set1 is a subset of set2
    assert lattice.is_less_than(set1, set2)

    # set2 is not a subset of set1
    assert not lattice.is_less_than(set2, set1)

    # set1 and set3 have no common elements
    assert not lattice.is_less_than(set1, set3)
    assert not lattice.is_less_than(set3, set1)


def test_liveness_var_lattice_transfer_assign():
    lattice = LivenessVarLattice()

    # Create an assignment: X = Y + Z
    binary_op = Binary(Y, "+", Z, inplace=False)
    assign = Assign(X, binary_op)

    # Initial liveness set with just X
    initial = Set([X])

    # After the assignment, X is no longer live (it is defined), but Y and Z are
    result = lattice.transfer(initial, assign, (0, 0))

    assert X not in result  # X is defined, so it is removed
    assert Y in result  # Y is used, so it is added
    assert Z in result  # Z is used, so it is added


def test_liveness_var_lattice_transfer_return():
    lattice = LivenessVarLattice()

    # Create a return statement: return X
    ret = Return(X)

    # Initial empty liveness set
    initial = Set()

    # After the return, X is live because it is used
    result = lattice.transfer(initial, ret, (0, 0))

    assert X in result  # X is used in the return, so it is live


def test_liveness_var_lattice_transfer_bottom():
    lattice = LivenessVarLattice()

    # Create an assignment
    assign = Assign(X, Const(42))

    # Test transfer with BOTTOM
    result = lattice.transfer(BOTTOM, assign, (0, 0))

    assert result is BOTTOM  # BOTTOM should propagate through transfer
