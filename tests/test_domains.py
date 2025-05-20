import pytest
from pythia.domains import Top, Bottom, TOP, BOTTOM


def test_top_str():
    # Test string representation of Top
    assert str(TOP) == "⊤"


def test_top_deepcopy():
    # Test that deepcopy returns the same instance
    from copy import deepcopy

    top_copy = deepcopy(TOP)
    assert top_copy is TOP


def test_top_or():
    # Test | operator (union)
    assert TOP | 42 is TOP
    assert TOP | "hello" is TOP
    assert TOP | TOP is TOP
    assert TOP | BOTTOM is TOP


def test_top_ror():
    # Test reverse | operator
    assert 42 | TOP is TOP
    assert "hello" | TOP is TOP


def test_top_and():
    # Test & operator (intersection)
    assert TOP & 42 == 42
    assert TOP & "hello" == "hello"
    assert TOP & TOP is TOP
    assert TOP & BOTTOM is BOTTOM


def test_top_rand():
    # Test reverse & operator
    assert 42 & TOP == 42
    assert "hello" & TOP == "hello"


def test_top_contains():
    # Test contains operator
    assert 42 in TOP
    assert "hello" in TOP
    assert TOP in TOP
    assert BOTTOM in TOP


def test_top_copy():
    # Test that copy returns the same instance
    from copy import copy

    top_copy = copy(TOP)
    assert top_copy is TOP


def test_bottom_str():
    # Test string representation of Bottom
    assert str(BOTTOM) == "⊥"


def test_bottom_deepcopy():
    # Test that deepcopy returns the same instance
    from copy import deepcopy

    bottom_copy = deepcopy(BOTTOM)
    assert bottom_copy is BOTTOM


def test_bottom_or():
    # Test | operator (union)
    assert BOTTOM | 42 == 42
    assert BOTTOM | "hello" == "hello"
    assert BOTTOM | TOP is TOP
    assert BOTTOM | BOTTOM is BOTTOM


def test_bottom_ror():
    # Test reverse | operator
    assert 42 | BOTTOM == 42
    assert "hello" | BOTTOM == "hello"


def test_bottom_and():
    # Test & operator (intersection)
    assert BOTTOM & 42 is BOTTOM
    assert BOTTOM & "hello" is BOTTOM
    assert BOTTOM & TOP is BOTTOM
    assert BOTTOM & BOTTOM is BOTTOM


def test_bottom_rand():
    # Test reverse & operator
    assert 42 & BOTTOM is BOTTOM
    assert "hello" & BOTTOM is BOTTOM


class TestLatticeImplementation:
    """Test a simple implementation of the Lattice protocol."""

    class IntLattice:
        @classmethod
        def name(cls):
            return "IntLattice"

        @classmethod
        def top(cls):
            return float("inf")

        @classmethod
        def bottom(cls):
            return float("-inf")

        @classmethod
        def is_bottom(cls, elem):
            return elem == float("-inf")

        def join(self, left, right):
            return max(left, right)

        def is_less_than(self, left, right):
            return left <= right

    def test_lattice_methods(self):
        lattice = self.IntLattice()

        # Test name
        assert lattice.name() == "IntLattice"

        # Test top and bottom
        assert lattice.top() == float("inf")
        assert lattice.bottom() == float("-inf")

        # Test is_bottom
        assert lattice.is_bottom(float("-inf"))
        assert not lattice.is_bottom(0)
        assert not lattice.is_bottom(float("inf"))

        # Test join
        assert lattice.join(1, 2) == 2
        assert lattice.join(-1, 1) == 1
        assert lattice.join(float("-inf"), 0) == 0
        assert lattice.join(0, float("inf")) == float("inf")

        # Test join_all
        assert lattice.join(0, 3) == 3
        assert lattice.join(float("-inf"), 3) == 3
        assert lattice.join(0, float("inf")) == float("inf")

        # Test is_less_than
        assert lattice.is_less_than(1, 2)
        assert lattice.is_less_than(float("-inf"), 0)
        assert lattice.is_less_than(0, float("inf"))
        assert not lattice.is_less_than(2, 1)


class TestInstructionLatticeImplementation:
    """Test a simple implementation of the InstructionLattice protocol."""

    class SimpleInstructionLattice:
        backward = False

        @classmethod
        def name(cls):
            return "SimpleInstructionLattice"

        @classmethod
        def top(cls):
            return TOP

        @classmethod
        def bottom(cls):
            return BOTTOM

        @classmethod
        def is_bottom(cls, elem):
            return elem is BOTTOM

        def join(self, left, right):
            if left is BOTTOM:
                return right
            if right is BOTTOM:
                return left
            if left is TOP or right is TOP:
                return TOP
            return left  # Simplified for testing

        def is_less_than(self, left, right):
            if left is BOTTOM:
                return True
            if right is TOP:
                return True
            if left is TOP or right is BOTTOM:
                return False
            return left <= right  # Simplified for testing

        def transfer(self, values, ins, location):
            # Simplified transfer function for testing
            return values

        def initial(self):
            return self.top()

    def test_instruction_lattice_methods(self):
        lattice = self.SimpleInstructionLattice()

        # Test inherited methods from Lattice
        assert lattice.name() == "SimpleInstructionLattice"
        assert lattice.top() is TOP
        assert lattice.bottom() is BOTTOM
        assert lattice.is_bottom(BOTTOM)
        assert not lattice.is_bottom(TOP)

        # Test join
        assert lattice.join(BOTTOM, 42) == 42
        assert lattice.join(42, BOTTOM) == 42
        assert lattice.join(TOP, 42) is TOP
        assert lattice.join(42, TOP) is TOP

        # Test is_less_than
        assert lattice.is_less_than(BOTTOM, 42)
        assert lattice.is_less_than(42, TOP)
        assert not lattice.is_less_than(TOP, 42)
        assert not lattice.is_less_than(42, BOTTOM)

        # Test initial
        assert lattice.initial() is TOP

        # Test transfer (simplified)
        assert lattice.transfer(42, None, None) == 42


class TestValueLatticeImplementation:
    """Test a simple implementation of the ValueLattice protocol."""

    class SimpleValueLattice:
        @classmethod
        def name(cls):
            return "SimpleValueLattice"

        @classmethod
        def top(cls):
            return TOP

        @classmethod
        def bottom(cls):
            return BOTTOM

        @classmethod
        def is_bottom(cls, elem):
            return elem is BOTTOM

        def join(self, left, right):
            if left is BOTTOM:
                return right
            if right is BOTTOM:
                return left
            if left is TOP or right is TOP:
                return TOP
            return left  # Simplified for testing

        def is_less_than(self, left, right):
            if left is BOTTOM:
                return True
            if right is TOP:
                return True
            if left is TOP or right is BOTTOM:
                return False
            return left <= right  # Simplified for testing

        def const(self, value):
            return value  # Simplified for testing

        def var(self, value):
            return value

        def attribute(self, var, attr):
            return self.top()

        def subscr(self, array, index):
            return self.top()

        def call(self, function, args):
            return self.top()

        def predefined(self, name):
            return self.top()

        def annotation(self, name, t):
            return self.top()

        def default(self):
            return self.top()

    def test_value_lattice_methods(self):
        lattice = self.SimpleValueLattice()

        # Test inherited methods from Lattice
        assert lattice.name() == "SimpleValueLattice"
        assert lattice.top() is TOP
        assert lattice.bottom() is BOTTOM
        assert lattice.is_bottom(BOTTOM)
        assert not lattice.is_bottom(TOP)

        # Test value-specific methods
        assert lattice.const(42) == 42
        assert lattice.var(42) == 42
        assert lattice.attribute(42, None) is TOP
        assert lattice.subscr(42, 0) is TOP
        assert lattice.call(42, []) is TOP
        assert lattice.predefined(None) is TOP
        assert lattice.annotation(None, "") is TOP
        assert lattice.default() is TOP
