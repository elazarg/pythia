"""Tests for container element tracking through @accessor annotation.

These tests verify that mutating elements accessed from containers correctly
marks the container as dirty.
"""
import pathlib
import tempfile
import pytest

from pythia import analysis


def analyze_code(code: str, function_name: str) -> set[str]:
    """Analyze code and return dirty roots for the function's loops.

    Returns the union of all dirty roots across all loops in the function.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        f.flush()
        filepath = pathlib.Path(f.name)

    try:
        result = analysis.analyze_function(filepath, function_name)
        # Collect all dirty roots from all loop locations
        dirty_roots: set[str] = set()
        for func_result in result.values():
            for loc, roots in func_result.dirty_map.items():
                dirty_roots.update(roots)
        return dirty_roots
    finally:
        filepath.unlink()


def test_list_direct_mutation_dirty():
    """Test that direct list mutation marks list as dirty (baseline test)."""
    code = '''
def test():
    a = list[int]()
    for i in range(10):  # type: int
        a.append(i)
    return a
'''
    dirty = analyze_code(code, "test")
    # This should already work - direct mutation via append
    assert "a" in dirty, f"List should be dirty when directly mutated, got {dirty}"


def test_nested_list_dirty_tracking():
    """Test that mutating nested list element marks outer list as dirty."""
    code = '''
def test():
    inner = list[int]()
    outer = [inner]
    for i in range(10):  # type: int
        x = outer[0]
        x.append(i)
    return outer
'''
    dirty = analyze_code(code, "test")
    # outer should be dirty because outer.* -> inner and inner is mutated
    assert "outer" in dirty, f"Outer list should be dirty when inner is mutated, got {dirty}"


def test_nested_list_via_variable():
    """Test dirty tracking through list element accessed via variable."""
    code = '''
def test(idx: int):
    inner = list[int]()
    outer = [inner]
    for i in range(10):  # type: int
        x = outer[idx]
        x.append(i)
    return outer
'''
    dirty = analyze_code(code, "test")
    # outer should be dirty because outer.* -> inner and inner is mutated
    assert "outer" in dirty, f"Outer list should be dirty when inner is mutated via variable index, got {dirty}"


def test_dict_value_mutation():
    """Test that mutating dict value marks dict as dirty."""
    code = '''
def test(key: str, d: dict[str, list[int]]):
    inner = list[int]()
    d[key] = inner
    for i in range(10):  # type: int
        v = d[key]
        v.append(i)
    return d
'''
    dirty = analyze_code(code, "test")
    assert "d" in dirty, f"Dict should be dirty when value is mutated, got {dirty}"
