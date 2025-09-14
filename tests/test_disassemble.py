import pytest
import pathlib
from spyte.disassemble import (
    read_pyc_file,
    do_compile,
    read_function_using_compile,
    read_file,
    ParsedFile,
)

# Path to test files
LISTS_PATH = pathlib.Path("test_data/lists.py")
ITERATION_PATH = pathlib.Path("test_data/iteration.py")


def test_read_file_lists():
    # Test reading the lists.py file
    parsed_file = read_file(LISTS_PATH)

    # Check that it's a ParsedFile
    assert isinstance(parsed_file, ParsedFile)

    # Check that it contains the expected functions
    assert "empty_list_add" in parsed_file.functions
    assert "list_append" in parsed_file.functions
    assert "list_set" in parsed_file.functions
    assert "list_add" in parsed_file.functions
    assert "append" in parsed_file.functions
    assert "build_list_of_ints" in parsed_file.functions
    assert "build_list_of_lists" in parsed_file.functions
    assert "build_aliased_list_of_lists" in parsed_file.functions
    assert "build_aliased_list_of_known_lists" in parsed_file.functions

    # Check that the functions are callable
    assert callable(parsed_file.functions["empty_list_add"])
    assert callable(parsed_file.functions["list_append"])

    # Check that the imports dictionary is empty (no imports in lists.py)
    assert parsed_file.imports == {}

    # Check that annotated_for contains the expected functions
    assert "append" in parsed_file.annotated_for
    assert "build_list_of_ints" in parsed_file.annotated_for
    assert "build_list_of_lists" in parsed_file.annotated_for
    assert "build_aliased_list_of_lists" in parsed_file.annotated_for
    assert "build_aliased_list_of_known_lists" in parsed_file.annotated_for


def test_read_file_iteration():
    # Test reading the iteration.py file
    parsed_file = read_file(ITERATION_PATH)

    # Check that it's a ParsedFile
    assert isinstance(parsed_file, ParsedFile)

    # Check that it contains the expected functions
    assert "first_shape" in parsed_file.functions
    assert "setitem" in parsed_file.functions
    assert "counter" in parsed_file.functions
    assert "length" in parsed_file.functions
    assert "comprehension" in parsed_file.functions
    assert "get_world" in parsed_file.functions
    assert "loopfor" in parsed_file.functions
    assert "test_dict" in parsed_file.functions
    assert "iterate" in parsed_file.functions
    assert "double_iterate" in parsed_file.functions
    assert "cmp" in parsed_file.functions
    assert "negative" in parsed_file.functions
    assert "access" in parsed_file.functions
    assert "tup" in parsed_file.functions
    assert "listing" in parsed_file.functions
    assert "make_int" in parsed_file.functions
    assert "simple_tuple" in parsed_file.functions
    assert "destruct" in parsed_file.functions
    assert "test_tuple_simple_assign" in parsed_file.functions
    assert "test_tuple_assign_through_var" in parsed_file.functions
    assert "test_tuple2" in parsed_file.functions

    # Check that the functions are callable
    assert callable(parsed_file.functions["first_shape"])
    assert callable(parsed_file.functions["setitem"])

    # Check that the imports dictionary contains the expected imports
    assert "np" in parsed_file.imports
    assert parsed_file.imports["np"] == "numpy"
    assert "collections" in parsed_file.imports
    assert parsed_file.imports["collections"] == "collections"

    # Check that annotated_for contains the expected functions
    assert "loopfor" in parsed_file.annotated_for
    assert "iterate" in parsed_file.annotated_for
    assert "double_iterate" in parsed_file.annotated_for
    assert "comprehension" in parsed_file.annotated_for
    assert "get_world" in parsed_file.annotated_for


def test_read_function_using_compile():
    # Test reading a specific function from a file
    function = read_function_using_compile(LISTS_PATH, "empty_list_add")

    # Check that it's a code object
    assert hasattr(function, "co_code")
    assert function.co_name == "empty_list_add"

    # Test reading another function
    function = read_function_using_compile(LISTS_PATH, "list_append")
    assert hasattr(function, "co_code")
    assert function.co_name == "list_append"

    # Test reading a function from iteration.py
    function = read_function_using_compile(ITERATION_PATH, "first_shape")
    assert hasattr(function, "co_code")
    assert function.co_name == "first_shape"

    # Test reading a non-existent function
    with pytest.raises(ValueError):
        read_function_using_compile(LISTS_PATH, "non_existent_function")


def test_do_compile():
    # Test compiling a file
    code = do_compile(LISTS_PATH)

    # Check that it's a code object
    assert hasattr(code, "co_code")
    assert code.co_filename == str(LISTS_PATH)

    # Test compiling another file
    code = do_compile(ITERATION_PATH)
    assert hasattr(code, "co_code")
    assert code.co_filename == str(ITERATION_PATH)


def test_parsed_file_structure():
    # Test the structure of a ParsedFile
    parsed_file = read_file(LISTS_PATH)

    # Check that it has the expected attributes
    assert hasattr(parsed_file, "functions")
    assert hasattr(parsed_file, "imports")
    assert hasattr(parsed_file, "annotated_for")

    # Check that the attributes have the expected types
    assert isinstance(parsed_file.functions, dict)
    assert isinstance(parsed_file.imports, dict)
    assert isinstance(parsed_file.annotated_for, dict)

    # Check that the functions dictionary contains callable values
    for func in parsed_file.functions.values():
        assert callable(func)

    # Check that the annotated_for dictionary contains frozenset values
    for annotated in parsed_file.annotated_for.values():
        assert isinstance(annotated, frozenset)
