import pytest
from pythia.dom_typed_pointer import (
    Param,
    Immutable,
    Scope,
    LocationObject,
    make_fields,
    make_graph,
    make_dirty,
    make_dirty_from_keys,
    make_type_map,
    Pointer,
    TypeMap,
    TypedPointer,
    typed_pointer,
    LOCALS,
    GLOBALS,
    immutable,
)
from pythia.dom_concrete import Set, Map
from pythia.graph_utils import Location
from pythia.tac import Var
import pythia.type_system as ts


def test_param_repr():
    param = Param(Var("x", 0))
    assert repr(param) == "@param x"


def test_immutable_repr():
    imm = Immutable(42)
    assert repr(imm) == "@type 42"


def test_scope_repr():
    scope = Scope("test_scope")
    assert repr(scope) == "@scope test_scope"


def test_location_object_repr():
    loc = LocationObject((10, 5))
    assert repr(loc) == "@location 15"


def test_immutable_function():
    # Test that immutable returns a singleton set with an Immutable object
    result = immutable(ts.INT)
    assert isinstance(result, Set)
    assert len(result.as_set()) == 1
    obj = next(iter(result.as_set()))
    assert isinstance(obj, Immutable)

    # Test that calling immutable with the same type returns the same object
    result2 = immutable(ts.INT)
    assert result == result2

    # Test that calling immutable with a different type returns a different object
    result3 = immutable(ts.FLOAT)
    assert result != result3


def test_make_fields():
    # Test empty fields
    fields = make_fields()
    assert isinstance(fields, Map)
    assert len(fields) == 0

    # Test with initial values
    x = Var("x", 0)
    obj = Immutable(42)
    obj_set = Set([obj])
    fields = make_fields({x: obj_set})
    assert len(fields) == 1
    assert fields[x] == obj_set


def test_make_graph():
    # Test empty graph
    graph = make_graph()
    assert isinstance(graph, Map)
    assert len(graph) == 0

    # Test with initial values
    obj1 = Immutable(1)
    obj2 = Immutable(2)
    x = Var("x", 0)
    fields = make_fields({x: Set([obj2])})
    graph = make_graph({obj1: fields})
    assert len(graph) == 1
    assert graph[obj1] == fields
    assert graph[obj1][x] == Set([obj2])


def test_make_dirty():
    # Test empty dirty map
    dirty = make_dirty()
    assert isinstance(dirty, Map)
    assert len(dirty) == 0

    # Test with initial values
    obj = Immutable(42)
    x = Var("x", 0)
    dirty = make_dirty({obj: [x]})
    assert len(dirty) == 1
    assert dirty[obj] == Set([x])


def test_make_dirty_from_keys():
    # Test creating dirty map from keys and field
    obj1 = Immutable(1)
    obj2 = Immutable(2)
    x = Var("x", 0)
    keys = Set([obj1, obj2])
    field = Set([x])

    dirty = make_dirty_from_keys(keys, field)
    assert len(dirty) == 2
    assert dirty[obj1] == field
    assert dirty[obj2] == field


def test_make_type_map():
    # Test empty type map
    type_map = make_type_map()
    assert isinstance(type_map, Map)
    assert len(type_map) == 0

    # Test with initial values
    obj = Immutable(42)
    type_map = make_type_map({obj: ts.INT})
    assert len(type_map) == 1
    assert type_map[obj] == ts.INT

    # Test default value
    obj2 = Immutable(43)
    assert type_map[obj2] == ts.BOTTOM


def test_pointer_init():
    graph = make_graph()
    ptr = Pointer(graph)
    assert ptr.graph == graph


#
# def test_pointer_iter_and_items():
#     obj1 = Immutable(1)
#     obj2 = Immutable(2)
#     graph = make_graph({obj1: make_fields(), obj2: make_fields()})
#     ptr = Pointer(graph)
#
#     # Test __iter__
#     objects = list(ptr)
#     assert len(objects) == 2
#     assert obj1 in objects
#     assert obj2 in objects
#
#     # Test items
#     items = list(ptr.items())
#     assert len(items) == 2
#     assert (obj1, graph[obj1]) in items
#     assert (obj2, graph[obj2]) in items


def test_pointer_getitem():
    obj1 = Immutable(1)
    obj2 = Immutable(2)
    x = Var("x", 0)

    fields = make_fields({x: Set([obj2])})
    graph = make_graph({obj1: fields})
    ptr = Pointer(graph)

    # Test getting fields for an object
    assert ptr[obj1] == fields

    # Test getting objects for a field
    assert ptr[obj1, x] == Set([obj2])

    # Test getting objects for a set of objects and a field
    assert ptr[Set([obj1]), x] == Set([obj2])


def test_pointer_setitem():
    obj1 = Immutable(1)
    obj2 = Immutable(2)
    obj3 = Immutable(3)
    x = Var("x", 0)
    y = Var("y", 0)

    ptr = Pointer(make_graph())

    # Test setting fields for an object
    fields = make_fields({x: Set([obj2])})
    ptr[obj1] = fields
    assert ptr[obj1] == fields

    # Test setting objects for a field
    ptr[obj1, y] = Set([obj3])
    assert ptr[obj1, y] == Set([obj3])

    # Test setting objects for a set of objects and a field
    ptr[Set([obj1]), x] = Set([obj3])
    assert ptr[obj1, x] == Set([obj3])


def test_pointer_update():
    obj1 = Immutable(1)
    obj2 = Immutable(2)
    obj3 = Immutable(3)
    x = Var("x", 0)

    ptr = Pointer(make_graph())
    ptr[obj1, x] = Set([obj2])

    # Test updating a field
    ptr.update(obj1, x, Set([obj3]))
    assert ptr[obj1, x] == Set([obj3])


def test_pointer_keep_keys():
    obj1 = Immutable(1)
    obj2 = Immutable(2)
    x = Var("x", 0)

    ptr = Pointer(make_graph())
    ptr[obj1, x] = Set([obj2])
    ptr[obj2, x] = Set([obj1])

    # Test keeping only specific keys
    ptr.keep_keys([obj1])
    assert obj1 in ptr
    assert obj2 not in ptr


def test_pointer_join():
    obj1 = Immutable(1)
    obj2 = Immutable(2)
    obj3 = Immutable(3)
    x = Var("x", 0)
    y = Var("y", 0)

    ptr1 = Pointer(make_graph())
    ptr1[obj1, x] = Set([obj2])

    ptr2 = Pointer(make_graph())
    ptr2[obj1, y] = Set([obj3])
    ptr2[obj2, x] = Set([obj1])

    # Test joining two pointers
    result = ptr1.join(ptr2)
    assert result[obj1, x] == Set([obj2])
    assert result[obj1, y] == Set([obj3])
    assert result[obj2, x] == Set([obj1])


def test_type_map_init():
    map = make_type_map({Immutable(1): ts.INT})
    type_map = TypeMap(map)
    assert type_map.map == map


def test_type_map_getitem():
    obj1 = Immutable(1)
    obj2 = Immutable(2)

    map = make_type_map({obj1: ts.INT})
    type_map = TypeMap(map)

    # Test getting type for an object
    assert type_map[obj1] == ts.INT

    # Test getting type for a non-existent object (should return BOTTOM)
    assert type_map[obj2] == ts.BOTTOM

    # Test getting type for a set of objects
    assert type_map[Set([obj1])] == ts.INT


def test_type_map_setitem():
    obj1 = Immutable(1)
    obj2 = Immutable(2)

    type_map = TypeMap(make_type_map())

    # Test setting type for an object
    type_map[obj1] = ts.INT
    assert type_map[obj1] == ts.INT

    # Test setting type for a set of objects
    type_map[Set([obj2])] = ts.FLOAT
    assert type_map[obj2] == ts.FLOAT


def test_type_map_join():
    obj1 = Immutable(1)
    obj2 = Immutable(2)

    type_map1 = TypeMap(make_type_map({obj1: ts.INT}))
    type_map2 = TypeMap(make_type_map({obj2: ts.FLOAT}))

    # Test joining two type maps
    result = type_map1.join(type_map2)
    assert result[obj1] == ts.INT
    assert result[obj2] == ts.FLOAT


def test_typed_pointer_function():
    # Test creating a TypedPointer from components
    ptr = Pointer(make_graph())
    types = TypeMap(make_type_map())
    dirty = make_dirty()

    tp = typed_pointer(ptr, types, dirty)
    assert tp.pointers == ptr
    assert tp.types == types
    assert tp.dirty == dirty


def test_typed_pointer_join():
    obj1 = Immutable(1)
    obj2 = Immutable(2)
    x = Var("x", 0)

    # Create first TypedPointer
    ptr1 = Pointer(make_graph())
    ptr1[obj1, x] = Set([obj2])
    types1 = TypeMap(make_type_map({obj1: ts.INT}))
    dirty1 = make_dirty({obj1: [x]})
    tp1 = typed_pointer(ptr1, types1, dirty1)

    # Create second TypedPointer
    ptr2 = Pointer(make_graph())
    ptr2[obj2, x] = Set([obj1])
    types2 = TypeMap(make_type_map({obj2: ts.FLOAT}))
    dirty2 = make_dirty()
    tp2 = typed_pointer(ptr2, types2, dirty2)

    # Test joining two TypedPointers
    result = tp1.join(tp2)
    assert result.pointers[obj1, x] == Set([obj2])
    assert result.pointers[obj2, x] == Set([obj1])
    assert result.types[obj1] == ts.INT
    assert result.types[obj2] == ts.FLOAT
    assert result.dirty[obj1] == Set([x])
