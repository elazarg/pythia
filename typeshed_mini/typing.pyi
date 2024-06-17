
class Any: ...

class Iterator[T](Protocol):
    @new
    @update(Iterator[T])
    def __next__(self: Iterator[T]) -> T: ...

class Iterable[T](Protocol):
    @new
    def __iter__(self: Iterable[T]) -> Iterator[T]: ...

class Indexable[T](Protocol):
    @new
    def __getitem__(self: Indexable[T], slice) -> T: ...
