
class Counter[T]:
    @update(Counter[T])
    def __setitem__(self: Counter[T], key: T, value) -> None: ...

    def __getitem__(self: Counter[T], key: T) -> int: ...
