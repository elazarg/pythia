class Counter[T]:
    @update(Counter[T])
    def __setitem__[T](self: Counter[T], key: T, value: int) -> None: ...
    def __getitem__[T](self: Counter[T], key: T) -> int: ...
