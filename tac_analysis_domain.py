from typing import TypeVar, Protocol, Type

T = TypeVar('T')


# mix of domain and analysis-specific choice of operations
# nothing here really works...
class AbstractDomain(Protocol):
    @classmethod
    def is_forward(cls) -> bool:
        ...

    @classmethod
    def top(cls: Type[T]) -> T:
        ...

    @classmethod
    def bottom(cls: Type[T]) -> T:
        ...

    def join(self: T, other) -> T:
        ...

    def meet(self: T, other) -> T:
        ...

    def is_bottom(self) -> bool:
        ...

    def is_top(self) -> bool:
        ...

    def transfer(self, ins: T) -> None:
        ...
