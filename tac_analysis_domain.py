from typing import TypeVar, Protocol

T = TypeVar('T')


# mix of domain and analysis-specific choice of operations
# nothing here really works...
class AbstractDomain(Protocol):

    def transfer(self):
        ...

    @classmethod
    def top(cls: T) -> T:
        ...

    @classmethod
    def bottom(cls: T) -> T:
        ...

    def join(self: T, other) -> T:
        ...

    def meet(self: T, other) -> T:
        ...

    def is_bottom(self) -> bool:
        ...

    def is_top(self) -> bool:
        ...
