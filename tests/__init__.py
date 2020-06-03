from __future__ import annotations

from typing import Callable
from typing import cast
from typing import TypeVar

from pytest import mark


TestLike = TypeVar("TestLike", bound=Callable[..., None])
parametrize = cast(
    Callable[..., Callable[[TestLike], TestLike]], mark.parametrize,
)
