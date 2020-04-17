from __future__ import annotations

from typing import Tuple
from typing import Union

from hypothesis.extra.numpy import array_shapes
from hypothesis.extra.numpy import arrays as _arrays
from hypothesis.extra.numpy import scalar_dtypes
from hypothesis.strategies import SearchStrategy
from hypothesis_auto_draw import auto_draw_flatmap
from numpy import dtype
from numpy import ndarray


DTypeInput = Union[type, dtype]
ShapeInput = Union[int, Tuple[int, ...]]
DTYPES = scalar_dtypes()
SHAPES = array_shapes()


def arrays(
    *,
    dtypes: Union[DTypeInput, SearchStrategy[DTypeInput]] = DTYPES,
    shapes: Union[ShapeInput, SearchStrategy[ShapeInput]] = SHAPES,
) -> SearchStrategy[ndarray]:
    def inner(dtype: DTypeInput, shape: ShapeInput) -> SearchStrategy[ndarray]:
        return _arrays(dtype, shape)

    return auto_draw_flatmap(inner, dtypes, shapes)
