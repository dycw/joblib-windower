from __future__ import annotations

from typing import TypeVar
from typing import Union

import numpy as np
from hypothesis import given
from hypothesis.strategies import data
from hypothesis.strategies import DataObject
from hypothesis.strategies import just
from hypothesis.strategies import sampled_from
from hypothesis.strategies import SearchStrategy
from numpy import ndarray

from joblib_windower.joblib_windower import arrays
from joblib_windower.joblib_windower import DTYPES
from joblib_windower.joblib_windower import ShapeInput
from joblib_windower.joblib_windower import SHAPES


T = TypeVar("T")


def maybe_just(x: T) -> Union[T, SearchStrategy[T]]:
    return sampled_from([x, just(x)])


@given(data=data(), dtype=DTYPES, shape=SHAPES)
def test_arrays(data: DataObject, dtype: np.dtype, shape: ShapeInput) -> None:
    array = data.draw(
        arrays(dtypes=data.draw(maybe_just(dtype)), shapes=data.draw(maybe_just(shape))),
    )
    assert isinstance(array, ndarray)
    assert array.dtype == dtype
    assert array.shape == shape
