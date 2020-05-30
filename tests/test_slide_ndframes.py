from __future__ import annotations

from re import escape
from typing import Any
from typing import List
from typing import Optional
from typing import Type
from typing import Union

from numpy import dtype
from numpy import float32
from numpy import float64
from numpy import int32
from numpy import int64
from numpy import nan
from pandas import DataFrame
from pandas import Index
from pandas import Series
from pandas.testing import assert_index_equal
from pytest import mark
from pytest import raises

from joblib_windower.errors import DistinctIndicesError
from joblib_windower.slide_ndarrays import datetime64ns
from joblib_windower.slide_ndarrays import NaT
from joblib_windower.slide_ndarrays import width_to_str_dtype
from joblib_windower.slide_ndframes import get_maybe_dataframe_columns
from joblib_windower.slide_ndframes import get_maybe_ndframe_index
from joblib_windower.slide_ndframes import get_ndframe_spec
from joblib_windower.slide_ndframes import get_unique_index
from joblib_windower.slide_ndframes import NDFrameSpec


@mark.parametrize(
    "obj, expected",
    [
        (None, None),
        (Index([0, 1, 2]), None),
        (Series(0, index=list("abc")), None),
        (
            DataFrame(0, index=list("abc"), columns=list("xyz")),
            Index(list("xyz")),
        ),
    ],
)
def test_get_maybe_dataframe_columns(
    obj: Any, expected: Optional[Index],
) -> None:
    result = get_maybe_dataframe_columns(obj)
    if isinstance(result, Index) and isinstance(expected, Index):
        assert_index_equal(result, expected)
    else:
        assert result is None
        assert expected is None


@mark.parametrize(
    "obj, expected",
    [
        (None, None),
        (Index([0, 1, 2]), None),
        (Series(0, index=list("abc")), Index(list("abc"))),
        (
            DataFrame(0, index=list("abc"), columns=list("xyz")),
            Index(list("abc")),
        ),
    ],
)
def test_get_maybe_ndframe_index(obj: Any, expected: Optional[Index]) -> None:
    result = get_maybe_ndframe_index(obj)
    if isinstance(result, Index) and isinstance(expected, Index):
        assert_index_equal(result, expected)
    else:
        assert result is None
        assert expected is None


@mark.parametrize(
    "dtype, expected",
    [
        (dtype(bool), NDFrameSpec(dtype=dtype(object), masked=nan)),
        (dtype(int), NDFrameSpec(dtype=dtype(float), masked=nan)),
        (dtype(int32), NDFrameSpec(dtype=dtype(float), masked=nan)),
        (dtype(int64), NDFrameSpec(dtype=dtype(float), masked=nan)),
        (dtype(float), NDFrameSpec(dtype=dtype(float), masked=nan)),
        (dtype(float32), NDFrameSpec(dtype=dtype(float), masked=nan)),
        (dtype(float64), NDFrameSpec(dtype=dtype(float), masked=nan)),
        (width_to_str_dtype(1), NDFrameSpec(dtype=dtype(object), masked=nan)),
        (width_to_str_dtype(2), NDFrameSpec(dtype=dtype(object), masked=nan)),
        (datetime64ns, NDFrameSpec(dtype=datetime64ns, masked=NaT)),
    ],
)
def test_get_ndframe_spec(dtype: dtype, expected: NDFrameSpec) -> None:
    assert get_ndframe_spec(dtype) == expected


@mark.parametrize(
    "indices, expected",
    [
        ([Index(list("abc"))], Index(list("abc"))),
        ([Index(list("abc")), Index(list("abc"))], Index(list("abc"))),
        (
            [Index(list("abc"), name="x"), Index(list("abc"), name="y")],
            Index(list("abc")),
        ),
        ([Index(list("abc")), Index(list("xyz"))], DistinctIndicesError),
    ],
)
def test_get_unique_index(
    indices: List[Index], expected: Union[Index, Type[DistinctIndicesError]],
) -> None:
    if isinstance(expected, Index):
        result = get_unique_index(indices)
        assert isinstance(result, Index)
        assert_index_equal(result, expected)
    else:
        with raises(expected, match=escape(", ".join(map(str, indices)))):
            get_unique_index(indices)
