from __future__ import annotations

from typing import Any
from typing import Optional

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

from joblib_windower.slide_ndframes import get_maybe_dataframe_columns
from joblib_windower.slide_ndframes import get_ndframe_spec
from joblib_windower.slide_ndframes import NDFrameSpec
from joblib_windower.utilities import datetime64ns
from joblib_windower.utilities import NaT
from joblib_windower.utilities import width_to_str_dtype


@mark.parametrize(
    "obj, expected",
    [
        (None, None),
        (Index([0, 1, 2]), None),
        (Series(0, index=list("abc")), None),
        (DataFrame(0, index=list("abc"), columns=list("xyz")), Index(list("xyz"))),
    ],
)
def test_get_maybe_dataframe_columns(obj: Any, expected: Optional[Index]) -> None:
    result = get_maybe_dataframe_columns(obj)
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
