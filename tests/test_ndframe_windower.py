from __future__ import annotations

from string import ascii_uppercase
from typing import Optional
from typing import Union

from numpy import arange
from numpy import int64
from pandas import Series
from pandas.testing import assert_series_equal
from pytest import mark

from joblib_windower import ndframe_windower


INDEX = list(ascii_uppercase[:5])


@mark.parametrize(
    "window, min_frac, expected",
    [
        (1, None, Series(arange(5), index=INDEX)),
        # (1, 0.5, arange(5)),
        # (2, None, array([0.0, 0.5, 1.5, 2.5, 3.5])),
        # (2, 0.5, array([0.0, 0.5, 1.5, 2.5, 3.5])),
        # (3, None, array([0.0, 0.5, 1.0, 2.0, 3.0])),
        # (3, 0.5, array([nan, 0.5, 1.0, 2.0, 3.0])),
    ],
)
def test_1D_to_float(window: int, min_frac: Optional[float], expected: Series) -> None:
    @ndframe_windower
    def mean(x: Union[int64, Series]) -> float:
        if window == 1:
            assert isinstance(x, int64)
            return x
        elif window == 2:
            assert x.shape in [(1,), (2,)]
        elif window == 3:
            assert x.shape in [(1,), (2,), (3,)]
        return x.mean()

    assert_series_equal(
        mean(x=Series(arange(5), index=INDEX), window=window, min_frac=min_frac, n_jobs=None),
        expected,
    )
