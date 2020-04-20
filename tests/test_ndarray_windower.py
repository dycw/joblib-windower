from __future__ import annotations

from string import ascii_uppercase
from typing import Optional
from typing import Union

from numpy import arange
from numpy import array
from numpy import int64
from numpy import nan
from numpy import ndarray
from numpy.testing import assert_array_equal
from pytest import mark

from joblib_windower import ndarray_windower


@mark.parametrize(
    "window, min_frac, expected",
    [
        (1, None, array(["A", "B", "C", "D", "E"])),
        (1, 0.5, array(["A", "B", "C", "D", "E"])),
        (2, None, array(["A", "AB", "BC", "CD", "DE"])),
        (2, 0.5, array(["A", "AB", "BC", "CD", "DE"])),
        (3, None, array(["A", "AB", "ABC", "BCD", "CDE"])),
        (3, 0.5, array([nan, "AB", "ABC", "BCD", "CDE"])),
    ],
)
def test_returning_non_float(window: int, min_frac: Optional[float], expected: ndarray) -> None:
    @ndarray_windower
    def get_text(x: Union[int64, ndarray]) -> str:
        if window == 1:
            assert isinstance(x, int64)
            return ascii_uppercase[int(x)]
        elif window == 2:
            assert x.shape in [(1,), (2,)]
        elif window == 3:
            assert x.shape in [(1,), (2,), (3,)]
        return "".join([ascii_uppercase[i] for i in x])

    assert_array_equal(
        get_text(x=arange(5), window=window, min_frac=min_frac, n_jobs=None), expected,
    )
