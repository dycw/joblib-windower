from __future__ import annotations

from string import ascii_uppercase
from typing import Optional

from numpy import arange
from numpy import array
from numpy import int64
from numpy import nan
from numpy import ndarray
from numpy.testing import assert_array_equal
from pytest import mark

from joblib_windower import windower


@mark.parametrize(
    "window, min_frac, expected",
    [
        (1, None, arange(5)),
        (1, 0.5, arange(5)),
        (2, None, array([0.0, 0.5, 1.5, 2.5, 3.5])),
        (2, 0.5, array([0.0, 0.5, 1.5, 2.5, 3.5])),
        (3, None, array([0.0, 0.5, 1.0, 2.0, 3.0])),
        (3, 0.5, array([nan, 0.5, 1.0, 2.0, 3.0])),
    ],
)
def test_1D_to_float(window: int, min_frac: Optional[float], expected: ndarray) -> None:
    @windower
    def mean(x: ndarray) -> float:
        if window == 1:
            assert isinstance(x, int64)
        elif window == 2:
            assert x.shape in [(1,), (2,)]
        elif window == 3:
            assert x.shape in [(1,), (2,), (3,)]

        return x.mean()

    assert_array_equal(
        mean(arange(5), window=window, min_frac=min_frac), expected,
    )


@mark.parametrize(
    "window, min_frac, expected",
    [
        (1, None, array([1.5, 5.5, 9.5, 13.5, 17.5])),
        (1, 0.5, array([1.5, 5.5, 9.5, 13.5, 17.5])),
        (2, None, array([1.5, 3.5, 7.5, 11.5, 15.5])),
        (2, 0.5, array([1.5, 3.5, 7.5, 11.5, 15.5])),
        (3, None, array([1.5, 3.5, 5.5, 9.5, 13.5])),
        (3, 0.5, array([nan, 3.5, 5.5, 9.5, 13.5])),
    ],
)
def test_2D_to_float(window: int, min_frac: Optional[float], expected: ndarray) -> None:
    @windower
    def mean(x: ndarray) -> float:
        if window == 1:
            assert x.shape == (4,)
        elif window == 2:
            assert x.shape in [(1, 4), (2, 4)]
        elif window == 3:
            assert x.shape in [(1, 4), (2, 4), (3, 4)]

        return x.mean()

    assert_array_equal(
        mean(arange(20).reshape((5, 4)), window=window, min_frac=min_frac), expected,
    )


@mark.parametrize(
    "window, expected",
    [(1, array(["A", "B", "C", "D", "E"])), (2, array(["A", "AB", "BC", "CD", "DE"]))],
)
@mark.parametrize("min_frac", [None, 0.5])
def test_returning_non_float(window: int, min_frac: Optional[float], expected: ndarray) -> None:
    @windower
    def get_letter(x: ndarray) -> str:
        try:
            return ascii_uppercase[int(x)]
        except TypeError:
            return "".join([ascii_uppercase[i] for i in x])

    assert_array_equal(
        get_letter(x=arange(5), window=window, min_frac=min_frac, n_jobs=None), expected,
    )


@mark.parametrize("min_frac", [None, 0.5])
def test_returning_1D_array(min_frac: Optional[float]) -> None:
    @windower
    def mean(x: ndarray) -> ndarray:
        return array([x.min(), x.mean(), x.max()])

    first = 0.0 if min_frac is None else nan
    assert_array_equal(
        mean(x=arange(5), window=3, min_frac=min_frac, n_jobs=None),
        array(
            [
                [first, first, first],
                [0.0, 0.5, 1.0],
                [0.0, 1.0, 2.0],
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0],
            ],
        ),
    )


@mark.parametrize("min_frac", [None, 0.5])
def test_2D_to_1D(min_frac: Optional[float]) -> None:
    @windower
    def mean(x: ndarray) -> ndarray:
        assert x.shape in [(1, 5), (2, 5), (3, 5)]
        return x.mean(axis=0)

    first = [0.0, 1.0, 2.0, 3.0, 4.0] if min_frac is None else [nan, nan, nan, nan, nan]
    assert_array_equal(
        mean(x=arange(5 * 5).reshape((5, 5)), window=3, min_frac=min_frac, n_jobs=None),
        array(
            [
                first,
                [2.5, 3.5, 4.5, 5.5, 6.5],
                [5.0, 6.0, 7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0, 13.0, 14.0],
                [15.0, 16.0, 17.0, 18.0, 19.0],
            ],
        ),
    )
