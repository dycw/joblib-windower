from __future__ import annotations

from string import ascii_uppercase
from typing import Optional
from typing import TypeVar

from numpy import arange
from numpy import array
from numpy import nan
from numpy import ndarray
from numpy.testing import assert_array_equal
from pytest import mark

from joblib_windower import windower


T = TypeVar("T")


@mark.parametrize(
    "window, expected", [(1, arange(5)), (2, array([0.0, 0.5, 1.5, 2.5, 3.5]))],
)
@mark.parametrize("min_frac", [None, 0.5])
def test_window_1_and_2(window: int, min_frac: Optional[float], expected: ndarray) -> None:
    @windower
    def mean(x: ndarray) -> float:
        return x.mean()

    assert_array_equal(
        mean(arange(5), window=window, min_frac=min_frac), expected,
    )


@mark.parametrize("min_frac", [None, 0.5])
def test_window_3(min_frac: Optional[float]) -> None:
    @windower
    def mean(x: ndarray) -> float:
        return x.mean()

    first = 0.0 if min_frac is None else nan
    assert_array_equal(
        mean(x=arange(5), window=3, min_frac=min_frac, n_jobs=None),
        array([first, 0.5, 1.0, 2.0, 3.0]),
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
