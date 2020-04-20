from __future__ import annotations

from pathlib import Path
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
        (1, None, arange(5)),
        (1, 0.5, arange(5)),
        (2, None, array([0.0, 0.5, 1.5, 2.5, 3.5])),
        (2, 0.5, array([0.0, 0.5, 1.5, 2.5, 3.5])),
        (3, None, array([0.0, 0.5, 1.0, 2.0, 3.0])),
        (3, 0.5, array([nan, 0.5, 1.0, 2.0, 3.0])),
    ],
)
def test_1D_to_float(window: int, min_frac: Optional[float], expected: ndarray) -> None:
    @ndarray_windower
    def mean(x: Union[int64, ndarray]) -> float:
        if window == 1:
            assert isinstance(x, int64)
        elif window == 2:
            assert x.shape in [(1,), (2,)]
        elif window == 3:
            assert x.shape in [(1,), (2,), (3,)]

        return x.mean()

    assert_array_equal(
        mean(arange(5), window=window, min_frac=min_frac, n_jobs=None), expected,
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
    @ndarray_windower
    def mean(x: ndarray) -> float:
        if window == 1:
            assert x.shape == (4,)
        elif window == 2:
            assert x.shape in [(1, 4), (2, 4)]
        elif window == 3:
            assert x.shape in [(1, 4), (2, 4), (3, 4)]

        return x.mean()

    assert_array_equal(
        mean(arange(20).reshape((5, 4)), window=window, min_frac=min_frac, n_jobs=None), expected,
    )


@mark.parametrize(
    "window, min_frac, expected",
    [
        (
            1,
            None,
            array(
                [
                    [0.0, 1.5, 3.0],
                    [4.0, 5.5, 7.0],
                    [8.0, 9.5, 11.0],
                    [12.0, 13.5, 15.0],
                    [16.0, 17.5, 19.0],
                ],
            ),
        ),
        (
            1,
            0.5,
            array(
                [
                    [0.0, 1.5, 3.0],
                    [4.0, 5.5, 7.0],
                    [8.0, 9.5, 11.0],
                    [12.0, 13.5, 15.0],
                    [16.0, 17.5, 19.0],
                ],
            ),
        ),
        (
            2,
            None,
            array(
                [
                    [0.0, 1.5, 3.0],
                    [0.0, 3.5, 7.0],
                    [4.0, 7.5, 11.0],
                    [8.0, 11.5, 15.0],
                    [12.0, 15.5, 19.0],
                ],
            ),
        ),
        (
            2,
            0.5,
            array(
                [
                    [0.0, 1.5, 3.0],
                    [0.0, 3.5, 7.0],
                    [4.0, 7.5, 11.0],
                    [8.0, 11.5, 15.0],
                    [12.0, 15.5, 19.0],
                ],
            ),
        ),
        (
            3,
            None,
            array(
                [
                    [0.0, 1.5, 3.0],
                    [0.0, 3.5, 7.0],
                    [0.0, 5.5, 11.0],
                    [4.0, 9.5, 15.0],
                    [8.0, 13.5, 19.0],
                ],
            ),
        ),
        (
            3,
            0.5,
            array(
                [
                    [nan, nan, nan],
                    [0.0, 3.5, 7.0],
                    [0.0, 5.5, 11.0],
                    [4.0, 9.5, 15.0],
                    [8.0, 13.5, 19.0],
                ],
            ),
        ),
    ],
)
def test_2D_to_1D(window: int, min_frac: Optional[float], expected: ndarray) -> None:
    @ndarray_windower
    def summary(x: ndarray) -> ndarray:
        if window == 1:
            assert x.shape == (4,)
        elif window == 2:
            assert x.shape in [(1, 4), (2, 4)]
        elif window == 3:
            assert x.shape in [(1, 4), (2, 4), (3, 4)]
        return array([x.min(), x.mean(), x.max()])

    assert_array_equal(
        summary(x=arange(20).reshape((5, 4)), window=window, min_frac=min_frac, n_jobs=None),
        expected,
    )


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


def test_custom_temp_dir(tmp_path: Path) -> None:
    @ndarray_windower(temp_dir=tmp_path)
    def mean(x: ndarray) -> float:
        return x.mean()

    assert_array_equal(
        mean(arange(5), window=2, n_jobs=None), array([0.0, 0.5, 1.5, 2.5, 3.5]),
    )
