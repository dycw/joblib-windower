from __future__ import annotations

from string import ascii_uppercase
from typing import Optional
from typing import Union

from numpy import arange
from numpy import int64
from numpy import nan
from pandas import DataFrame
from pandas import Series
from pandas.testing import assert_frame_equal
from pandas.testing import assert_series_equal
from pytest import mark

from joblib_windower import ndframe_windower


INDEX = list(ascii_uppercase[:5])
COLUMNS = list(ascii_uppercase[-4:])
MMM = ["min", "mean", "max"]


@mark.parametrize(
    "window, min_frac, expected",
    [
        (1, None, Series(arange(5), index=INDEX)),
        (1, 0.5, Series(arange(5), index=INDEX)),
        (2, None, Series([0.0, 0.5, 1.5, 2.5, 3.5], index=INDEX)),
        (2, 0.5, Series([0.0, 0.5, 1.5, 2.5, 3.5], index=INDEX)),
        (3, None, Series([0.0, 0.5, 1.0, 2.0, 3.0], index=INDEX)),
        (3, 0.5, Series([nan, 0.5, 1.0, 2.0, 3.0], index=INDEX)),
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


@mark.parametrize(
    "window, min_frac, expected",
    [
        (1, None, Series([1.5, 5.5, 9.5, 13.5, 17.5], index=INDEX)),
        (1, 0.5, Series([1.5, 5.5, 9.5, 13.5, 17.5], index=INDEX)),
        (2, None, Series([1.5, 3.5, 7.5, 11.5, 15.5], index=INDEX)),
        (2, 0.5, Series([1.5, 3.5, 7.5, 11.5, 15.5], index=INDEX)),
        (3, None, Series([1.5, 3.5, 5.5, 9.5, 13.5], index=INDEX)),
        (3, 0.5, Series([nan, 3.5, 5.5, 9.5, 13.5], index=INDEX)),
    ],
)
def test_2D_to_float(window: int, min_frac: Optional[float], expected: Series) -> None:
    @ndframe_windower
    def mean(x: Union[Series, DataFrame]) -> float:
        if window == 1:
            assert x.shape == (4,)
        elif window == 2:
            assert x.shape in [(1, 4), (2, 4)]
        elif window == 3:
            assert x.shape in [(1, 4), (2, 4), (3, 4)]
        return x.to_numpy().mean()

    assert_series_equal(
        mean(
            DataFrame(arange(20).reshape((5, 4)), index=INDEX, columns=COLUMNS),
            window=window,
            min_frac=min_frac,
            n_jobs=None,
        ),
        expected,
    )


@mark.parametrize(
    "window, min_frac, expected",
    [
        (
            1,
            None,
            DataFrame(
                [
                    [0.0, 1.5, 3.0],
                    [4.0, 5.5, 7.0],
                    [8.0, 9.5, 11.0],
                    [12.0, 13.5, 15.0],
                    [16.0, 17.5, 19.0],
                ],
                index=INDEX,
                columns=MMM,
            ),
        ),
        (
            1,
            0.5,
            DataFrame(
                [
                    [0.0, 1.5, 3.0],
                    [4.0, 5.5, 7.0],
                    [8.0, 9.5, 11.0],
                    [12.0, 13.5, 15.0],
                    [16.0, 17.5, 19.0],
                ],
                index=INDEX,
                columns=MMM,
            ),
        ),
        (
            2,
            None,
            DataFrame(
                [
                    [0.0, 1.5, 3.0],
                    [0.0, 3.5, 7.0],
                    [4.0, 7.5, 11.0],
                    [8.0, 11.5, 15.0],
                    [12.0, 15.5, 19.0],
                ],
                index=INDEX,
                columns=MMM,
            ),
        ),
        (
            2,
            0.5,
            DataFrame(
                [
                    [0.0, 1.5, 3.0],
                    [0.0, 3.5, 7.0],
                    [4.0, 7.5, 11.0],
                    [8.0, 11.5, 15.0],
                    [12.0, 15.5, 19.0],
                ],
                index=INDEX,
                columns=MMM,
            ),
        ),
        (
            3,
            None,
            DataFrame(
                [
                    [0.0, 1.5, 3.0],
                    [0.0, 3.5, 7.0],
                    [0.0, 5.5, 11.0],
                    [4.0, 9.5, 15.0],
                    [8.0, 13.5, 19.0],
                ],
                index=INDEX,
                columns=MMM,
            ),
        ),
        (
            3,
            0.5,
            DataFrame(
                [
                    [nan, nan, nan],
                    [0.0, 3.5, 7.0],
                    [0.0, 5.5, 11.0],
                    [4.0, 9.5, 15.0],
                    [8.0, 13.5, 19.0],
                ],
                index=INDEX,
                columns=MMM,
            ),
        ),
    ],
)
def test_2D_to_1D(window: int, min_frac: Optional[float], expected: DataFrame) -> None:
    @ndframe_windower
    def summary(x: Union[Series, DataFrame]) -> Series:
        if window == 1:
            assert x.shape == (4,)
        elif window == 2:
            assert x.shape in [(1, 4), (2, 4)]
        elif window == 3:
            assert x.shape in [(1, 4), (2, 4), (3, 4)]
        return Series([x.to_numpy().min(), x.to_numpy().mean(), x.to_numpy().max()], index=MMM)

    assert_frame_equal(
        summary(
            x=DataFrame(arange(20).reshape((5, 4)), index=INDEX, columns=COLUMNS),
            columns=MMM,
            window=window,
            min_frac=min_frac,
            n_jobs=None,
        ),
        expected,
    )
