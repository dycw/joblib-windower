from __future__ import annotations

from pathlib import Path
from string import ascii_uppercase
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import TypeVar
from typing import Union

from numpy import arange
from numpy import array
from numpy import float64
from numpy import int64
from numpy import nan
from numpy import ndarray
from numpy.testing import assert_array_equal
from pandas import DataFrame
from pandas import Series
from pandas.testing import assert_frame_equal
from pandas.testing import assert_series_equal
from pytest import mark

from joblib_windower import ndarray_windower
from joblib_windower import ndframe_windower
from joblib_windower.ndframe_windower import to_numpy


INDEX = list(ascii_uppercase[:5])
COLUMNS = list(ascii_uppercase[-4:])
MMM_COLUMNS = ["min", "mean", "max"]
T = TypeVar("T")


def identity(x: T) -> T:
    return x


@mark.parametrize(
    "window, min_frac, expected",
    [
        (1, None, Series(arange(5.0), index=INDEX)),
        (1, 0.5, Series(arange(5.0), index=INDEX)),
        (2, None, Series([0.0, 0.5, 1.5, 2.5, 3.5], index=INDEX)),
        (2, 0.5, Series([0.0, 0.5, 1.5, 2.5, 3.5], index=INDEX)),
        (3, None, Series([0.0, 0.5, 1.0, 2.0, 3.0], index=INDEX)),
        (3, 0.5, Series([nan, 0.5, 1.0, 2.0, 3.0], index=INDEX)),
    ],
)
@mark.parametrize(
    "windower, transform, asserter",
    [
        (ndarray_windower, to_numpy, assert_array_equal),
        (ndframe_windower, identity, assert_series_equal),
    ],
)
def test_1D_to_float(
    window: int,
    min_frac: Optional[float],
    expected: Series,
    windower: Any,
    transform: Callable[[Any], Any],
    asserter: Callable[[Any, Any], None],
) -> None:
    @windower
    def mean(x: Union[float64, ndarray, Series]) -> float:
        if isinstance(x, float64):
            return x
        elif isinstance(x, (ndarray, Series)):
            return x.mean()
        else:
            raise TypeError(x)

    asserter(
        mean(
            x=transform(Series(arange(5.0), index=INDEX)),
            window=window,
            min_frac=min_frac,
            n_jobs=None,
        ),
        transform(expected),
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
@mark.parametrize(
    "windower, transform, asserter",
    [
        (ndarray_windower, to_numpy, assert_array_equal),
        (ndframe_windower, identity, assert_series_equal),
    ],
)
def test_2D_to_float(
    window: int,
    min_frac: Optional[float],
    expected: Series,
    windower: Any,
    transform: Callable[[Any], Any],
    asserter: Callable[[Any, Any], None],
) -> None:
    @windower
    def mean(x: Union[ndarray, Series, DataFrame]) -> float:
        if isinstance(x, (ndarray, Series)):
            return x.mean()
        elif isinstance(x, DataFrame):
            return x.to_numpy().mean()
        else:
            raise TypeError(x)

    asserter(
        mean(
            transform(DataFrame(arange(20.0).reshape((5, 4)), index=INDEX, columns=COLUMNS)),
            window=window,
            min_frac=min_frac,
            n_jobs=None,
        ),
        transform(expected),
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
                columns=MMM_COLUMNS,
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
                columns=MMM_COLUMNS,
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
                columns=MMM_COLUMNS,
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
                columns=MMM_COLUMNS,
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
                columns=MMM_COLUMNS,
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
                columns=MMM_COLUMNS,
            ),
        ),
    ],
)
@mark.parametrize(
    "windower, transform, asserter, kwargs",
    [
        (ndarray_windower, to_numpy, assert_array_equal, {}),
        (ndframe_windower, identity, assert_frame_equal, {"columns": MMM_COLUMNS}),
    ],
)
def test_2D_to_1D(
    window: int,
    min_frac: Optional[float],
    expected: DataFrame,
    windower: Any,
    transform: Callable[[Any], Any],
    asserter: Callable[[Any, Any], None],
    kwargs: Dict[str, Any],
) -> None:
    @windower
    def summary(x: Union[ndarray, Series, DataFrame]) -> Union[ndarray, Series]:
        if isinstance(x, ndarray):
            return array([x.min(), x.mean(), x.max()])
        elif isinstance(x, Series):
            return Series([x.min(), x.mean(), x.max()], index=MMM_COLUMNS)
        elif isinstance(x, DataFrame):
            x_array = to_numpy(x)
            return Series([x_array.min(), x_array.mean(), x_array.max()], index=MMM_COLUMNS)
        else:
            raise TypeError(x)

    asserter(
        summary(
            x=transform(DataFrame(arange(20.0).reshape((5, 4)), index=INDEX, columns=COLUMNS)),
            window=window,
            min_frac=min_frac,
            n_jobs=None,
            **kwargs,
        ),
        transform(expected),
    )


@mark.parametrize(
    "window, min_frac, expected",
    [
        (1, None, Series(["A", "B", "C", "D", "E"], index=INDEX)),
        (1, 0.5, Series(["A", "B", "C", "D", "E"], index=INDEX)),
        (2, None, Series(["A", "AB", "BC", "CD", "DE"], index=INDEX)),
        (2, 0.5, Series(["A", "AB", "BC", "CD", "DE"], index=INDEX)),
        (3, None, Series(["A", "AB", "ABC", "BCD", "CDE"], index=INDEX)),
        (3, 0.5, Series(["nan", "AB", "ABC", "BCD", "CDE"], index=INDEX)),
    ],
)
@mark.parametrize(
    "windower, transform, asserter",
    [
        (ndarray_windower, to_numpy, assert_array_equal),
        (ndframe_windower, identity, assert_series_equal),
    ],
)
def test_returning_non_float(
    window: int,
    min_frac: Optional[float],
    expected: ndarray,
    windower: Any,
    transform: Callable[[Any], Any],
    asserter: Callable[[Any, Any], None],
) -> None:
    @windower
    def get_text(x: Union[int64, ndarray, Series]) -> str:
        if isinstance(x, int64):
            return ascii_uppercase[x]
        elif isinstance(x, (ndarray, Series)):
            return "".join([ascii_uppercase[i] for i in x])
        else:
            raise TypeError(x)

    asserter(
        get_text(
            x=transform(Series(arange(5), index=INDEX)),
            window=window,
            min_frac=min_frac,
            n_jobs=None,
        ),
        transform(expected),
    )


@mark.parametrize(
    "windower, transform, asserter",
    [
        (ndarray_windower, to_numpy, assert_array_equal),
        (ndframe_windower, identity, assert_series_equal),
    ],
)
def test_custom_temp_dir(
    windower: Any,
    transform: Callable[[Any], Any],
    asserter: Callable[[Any, Any], None],
    tmp_path: Path,
) -> None:
    @windower(temp_dir=tmp_path)
    def mean(x: Union[ndarray, Series]) -> float:
        return x.mean()

    asserter(
        mean(transform(Series(arange(5.0), index=INDEX)), window=2, n_jobs=None),
        transform(Series([0.0, 0.5, 1.5, 2.5, 3.5], index=INDEX)),
    )
