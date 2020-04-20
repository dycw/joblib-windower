from __future__ import annotations

from numpy import arange
from pandas import DataFrame
from pandas import Index
from pandas import Series
from pandas.testing import assert_frame_equal
from pytest import mark

from joblib_windower import ndframe_windower
from tests.test_windower import COLUMNS
from tests.test_windower import INDEX


NEW_COLUMNS = [f"new_{column}" for column in COLUMNS]


@mark.parametrize(
    "columns, expected", [(None, COLUMNS), (NEW_COLUMNS, NEW_COLUMNS)],
)
def test_columns(columns: Index, expected: Index) -> None:
    @ndframe_windower
    def mean(x: DataFrame) -> Series:
        return x.mean(axis=0)

    assert_frame_equal(
        mean(
            DataFrame(arange(20.0).reshape((5, 4)), index=INDEX, columns=COLUMNS),
            window=2,
            n_jobs=None,
            columns=columns,
        ),
        DataFrame(
            [
                [0.0, 1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0, 13.0],
                [14.0, 15.0, 16.0, 17.0],
            ],
            index=INDEX,
            columns=expected,
        ),
    )
