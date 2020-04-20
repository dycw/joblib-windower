from __future__ import annotations

from numpy import arange
from pandas import DataFrame
from pandas import Series
from pandas.testing import assert_frame_equal

from joblib_windower import ndframe_windower
from tests.test_windower import COLUMNS
from tests.test_windower import INDEX


def test_auto_columns() -> None:
    @ndframe_windower
    def mean(x: DataFrame) -> Series:
        return x.mean(axis=0)

    df = DataFrame(arange(20.0).reshape((5, 4)), index=INDEX, columns=COLUMNS)
    assert_frame_equal(
        mean(df, window=2, n_jobs=None), df.rolling(2, min_periods=0).mean(),
    )
