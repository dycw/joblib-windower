from __future__ import annotations

import datetime as dt
from typing import Any

from attr import attrs
from functional_itertools import CList
from functional_itertools import CSet
from numpy import array
from numpy import bool_
from numpy import datetime64
from numpy import dtype
from numpy import float32
from numpy import float64
from numpy import int32
from numpy import int64
from numpy import nan
from numpy import ndarray
from numpy import timedelta64
from numpy.ma import arange
from pandas import DataFrame
from pandas import Float64Index
from pandas import Index
from pandas import Int64Index
from pandas import Series
from pytest import mark

from joblib_windower.slide_ndarrays import OutputSpec
from joblib_windower.utilities import are_equal_arrays
from joblib_windower.utilities import are_equal_objects
from joblib_windower.utilities import datetime64ns
from joblib_windower.utilities import DEFAULT_STR_LEN_FACTOR
from joblib_windower.utilities import get_output_spec
from joblib_windower.utilities import is_not_none
from joblib_windower.utilities import merge_dtypes
from joblib_windower.utilities import merge_str_dtypes
from joblib_windower.utilities import pandas_obj_to_ndarray
from joblib_windower.utilities import primitive_to_dtype
from joblib_windower.utilities import str_dtype_to_width
from joblib_windower.utilities import timedelta64ns
from joblib_windower.utilities import trim_str_dtype
from joblib_windower.utilities import width_to_str_dtype


@attrs(auto_attribs=True)
class PrimitiveToDtypeCase:
    value: Any
    dtype: dtype


PRIMITIVE_TO_DTYPE_CASES = CList(
    [
        PrimitiveToDtypeCase(value=True, dtype=dtype(bool)),
        PrimitiveToDtypeCase(value=bool_(True), dtype=dtype(bool)),
        PrimitiveToDtypeCase(value=0, dtype=dtype(int)),
        PrimitiveToDtypeCase(value=int32(0), dtype=dtype(int32)),
        PrimitiveToDtypeCase(value=int64(0), dtype=dtype(int64)),
        PrimitiveToDtypeCase(value=0.0, dtype=dtype(float)),
        PrimitiveToDtypeCase(value=float32(0.0), dtype=dtype(float32)),
        PrimitiveToDtypeCase(value=float64(0.0), dtype=dtype(float64)),
        PrimitiveToDtypeCase(value="a", dtype=width_to_str_dtype(DEFAULT_STR_LEN_FACTOR)),
        PrimitiveToDtypeCase(value="ab", dtype=width_to_str_dtype(2 * DEFAULT_STR_LEN_FACTOR)),
        PrimitiveToDtypeCase(value=dt.date.today(), dtype=datetime64ns),
        PrimitiveToDtypeCase(value=datetime64(0, "ns"), dtype=datetime64ns),
        PrimitiveToDtypeCase(value=dt.timedelta(days=0), dtype=timedelta64ns),
        PrimitiveToDtypeCase(value=timedelta64(0, "ns"), dtype=timedelta64ns),
    ],
)


@mark.parametrize(
    "value, expected",
    PRIMITIVE_TO_DTYPE_CASES.map(lambda x: (x.value, OutputSpec(dtype=x.dtype, shape=(5,)))).chain(
        [
            # ndarrays
            (array([0.0, 1.0]), OutputSpec(dtype=dtype(float), shape=(5, 2))),
            (array([0.0, 1.0], dtype=float32), OutputSpec(dtype=dtype(float32), shape=(5, 2))),
            (arange(6.0).reshape((2, 3)), OutputSpec(dtype=dtype(float), shape=(5, 2, 3))),
            (
                arange(6.0, dtype=float32).reshape((2, 3)),
                OutputSpec(dtype=dtype(float32), shape=(5, 2, 3)),
            ),
            # sequences
            ((True, False), OutputSpec(dtype=dtype(bool), shape=(5, 2))),
            ((0, 1, 2), OutputSpec(dtype=dtype(int), shape=(5, 3))),
            ((int32(0), int32(1), int32(2)), OutputSpec(dtype=dtype(int32), shape=(5, 3))),
            ((int64(0), int64(1), int64(2)), OutputSpec(dtype=dtype(int64), shape=(5, 3))),
            ((0.0, 1.0, 2.0), OutputSpec(dtype=dtype(float), shape=(5, 3))),
            (
                (float32(0.0), float32(1.0), float32(2.0)),
                OutputSpec(dtype=dtype(float32), shape=(5, 3)),
            ),
            (
                (float64(0.0), float64(1.0), float64(2.0)),
                OutputSpec(dtype=dtype(float64), shape=(5, 3)),
            ),
            (
                ["a", "b", "c"],
                OutputSpec(dtype=width_to_str_dtype(DEFAULT_STR_LEN_FACTOR), shape=(5, 3)),
            ),
            (
                ["a", "ab", "abc"],
                OutputSpec(dtype=width_to_str_dtype(3 * DEFAULT_STR_LEN_FACTOR), shape=(5, 3)),
            ),
            (
                [dt.date.today(), dt.date.today() + dt.timedelta(days=1)],
                OutputSpec(dtype=datetime64ns, shape=(5, 2)),
            ),
            (
                [dt.timedelta(), dt.timedelta(days=1), dt.timedelta(days=2)],
                OutputSpec(dtype=timedelta64ns, shape=(5, 3)),
            ),
            # series
            (Series([True, False]), OutputSpec(dtype=dtype(bool), shape=(5, 2))),
            (Series([True, False, nan]), OutputSpec(dtype=dtype(bool), shape=(5, 3))),
            (Series([0, 1, 2]), OutputSpec(dtype=dtype(int), shape=(5, 3))),
            (Series([0.0, 1.0, 2.0]), OutputSpec(dtype=dtype(float), shape=(5, 3))),
            (Series([0.0, 1.0, 2.0, nan]), OutputSpec(dtype=dtype(float), shape=(5, 4))),
            (
                Series(["a", "b", "c"]),
                OutputSpec(dtype=width_to_str_dtype(DEFAULT_STR_LEN_FACTOR), shape=(5, 3)),
            ),
            (
                Series(["a", "b", "c", nan]),
                OutputSpec(dtype=width_to_str_dtype(DEFAULT_STR_LEN_FACTOR), shape=(5, 4)),
            ),
        ],
    ),
)
def test_get_output_spec(value: Any, expected: OutputSpec) -> None:
    assert get_output_spec(value, 5) == expected


@mark.parametrize("x, expected", [(None, False), (0, True)])
def test_is_not_none(x: Any, expected: bool) -> None:
    assert is_not_none(x) == expected


@mark.parametrize(
    "x, expected",
    [
        (CSet({dtype(int), dtype(float)}), CSet({dtype(int), dtype(float)})),
        (CSet({dtype("U1"), dtype("U10"), dtype(int)}), CSet({dtype("U10"), dtype(int)})),
    ],
)
def test_merge_dtypes(x: CSet[dtype], expected: CSet[dtype]) -> None:
    assert merge_dtypes(x) == expected


@mark.parametrize(
    "x, expected",
    [(CSet({dtype("U1")}), dtype("U1")), (CSet({dtype("U1"), dtype("U10")}), dtype("U10"))],
)
def test_merge_str_dtypes(x: CSet[dtype], expected: dtype) -> None:
    assert merge_str_dtypes(x) == expected


@mark.parametrize(
    "obj, expected",
    [
        (Int64Index([0, 1, 2]), array([0, 1, 2], dtype=int)),
        (Float64Index([0.0, 1.0, 2.0]), array([0.0, 1.0, 2.0], dtype=float)),
        (
            Index(["a", "b", "c"]),
            array(["a", "b", "c"], dtype=width_to_str_dtype(DEFAULT_STR_LEN_FACTOR)),
        ),
        (Series([0, 1, 2]), array([0, 1, 2], dtype=int)),
        (Series([0.0, 1.0, 2.0]), array([0, 1, 2], dtype=float)),
        (
            Series(["a", "b", "c"]),
            array(["a", "b", "c"], dtype=width_to_str_dtype(DEFAULT_STR_LEN_FACTOR)),
        ),
        (Series([True, False, nan]), array([True, False, True], dtype=bool)),
        (
            DataFrame([[True, False], [True, False]], dtype=dtype(bool)),
            array([[True, False], [True, False]], dtype=bool),
        ),
        (
            DataFrame([[True, False], [True, nan]], dtype=dtype(object)),
            array([[True, False], [True, True]], dtype=bool),
        ),
    ],
)
def test_pandas_obj_to_array(obj: Any, expected: ndarray) -> None:
    assert are_equal_objects(pandas_obj_to_ndarray(obj), expected)


@mark.parametrize("case", PRIMITIVE_TO_DTYPE_CASES)
def test_primitive_to_dtype(case: PrimitiveToDtypeCase) -> None:
    assert primitive_to_dtype(case.value) == case.dtype


@mark.parametrize("x, expected", [(dtype("U1"), 1), (dtype("U10"), 10)])
def test_str_dtype_to_width(x: dtype, expected: int) -> None:
    assert str_dtype_to_width(x) == expected


@mark.parametrize(
    "x, expected",
    [
        (array([0, 1, 2], dtype=int), array([0, 1, 2], dtype=int)),
        (array(["a", "b", "c"], dtype="U100"), array(["a", "b", "c"], dtype="U1")),
    ],
)
def test_trim_str_dtype(x: ndarray, expected: ndarray) -> None:
    assert are_equal_arrays(trim_str_dtype(x), expected)


@mark.parametrize(
    "width, expected", [(1, dtype("U1")), (2, dtype("U2"))],
)
def test_width_to_str_dtype(width: int, expected: dtype) -> None:
    assert width_to_str_dtype(width) == expected
