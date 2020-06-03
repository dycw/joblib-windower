from __future__ import annotations

import datetime as dt
from functools import partial
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Type
from typing import Union

from attr import attrs
from functional_itertools import CDict
from functional_itertools import CList
from functional_itertools import CSet
from functional_itertools import CTuple
from numpy import arange
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
from numpy import ones
from numpy import timedelta64
from pandas import DataFrame
from pandas import Float64Index
from pandas import Index
from pandas import Int64Index
from pandas import Series
from pytest import raises

from joblib_windower.errors import InvalidLagError
from joblib_windower.errors import InvalidLengthError
from joblib_windower.errors import InvalidMinFracError
from joblib_windower.errors import InvalidStepError
from joblib_windower.errors import InvalidWindowError
from joblib_windower.errors import NoWindowButMinFracProvidedError
from joblib_windower.slide_ndarrays import are_equal_arrays
from joblib_windower.slide_ndarrays import are_equal_indices
from joblib_windower.slide_ndarrays import are_equal_objects
from joblib_windower.slide_ndarrays import Arguments
from joblib_windower.slide_ndarrays import datetime64ns
from joblib_windower.slide_ndarrays import DEFAULT_STR_LEN_FACTOR
from joblib_windower.slide_ndarrays import get_maybe_ndarray_length
from joblib_windower.slide_ndarrays import get_output_spec
from joblib_windower.slide_ndarrays import get_slicers
from joblib_windower.slide_ndarrays import is_not_none
from joblib_windower.slide_ndarrays import maybe_slice
from joblib_windower.slide_ndarrays import merge_dtypes
from joblib_windower.slide_ndarrays import merge_str_dtypes
from joblib_windower.slide_ndarrays import OutputSpec
from joblib_windower.slide_ndarrays import pandas_obj_to_ndarray
from joblib_windower.slide_ndarrays import primitive_to_dtype
from joblib_windower.slide_ndarrays import slice_arguments
from joblib_windower.slide_ndarrays import Sliced
from joblib_windower.slide_ndarrays import Slicer
from joblib_windower.slide_ndarrays import str_dtype_to_width
from joblib_windower.slide_ndarrays import timedelta64ns
from joblib_windower.slide_ndarrays import trim_str_dtype
from joblib_windower.slide_ndarrays import width_to_str_dtype
from tests import parametrize


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
        PrimitiveToDtypeCase(
            value="a", dtype=width_to_str_dtype(DEFAULT_STR_LEN_FACTOR),
        ),
        PrimitiveToDtypeCase(
            value="ab", dtype=width_to_str_dtype(2 * DEFAULT_STR_LEN_FACTOR),
        ),
        PrimitiveToDtypeCase(value=dt.date.today(), dtype=datetime64ns),
        PrimitiveToDtypeCase(value=datetime64(0, "ns"), dtype=datetime64ns),
        PrimitiveToDtypeCase(value=dt.timedelta(days=0), dtype=timedelta64ns),
        PrimitiveToDtypeCase(value=timedelta64(0, "ns"), dtype=timedelta64ns),
    ],
)


@attrs(auto_attribs=True)
class SliceCase:
    value: Any
    int_or_slice: Union[int, slice]
    expected: Any


SLICE_CASES = CList(
    [
        SliceCase(value=None, int_or_slice=0, expected=None),
        SliceCase(value=None, int_or_slice=slice(0, 1), expected=None),
        SliceCase(value=arange(5), int_or_slice=0, expected=0),
        SliceCase(value=arange(5), int_or_slice=1, expected=1),
        SliceCase(
            value=arange(5), int_or_slice=slice(0, 1), expected=array([0]),
        ),
        SliceCase(
            value=arange(5), int_or_slice=slice(0, 2), expected=array([0, 1]),
        ),
        SliceCase(
            value=arange(10).reshape((5, 2)),
            int_or_slice=0,
            expected=array([0, 1]),
        ),
        SliceCase(
            value=arange(10).reshape((5, 2)),
            int_or_slice=1,
            expected=array([2, 3]),
        ),
        SliceCase(
            value=arange(10).reshape((5, 2)),
            int_or_slice=slice(0, 1),
            expected=array([[0, 1]]),
        ),
        SliceCase(
            value=arange(10).reshape((5, 2)),
            int_or_slice=slice(0, 2),
            expected=array([[0, 1], [2, 3]]),
        ),
    ],
)


@parametrize(
    "x, y, expected",
    [
        (array([0, 1, 2], dtype=int), array([0, 1, 2], dtype=int), True),
        (
            array([0, 1, 2], dtype=int),
            array([0.0, 1.0, 2.0], dtype=float),
            False,
        ),
        (array([0, 1, 2], dtype=int), array([0, 1, 2, 3, 4], dtype=int), False),
    ],
)
def test_are_equal_arrays(x: ndarray, y: ndarray, expected: bool) -> None:
    assert are_equal_arrays(x, y) == expected


@parametrize(
    "x, y, check_names, expected",
    [
        (Index(list("abc")), Index(list("abc")), True, True),
        (
            Index(list("abc"), name="x"),
            Index(list("abc"), name="y"),
            False,
            True,
        ),
        (Index(list("abc")), Index(list("xyz")), True, False),
    ],
)
def test_are_equal_indices(
    x: Index, y: Index, check_names: bool, expected: bool,
) -> None:
    assert are_equal_indices(x, y, check_names=check_names) == expected


@parametrize(
    "x, expected",
    [(None, None), (ones(3), 3), (ones((3, 4)), 3), (ones((3, 4, 5)), 3)],
)
def test_get_maybe_ndarray_length(x: Any, expected: Optional[int]) -> None:
    assert get_maybe_ndarray_length(x) == expected


@parametrize(
    "value, expected",
    PRIMITIVE_TO_DTYPE_CASES.map(
        lambda x: (x.value, OutputSpec(dtype=x.dtype, shape=(5,))),
    ).chain(
        [
            # ndarrays
            (array([0.0, 1.0]), OutputSpec(dtype=dtype(float), shape=(5, 2))),
            (
                array([0.0, 1.0], dtype=float32),
                OutputSpec(dtype=dtype(float32), shape=(5, 2)),
            ),
            (
                arange(6.0).reshape((2, 3)),
                OutputSpec(dtype=dtype(float), shape=(5, 2, 3)),
            ),
            (
                arange(6.0, dtype=float32).reshape((2, 3)),
                OutputSpec(dtype=dtype(float32), shape=(5, 2, 3)),
            ),
            # sequences
            ((True, False), OutputSpec(dtype=dtype(bool), shape=(5, 2))),
            ((0, 1, 2), OutputSpec(dtype=dtype(int), shape=(5, 3))),
            (
                (int32(0), int32(1), int32(2)),
                OutputSpec(dtype=dtype(int32), shape=(5, 3)),
            ),
            (
                (int64(0), int64(1), int64(2)),
                OutputSpec(dtype=dtype(int64), shape=(5, 3)),
            ),
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
                OutputSpec(
                    dtype=width_to_str_dtype(DEFAULT_STR_LEN_FACTOR),
                    shape=(5, 3),
                ),
            ),
            (
                ["a", "ab", "abc"],
                OutputSpec(
                    dtype=width_to_str_dtype(3 * DEFAULT_STR_LEN_FACTOR),
                    shape=(5, 3),
                ),
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
            (
                Series([True, False]),
                OutputSpec(dtype=dtype(bool), shape=(5, 2)),
            ),
            (
                Series([True, False, nan]),
                OutputSpec(dtype=dtype(bool), shape=(5, 3)),
            ),
            (Series([0, 1, 2]), OutputSpec(dtype=dtype(int), shape=(5, 3))),
            (
                Series([0.0, 1.0, 2.0]),
                OutputSpec(dtype=dtype(float), shape=(5, 3)),
            ),
            (
                Series([0.0, 1.0, 2.0, nan]),
                OutputSpec(dtype=dtype(float), shape=(5, 4)),
            ),
            (
                Series(["a", "b", "c"]),
                OutputSpec(
                    dtype=width_to_str_dtype(DEFAULT_STR_LEN_FACTOR),
                    shape=(5, 3),
                ),
            ),
            (
                Series(["a", "b", "c", nan]),
                OutputSpec(
                    dtype=width_to_str_dtype(DEFAULT_STR_LEN_FACTOR),
                    shape=(5, 4),
                ),
            ),
        ],
    ),
)
def test_get_output_spec(value: Any, expected: OutputSpec) -> None:
    assert get_output_spec(value, 5) == expected


@parametrize(
    "callable_, expected",
    [
        (
            partial(get_slicers, 10),
            [
                Slicer(index=0, int_or_slice=0),
                Slicer(index=1, int_or_slice=1),
                Slicer(index=2, int_or_slice=2),
                Slicer(index=3, int_or_slice=3),
                Slicer(index=4, int_or_slice=4),
                Slicer(index=5, int_or_slice=5),
                Slicer(index=6, int_or_slice=6),
                Slicer(index=7, int_or_slice=7),
                Slicer(index=8, int_or_slice=8),
                Slicer(index=9, int_or_slice=9),
            ],
        ),
        (
            partial(get_slicers, 10, lag=1),
            [
                Slicer(index=1, int_or_slice=0),
                Slicer(index=2, int_or_slice=1),
                Slicer(index=3, int_or_slice=2),
                Slicer(index=4, int_or_slice=3),
                Slicer(index=5, int_or_slice=4),
                Slicer(index=6, int_or_slice=5),
                Slicer(index=7, int_or_slice=6),
                Slicer(index=8, int_or_slice=7),
                Slicer(index=9, int_or_slice=8),
            ],
        ),
        (
            partial(get_slicers, 10, lag=-1),
            [
                Slicer(index=0, int_or_slice=1),
                Slicer(index=1, int_or_slice=2),
                Slicer(index=2, int_or_slice=3),
                Slicer(index=3, int_or_slice=4),
                Slicer(index=4, int_or_slice=5),
                Slicer(index=5, int_or_slice=6),
                Slicer(index=6, int_or_slice=7),
                Slicer(index=7, int_or_slice=8),
                Slicer(index=8, int_or_slice=9),
            ],
        ),
        (
            partial(get_slicers, 10, step=2),
            [
                Slicer(index=0, int_or_slice=0),
                Slicer(index=2, int_or_slice=2),
                Slicer(index=4, int_or_slice=4),
                Slicer(index=6, int_or_slice=6),
                Slicer(index=8, int_or_slice=8),
            ],
        ),
        (
            partial(get_slicers, 10, lag=1, step=2),
            [
                Slicer(index=2, int_or_slice=1),
                Slicer(index=4, int_or_slice=3),
                Slicer(index=6, int_or_slice=5),
                Slicer(index=8, int_or_slice=7),
            ],
        ),
        (
            partial(get_slicers, 10, lag=-1, step=2),
            [
                Slicer(index=0, int_or_slice=1),
                Slicer(index=2, int_or_slice=3),
                Slicer(index=4, int_or_slice=5),
                Slicer(index=6, int_or_slice=7),
                Slicer(index=8, int_or_slice=9),
            ],
        ),
        (
            partial(get_slicers, 10, window=2),
            [
                Slicer(index=0, int_or_slice=slice(0, 1)),
                Slicer(index=1, int_or_slice=slice(0, 2)),
                Slicer(index=2, int_or_slice=slice(1, 3)),
                Slicer(index=3, int_or_slice=slice(2, 4)),
                Slicer(index=4, int_or_slice=slice(3, 5)),
                Slicer(index=5, int_or_slice=slice(4, 6)),
                Slicer(index=6, int_or_slice=slice(5, 7)),
                Slicer(index=7, int_or_slice=slice(6, 8)),
                Slicer(index=8, int_or_slice=slice(7, 9)),
                Slicer(index=9, int_or_slice=slice(8, 10)),
            ],
        ),
        (
            partial(get_slicers, 10, window=2, lag=1),
            [
                Slicer(index=1, int_or_slice=slice(0, 1)),
                Slicer(index=2, int_or_slice=slice(0, 2)),
                Slicer(index=3, int_or_slice=slice(1, 3)),
                Slicer(index=4, int_or_slice=slice(2, 4)),
                Slicer(index=5, int_or_slice=slice(3, 5)),
                Slicer(index=6, int_or_slice=slice(4, 6)),
                Slicer(index=7, int_or_slice=slice(5, 7)),
                Slicer(index=8, int_or_slice=slice(6, 8)),
                Slicer(index=9, int_or_slice=slice(7, 9)),
            ],
        ),
        (
            partial(get_slicers, 10, window=2, lag=-1),
            [
                Slicer(index=0, int_or_slice=slice(0, 2)),
                Slicer(index=1, int_or_slice=slice(1, 3)),
                Slicer(index=2, int_or_slice=slice(2, 4)),
                Slicer(index=3, int_or_slice=slice(3, 5)),
                Slicer(index=4, int_or_slice=slice(4, 6)),
                Slicer(index=5, int_or_slice=slice(5, 7)),
                Slicer(index=6, int_or_slice=slice(6, 8)),
                Slicer(index=7, int_or_slice=slice(7, 9)),
                Slicer(index=8, int_or_slice=slice(8, 10)),
                Slicer(index=9, int_or_slice=slice(9, 10)),
            ],
        ),
        (
            partial(get_slicers, 10, window=2, lag=1, step=2),
            [
                Slicer(index=2, int_or_slice=slice(0, 2)),
                Slicer(index=4, int_or_slice=slice(2, 4)),
                Slicer(index=6, int_or_slice=slice(4, 6)),
                Slicer(index=8, int_or_slice=slice(6, 8)),
            ],
        ),
        (
            partial(get_slicers, 10, window=2, lag=-1, step=2),
            [
                Slicer(index=0, int_or_slice=slice(0, 2)),
                Slicer(index=2, int_or_slice=slice(2, 4)),
                Slicer(index=4, int_or_slice=slice(4, 6)),
                Slicer(index=6, int_or_slice=slice(6, 8)),
                Slicer(index=8, int_or_slice=slice(8, 10)),
            ],
        ),
        (
            partial(get_slicers, 10, window=5),
            [
                Slicer(index=0, int_or_slice=slice(0, 1)),
                Slicer(index=1, int_or_slice=slice(0, 2)),
                Slicer(index=2, int_or_slice=slice(0, 3)),
                Slicer(index=3, int_or_slice=slice(0, 4)),
                Slicer(index=4, int_or_slice=slice(0, 5)),
                Slicer(index=5, int_or_slice=slice(1, 6)),
                Slicer(index=6, int_or_slice=slice(2, 7)),
                Slicer(index=7, int_or_slice=slice(3, 8)),
                Slicer(index=8, int_or_slice=slice(4, 9)),
                Slicer(index=9, int_or_slice=slice(5, 10)),
            ],
        ),
        (
            partial(get_slicers, 10, window=5, min_frac=0.75),
            [
                Slicer(index=3, int_or_slice=slice(0, 4)),
                Slicer(index=4, int_or_slice=slice(0, 5)),
                Slicer(index=5, int_or_slice=slice(1, 6)),
                Slicer(index=6, int_or_slice=slice(2, 7)),
                Slicer(index=7, int_or_slice=slice(3, 8)),
                Slicer(index=8, int_or_slice=slice(4, 9)),
                Slicer(index=9, int_or_slice=slice(5, 10)),
            ],
        ),
        (
            partial(get_slicers, 10, window=5, lag=1, min_frac=0.75),
            [
                Slicer(index=4, int_or_slice=slice(0, 4)),
                Slicer(index=5, int_or_slice=slice(0, 5)),
                Slicer(index=6, int_or_slice=slice(1, 6)),
                Slicer(index=7, int_or_slice=slice(2, 7)),
                Slicer(index=8, int_or_slice=slice(3, 8)),
                Slicer(index=9, int_or_slice=slice(4, 9)),
            ],
        ),
        (
            partial(get_slicers, 10, window=5, lag=-1, min_frac=0.75),
            [
                Slicer(index=2, int_or_slice=slice(0, 4)),
                Slicer(index=3, int_or_slice=slice(0, 5)),
                Slicer(index=4, int_or_slice=slice(1, 6)),
                Slicer(index=5, int_or_slice=slice(2, 7)),
                Slicer(index=6, int_or_slice=slice(3, 8)),
                Slicer(index=7, int_or_slice=slice(4, 9)),
                Slicer(index=8, int_or_slice=slice(5, 10)),
                Slicer(index=9, int_or_slice=slice(6, 10)),
            ],
        ),
        (
            partial(get_slicers, 10, window=5, step=2, min_frac=0.75),
            [
                Slicer(index=4, int_or_slice=slice(0, 5)),
                Slicer(index=6, int_or_slice=slice(2, 7)),
                Slicer(index=8, int_or_slice=slice(4, 9)),
            ],
        ),
        (
            partial(get_slicers, 10, window=5, lag=1, step=2, min_frac=0.75),
            [
                Slicer(index=4, int_or_slice=slice(0, 4)),
                Slicer(index=6, int_or_slice=slice(1, 6)),
                Slicer(index=8, int_or_slice=slice(3, 8)),
            ],
        ),
        (
            partial(get_slicers, 10, window=5, lag=-1, step=2, min_frac=0.75),
            [
                Slicer(index=2, int_or_slice=slice(0, 4)),
                Slicer(index=4, int_or_slice=slice(1, 6)),
                Slicer(index=6, int_or_slice=slice(3, 8)),
                Slicer(index=8, int_or_slice=slice(5, 10)),
            ],
        ),
    ],
)
def test_get_slicers(
    callable_: Callable[..., CList[Slicer]], expected: List[Slicer],
) -> None:
    result = callable_()
    assert isinstance(result, CList)
    assert result.map(lambda x: isinstance(x, Slicer)).all()
    assert result == expected


@parametrize(
    "callable_, error",
    [
        (partial(get_slicers, None), InvalidLengthError),
        (partial(get_slicers, -1), InvalidLengthError),
        (partial(get_slicers, 5, window=-1), InvalidWindowError),
        (partial(get_slicers, 5, lag=0.0), InvalidLagError),
        (partial(get_slicers, 5, step=0.0), InvalidStepError),
        (
            partial(get_slicers, 5, min_frac=0.5),
            NoWindowButMinFracProvidedError,
        ),
        (partial(get_slicers, 5, window=2, min_frac=-0.1), InvalidMinFracError),
        (partial(get_slicers, 5, window=2, min_frac=1.1), InvalidMinFracError),
    ],
)
def test_get_slicers_error(
    callable_: Callable[[], None], error: Type[Exception],
) -> None:
    with raises(error):
        callable_()


@parametrize("x, expected", [(None, False), (0, True)])
def test_is_not_none(x: Any, expected: bool) -> None:
    assert is_not_none(x) == expected


@parametrize("case", SLICE_CASES)
def test_maybe_slice(case: SliceCase) -> None:
    assert are_equal_objects(
        maybe_slice(case.value, int_or_slice=case.int_or_slice), case.expected,
    )


@parametrize(
    "x, expected",
    [
        (CSet({dtype(int), dtype(float)}), CSet({dtype(int), dtype(float)})),
        (
            CSet({dtype("U1"), dtype("U10"), dtype(int)}),
            CSet({dtype("U10"), dtype(int)}),
        ),
    ],
)
def test_merge_dtypes(x: CSet[dtype], expected: CSet[dtype]) -> None:
    assert merge_dtypes(x) == expected


@parametrize(
    "x, expected",
    [
        (CSet({dtype("U1")}), dtype("U1")),
        (CSet({dtype("U1"), dtype("U10")}), dtype("U10")),
    ],
)
def test_merge_str_dtypes(x: CSet[dtype], expected: dtype) -> None:
    assert merge_str_dtypes(x) == expected


@parametrize(
    "obj, expected",
    [
        (Int64Index([0, 1, 2]), array([0, 1, 2], dtype=int)),
        (Float64Index([0.0, 1.0, 2.0]), array([0.0, 1.0, 2.0], dtype=float)),
        (
            Index(["a", "b", "c"]),
            array(
                ["a", "b", "c"],
                dtype=width_to_str_dtype(DEFAULT_STR_LEN_FACTOR),
            ),
        ),
        (Series([0, 1, 2]), array([0, 1, 2], dtype=int)),
        (Series([0.0, 1.0, 2.0]), array([0, 1, 2], dtype=float)),
        (
            Series(["a", "b", "c"]),
            array(
                ["a", "b", "c"],
                dtype=width_to_str_dtype(DEFAULT_STR_LEN_FACTOR),
            ),
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


@parametrize("case", PRIMITIVE_TO_DTYPE_CASES)
def test_primitive_to_dtype(case: PrimitiveToDtypeCase) -> None:
    assert primitive_to_dtype(case.value) == case.dtype


@parametrize(
    "slicer, arguments, expected",
    [
        (
            Slicer(index=0, int_or_slice=0),
            Arguments(args=CTuple([arange(5)]), kwargs=CDict()),
            Sliced(
                index=0, arguments=Arguments(args=CTuple([0]), kwargs=CDict()),
            ),
        ),
    ],
)
def test_slice_arguments(
    slicer: Slicer, arguments: Arguments, expected: Sliced,
) -> None:
    assert slice_arguments(slicer, arguments=arguments) == expected


@parametrize("x, expected", [(dtype("U1"), 1), (dtype("U10"), 10)])
def test_str_dtype_to_width(x: dtype, expected: int) -> None:
    assert str_dtype_to_width(x) == expected


@parametrize(
    "x, expected",
    [
        (array([0, 1, 2], dtype=int), array([0, 1, 2], dtype=int)),
        (
            array(["a", "b", "c"], dtype="U100"),
            array(["a", "b", "c"], dtype="U1"),
        ),
    ],
)
def test_trim_str_dtype(x: ndarray, expected: ndarray) -> None:
    assert are_equal_arrays(trim_str_dtype(x), expected)


@parametrize(
    "width, expected", [(1, dtype("U1")), (2, dtype("U2"))],
)
def test_width_to_str_dtype(width: int, expected: dtype) -> None:
    assert width_to_str_dtype(width) == expected
