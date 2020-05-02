from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

from attr import attrs
from functional_itertools import CList
from numpy import arange
from numpy import array
from numpy import ones
from pytest import mark
from pytest import raises

from joblib_windower.errors import InvalidLengthError
from joblib_windower.errors import InvalidMinFracError
from joblib_windower.errors import InvalidStepError
from joblib_windower.errors import InvalidWindowError
from joblib_windower.slide_ndarrays import get_maybe_ndarray_length
from joblib_windower.slide_ndarrays import get_slicers
from joblib_windower.slide_ndarrays import maybe_slice
from joblib_windower.slide_ndarrays import slice_arguments
from joblib_windower.slide_ndarrays import Sliced
from joblib_windower.slide_ndarrays import Slicer
from joblib_windower.utilities import are_equal_objects
from joblib_windower.utilities import Arguments


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
        SliceCase(value=arange(5), int_or_slice=slice(0, 1), expected=array([0])),
        SliceCase(value=arange(5), int_or_slice=slice(0, 2), expected=array([0, 1])),
        SliceCase(value=arange(10).reshape((5, 2)), int_or_slice=0, expected=array([0, 1])),
        SliceCase(value=arange(10).reshape((5, 2)), int_or_slice=1, expected=array([2, 3])),
        SliceCase(
            value=arange(10).reshape((5, 2)), int_or_slice=slice(0, 1), expected=array([[0, 1]]),
        ),
        SliceCase(
            value=arange(10).reshape((5, 2)),
            int_or_slice=slice(0, 2),
            expected=array([[0, 1], [2, 3]]),
        ),
    ],
)


@mark.parametrize(
    "x, expected", [(None, None), (ones(3), 3), (ones((3, 4)), 3), (ones((3, 4, 5)), 3)],
)
def test_get_maybe_ndarray_length(x: Any, expected: Optional[int]) -> None:
    assert get_maybe_ndarray_length(x) == expected


@mark.parametrize(
    "result, expected",
    [
        (
            get_slicers(5),
            [
                Slicer(index=0, int_or_slice=0),
                Slicer(index=1, int_or_slice=1),
                Slicer(index=2, int_or_slice=2),
                Slicer(index=3, int_or_slice=3),
                Slicer(index=4, int_or_slice=4),
            ],
        ),
        (
            get_slicers(5, step=2),
            [
                Slicer(index=0, int_or_slice=0),
                Slicer(index=2, int_or_slice=2),
                Slicer(index=4, int_or_slice=4),
            ],
        ),
        (
            get_slicers(5, window=2),
            [
                Slicer(index=0, int_or_slice=slice(0, 1)),
                Slicer(index=1, int_or_slice=slice(0, 2)),
                Slicer(index=2, int_or_slice=slice(1, 3)),
                Slicer(index=3, int_or_slice=slice(2, 4)),
                Slicer(index=4, int_or_slice=slice(3, 5)),
            ],
        ),
        (
            get_slicers(5, window=2, step=2),
            [
                Slicer(index=0, int_or_slice=slice(0, 1)),
                Slicer(index=2, int_or_slice=slice(1, 3)),
                Slicer(index=4, int_or_slice=slice(3, 5)),
            ],
        ),
        (
            get_slicers(5, window=2, min_frac=0.9),
            [
                Slicer(index=1, int_or_slice=slice(0, 2)),
                Slicer(index=2, int_or_slice=slice(1, 3)),
                Slicer(index=3, int_or_slice=slice(2, 4)),
                Slicer(index=4, int_or_slice=slice(3, 5)),
            ],
        ),
        (
            get_slicers(5, window=2, min_frac=0.9, step=2),
            [Slicer(index=2, int_or_slice=slice(1, 3)), Slicer(index=4, int_or_slice=slice(3, 5))],
        ),
        (
            get_slicers(5, window=3),
            [
                Slicer(index=0, int_or_slice=slice(0, 1)),
                Slicer(index=1, int_or_slice=slice(0, 2)),
                Slicer(index=2, int_or_slice=slice(0, 3)),
                Slicer(index=3, int_or_slice=slice(1, 4)),
                Slicer(index=4, int_or_slice=slice(2, 5)),
            ],
        ),
        (
            get_slicers(5, window=3, step=2),
            [
                Slicer(index=0, int_or_slice=slice(0, 1)),
                Slicer(index=2, int_or_slice=slice(0, 3)),
                Slicer(index=4, int_or_slice=slice(2, 5)),
            ],
        ),
        (
            get_slicers(5, window=3, min_frac=0.9),
            [
                Slicer(index=2, int_or_slice=slice(0, 3)),
                Slicer(index=3, int_or_slice=slice(1, 4)),
                Slicer(index=4, int_or_slice=slice(2, 5)),
            ],
        ),
        (
            get_slicers(length=5, window=3, min_frac=0.9, step=2),
            [Slicer(index=2, int_or_slice=slice(0, 3)), Slicer(index=4, int_or_slice=slice(2, 5))],
        ),
    ],
)
def test_get_slicers(result: Any, expected: List[Slicer]) -> None:
    assert isinstance(result, CList)
    assert result.map(lambda x: isinstance(x, Slicer)).all()
    assert result == expected


@mark.parametrize(
    "length, kwargs, error",
    [
        (0, {}, InvalidLengthError),
        (1, {"window": 2}, InvalidWindowError),
        (1, {"step": 0}, InvalidStepError),
        (1, {"min_frac": 1.0}, InvalidMinFracError),
        (2, {"min_frac": -0.1}, InvalidMinFracError),
        (2, {"min_frac": 1.1}, InvalidMinFracError),
    ],
)
def test_get_slicers_error(length: int, kwargs: Dict[str, Any], error: Type[Exception]) -> None:
    with raises(error):
        get_slicers(length, **kwargs)


@mark.parametrize("case", SLICE_CASES)
def test_maybe_slice(case: SliceCase) -> None:
    assert are_equal_objects(
        maybe_slice(case.value, int_or_slice=case.int_or_slice), case.expected,
    )


@mark.parametrize(
    "slicer, arguments, expected",
    [
        (
            Slicer(index=0, int_or_slice=0),
            Arguments(args=(arange(5),)),
            Sliced(index=0, arguments=Arguments(args=(0,))),
        ),
    ],
)
def test_slice_arguments(slicer: Slicer, arguments: Arguments, expected: Sliced) -> None:
    assert slice_arguments(slicer, arguments=arguments) == expected
