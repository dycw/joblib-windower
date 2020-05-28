from __future__ import annotations

from functools import partial
from typing import Any
from typing import Callable
from typing import List
from typing import Type
from typing import Union

from attr import attrs
from functional_itertools import CList
from numpy import arange
from numpy import array
from pytest import mark
from pytest import raises

from joblib_windower.errors import InvalidLagError
from joblib_windower.errors import InvalidLengthError
from joblib_windower.errors import InvalidMinFracError
from joblib_windower.errors import InvalidStepError
from joblib_windower.errors import InvalidWindowError
from joblib_windower.errors import NoWindowButMinFracProvidedError
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
def test_get_slicers(callable_: Callable[..., CList[Slicer]], expected: List[Slicer]) -> None:
    result = callable_()
    assert isinstance(result, CList)
    assert result.map(lambda x: isinstance(x, Slicer)).all()
    assert result == expected


@mark.parametrize(
    "callable_, error",
    [
        (partial(get_slicers, None), InvalidLengthError),
        (partial(get_slicers, -1), InvalidLengthError),
        (partial(get_slicers, 5, window=-1), InvalidWindowError),
        (partial(get_slicers, 5, lag=0.0), InvalidLagError),
        (partial(get_slicers, 5, step=0.0), InvalidStepError),
        (partial(get_slicers, 5, min_frac=0.5), NoWindowButMinFracProvidedError),
        (partial(get_slicers, 5, window=2, min_frac=-0.1), InvalidMinFracError),
        (partial(get_slicers, 5, window=2, min_frac=1.1), InvalidMinFracError),
    ],
)
def test_get_slicers_error(callable_: Callable[[], None], error: Type[Exception]) -> None:
    with raises(error):
        callable_()


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
