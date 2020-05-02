from __future__ import annotations

import datetime as dt
from functools import partial
from os import cpu_count
from re import search
from tempfile import gettempdir
from typing import Any
from typing import Callable
from typing import Sequence
from typing import Tuple
from typing import TypeVar

import numpy
from attr import attrib
from attr import attrs
from functional_itertools import CAttrs
from functional_itertools import CDict
from functional_itertools import CIterable
from functional_itertools import CList
from functional_itertools import CSet
from functional_itertools import CTuple
from numpy import dtype
from numpy import issubdtype
from numpy import nan
from numpy import ndarray
from numpy import number
from numpy import str_
from numpy import vectorize
from numpy.ma import MaskedArray
from numpy.testing import assert_array_equal
from pandas import Series
from pandas import Timestamp


ArrayLike = TypeVar("ArrayLike", ndarray, MaskedArray)
IntOrSlice = TypeVar("IntOrSlice", int, slice)
T = TypeVar("T")
U = TypeVar("U")
CPU_COUNT = cpu_count()
DEFAULT_STR_LEN_FACTOR = 100
TEMP_DIR = gettempdir()
NaT = Timestamp(nan)
datetime64ns = dtype("datetime64[ns]")
timedelta64ns = dtype("timedelta64[ns]")


@attrs(eq=False)
class Arguments(CAttrs[T]):
    args: CTuple[T] = attrib(default=(), converter=CTuple)
    kwargs: CDict[str, T] = attrib(default={}, converter=CDict)

    def __eq__(self: Arguments, other: Arguments) -> bool:
        return (
            (len(self.args) == len(other.args))
            and CIterable(self.args).zip(other.args).starmap(are_equal_objects).all()
            and (set(self.kwargs) == set(other.kwargs))
            and CDict(self.kwargs)
            .map_items(lambda k, v: (k, are_equal_objects(v, other.kwargs[k])))
            .values()
            .all()
        )

    def all_values(self: Arguments[T]) -> CTuple[T]:
        return self.args.chain(self.kwargs.values())

    def map_values(self: Arguments[T], func: Callable[[T], U]) -> Arguments[U]:
        return Arguments(args=self.args.map(func), kwargs=self.kwargs.map_values(func))


@attrs(auto_attribs=True, frozen=True)
class OutputSpec(CAttrs[IntOrSlice]):
    dtype: dtype
    shape: Tuple[int, ...]


def are_equal_arrays(x: ndarray, y: ndarray) -> bool:
    try:
        assert_array_equal(x, y)
    except AssertionError:
        return False
    else:
        return x.dtype == y.dtype


def are_equal_objects(x: Any, y: Any) -> bool:
    if isinstance(x, ndarray) and isinstance(y, ndarray):
        return are_equal_arrays(x, y)
    else:
        return x == y


def get_output_spec(
    value: Any, length: int, *, str_len_factor: int = DEFAULT_STR_LEN_FACTOR,
) -> OutputSpec:
    try:
        dtype = primitive_to_dtype(value, str_len_factor=str_len_factor)
    except TypeError:
        if isinstance(value, ndarray):
            return OutputSpec(dtype=value.dtype, shape=CTuple([length]).chain(value.shape))
        if isinstance(value, Sequence):
            values_to_check = value
        elif isinstance(value, Series):
            values_to_check = value.dropna()
        else:
            raise TypeError(f"Invalid type: {type(value).__name__}") from None
        dtypes = (
            CList(values_to_check)
            .map(partial(primitive_to_dtype, str_len_factor=str_len_factor))
            .set()
        )
        return OutputSpec(dtype=get_unique_dtype(dtypes), shape=(length, len(value)))
    else:
        return OutputSpec(dtype=dtype, shape=(length,))


def get_unique_dtype(dtypes: CSet[dtype]) -> dtype:
    return merge_dtypes(dtypes).one()


def is_not_none(x: Any) -> bool:
    return x is not None


def merge_dtypes(x: CSet[dtype]) -> CSet[dtype]:
    not_str, is_str = x.partition(lambda x: issubdtype(x, str_))
    if is_str:
        return not_str.add(merge_str_dtypes(is_str))
    else:
        return not_str


def merge_str_dtypes(x: CSet[dtype]) -> dtype:
    max_width = CIterable(x).map(str_dtype_to_width).max()
    return width_to_str_dtype(max_width)


def primitive_to_dtype(value: Any, *, str_len_factor: int = DEFAULT_STR_LEN_FACTOR) -> dtype:
    if isinstance(value, bool):
        return dtype(bool)
    elif isinstance(value, int):
        return dtype(int)
    elif isinstance(value, float):
        return dtype(float)
    elif isinstance(value, str):
        return width_to_str_dtype(str_len_factor * len(value))
    elif isinstance(value, dt.date):
        return datetime64ns
    elif isinstance(value, dt.timedelta):
        return timedelta64ns
    elif isinstance(value, number):
        return value.dtype
    else:
        raise TypeError(f"Invalid type: {type(value).__name__}")


def str_dtype_to_width(x: dtype) -> int:
    return int(search(r"^<U(\d+)$", x.str).group(1))


def trim_str_dtype(x: ArrayLike) -> ArrayLike:
    if issubdtype(x.dtype, str_):
        max_width = numpy.max(vectorize(len)(x))
        return x.astype(width_to_str_dtype(max_width))
    else:
        return x


def width_to_str_dtype(n: int) -> dtype:
    return dtype(f"U{n}")
