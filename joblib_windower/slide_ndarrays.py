from __future__ import annotations

import datetime as dt
from functools import partial
from operator import attrgetter
from os import cpu_count
from pathlib import Path
from re import search
from tempfile import gettempdir
from tempfile import TemporaryDirectory
from typing import Any
from typing import Callable
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TypeVar
from typing import Union

import joblib
import numpy
from atomic_write_path import atomic_write_path
from attr import attrib
from attr import attrs
from functional_itertools import CAttrs
from functional_itertools import CDict
from functional_itertools import CIterable
from functional_itertools import CList
from functional_itertools import CSet
from functional_itertools import CTuple
from functional_itertools import EmptyIterableError
from functional_itertools import MultipleElementsError
from joblib import delayed
from joblib import Parallel
from numpy import bool_
from numpy import datetime64
from numpy import dtype
from numpy import issubdtype
from numpy import ma
from numpy import memmap
from numpy import nan
from numpy import ndarray
from numpy import number
from numpy import str_
from numpy import timedelta64
from numpy import vectorize
from numpy import zeros_like
from numpy.ma import MaskedArray
from numpy.testing import assert_array_equal
from pandas import DataFrame
from pandas import Index
from pandas import Series
from pandas import Timestamp
from pandas.testing import assert_index_equal

from joblib_windower.errors import InvalidDTypeError
from joblib_windower.errors import InvalidLagError
from joblib_windower.errors import InvalidLengthError
from joblib_windower.errors import InvalidMinFracError
from joblib_windower.errors import InvalidStepError
from joblib_windower.errors import InvalidWindowError
from joblib_windower.errors import NoSlicersError
from joblib_windower.errors import NoWindowButMinFracProvidedError


T = TypeVar("T")
U = TypeVar("U")
IntOrSlice = TypeVar("IntOrSlice", int, slice)
NdarrayOrMaskedArray = TypeVar("NdarrayOrMaskedArray", ndarray, MaskedArray)
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


@attrs(auto_attribs=True, frozen=True)
class Slicer(CAttrs[IntOrSlice]):
    index: int
    int_or_slice: IntOrSlice


@attrs(auto_attribs=True, frozen=True)
class Sliced(CAttrs):
    index: int
    arguments: Arguments


def apply_sliced(
    sliced: Sliced, *, func: Callable[..., T], output: Optional[memmap] = None,
) -> Optional[T]:
    result = func(*sliced.arguments.args, **sliced.arguments.kwargs)
    if output is None:
        return result
    else:
        output[sliced.index] = result


def are_equal_arrays(x: ndarray, y: ndarray) -> bool:
    try:
        assert_array_equal(x, y)
    except AssertionError:
        return False
    else:
        return x.dtype == y.dtype


def are_equal_indices(x: Index, y: Index, *, check_names: bool = True) -> bool:
    try:
        assert_index_equal(x, y, check_names=check_names)
    except AssertionError:
        return False
    else:
        return True


def are_equal_objects(x: Any, y: Any) -> bool:
    if isinstance(x, ndarray) and isinstance(y, ndarray):
        return are_equal_arrays(x, y)
    elif isinstance(x, Index) and isinstance(y, Index):
        return are_equal_indices(x, y)
    else:
        return x == y


def get_maybe_ndarray_length(x: Any) -> Optional[int]:
    if isinstance(x, ndarray):
        length, *_ = x.shape
        return length
    else:
        return None


def get_output(spec: OutputSpec, temp_dir: Union[Path, str]) -> memmap:
    return memmap(
        filename=str(Path(temp_dir).joinpath("_output_memmap")),
        dtype=spec.dtype,
        mode="w+",
        shape=spec.shape,
    )


def get_output_spec(
    x: Any, length: int, *, str_len_factor: int = DEFAULT_STR_LEN_FACTOR,
) -> OutputSpec:
    try:
        dtype = primitive_to_dtype(x, str_len_factor=str_len_factor)
    except TypeError:
        if isinstance(x, ndarray):
            return OutputSpec(dtype=x.dtype, shape=CTuple([length]).chain(x.shape))
        elif isinstance(x, (Index, Series, DataFrame)):
            as_array = pandas_obj_to_ndarray(x, str_len_factor=str_len_factor)
            return OutputSpec(dtype=as_array.dtype, shape=CTuple([length]).chain(x.shape))
        elif isinstance(x, Sequence):
            values_to_check = x
        else:
            raise TypeError(f"Invalid type: {type(x).__name__}") from None
        dtypes = (
            CList(values_to_check)
            .map(partial(primitive_to_dtype, str_len_factor=str_len_factor))
            .set()
        )
        return OutputSpec(dtype=get_unique_dtype(dtypes), shape=(length, len(x)))
    else:
        return OutputSpec(dtype=dtype, shape=(length,))


def get_slicers(
    length: int,
    *,
    window: Optional[int] = None,
    lag: Optional[int] = None,
    step: Optional[int] = None,
    min_frac: Optional[float] = None,
) -> CList[Slicer[IntOrSlice]]:
    if not (isinstance(length, int) and (length >= 0)):
        raise InvalidLengthError(f"length = {length}")
    if not ((window is None) or (isinstance(window, int) and (window >= 0))):
        raise InvalidWindowError(f"window = {window}")
    if not ((lag is None) or (isinstance(lag, int))):
        raise InvalidLagError(f"lag = {lag}")
    if not ((step is None) or (isinstance(step, int) and step >= 1)):
        raise InvalidStepError(f"step = {step}")
    indices = CIterable.range(length)
    if lag is None:
        stops = CIterable.range(length)
    else:
        stops = CIterable.count(start=-lag).islice(length)
    valid_indices = CSet.range(0, stop=length, step=step)
    pairs = indices.zip(stops).starfilter(lambda x, _: x in valid_indices)
    if window is None:
        if min_frac is None:
            slicers = pairs.starfilter(lambda _, y: 0 <= y < length).starmap(Slicer)
        else:
            raise NoWindowButMinFracProvidedError(f"window = {window}; min_frac = {min_frac}")
    else:
        slicers = (
            pairs.starmap(lambda x, y: (x, max(y - window + 1, 0), min(y + 1, length)))
            .starfilter(lambda _, start, stop: (stop - start) >= 1)
            .starmap(lambda x, start, stop: Slicer(index=x, int_or_slice=slice(start, stop)))
        )
        if min_frac is not None:
            if isinstance(min_frac, float) and (0.0 <= min_frac <= 1):
                slicers = slicers.filter(
                    lambda x: (x.int_or_slice.stop - x.int_or_slice.start) >= (min_frac * window),
                )
            else:
                raise InvalidMinFracError(f"min_frac = {min_frac}")
    return slicers.list()


def get_unique_dtype(dtypes: CSet[dtype]) -> dtype:
    return merge_dtypes(dtypes).one()


def get_unique_ndarray_length(arguments: Arguments) -> int:
    lengths = arguments.map_values(get_maybe_ndarray_length).all_values().filter(is_not_none).set()
    try:
        return lengths.one()
    except EmptyIterableError:
        raise ValueError("Expected at least 1 ndarray; got none") from None
    except MultipleElementsError as error:
        (msg,) = error.args
        raise ValueError(f"Expect a unique ndarray length; got {msg}") from None


def is_not_none(x: Any) -> bool:
    return x is not None


def maybe_replace_by_memmap(
    name: Union[int, str], value: Any, *, temp_dir: Union[Path, str],
) -> Any:
    if isinstance(value, ndarray):
        if value.dtype == object:
            raise InvalidDTypeError(f"dtype = {value.dtype}")
        else:
            path = Path(temp_dir).joinpath(str(name))
            with atomic_write_path(path) as temp:
                joblib.dump(value, temp)
            return joblib.load(path, mmap_mode="r")
    else:
        return value


def maybe_slice(x: Any, *, int_or_slice: Union[int, slice]) -> Any:
    if isinstance(x, ndarray):
        return x[int_or_slice]
    else:
        return x


def merge_dtypes(x: CSet[dtype]) -> CSet[dtype]:
    not_str, is_str = x.partition(lambda x: issubdtype(x, str_))
    if is_str:
        return not_str.add(merge_str_dtypes(is_str))
    else:
        return not_str


def merge_str_dtypes(x: CSet[dtype]) -> dtype:
    return width_to_str_dtype(x.map(str_dtype_to_width).max())


def pandas_obj_to_ndarray(
    x: Union[Index, Series, DataFrame], *, str_len_factor: int = DEFAULT_STR_LEN_FACTOR,
) -> ndarray:
    if isinstance(x, (Index, Series)):
        if x.dtype == object:
            dtype = get_unique_dtype(
                CList(x.dropna())
                .map(partial(primitive_to_dtype, str_len_factor=str_len_factor))
                .set(),
            )
            return x.to_numpy().astype(dtype)
        else:
            return x.to_numpy()
    elif isinstance(x, DataFrame):
        try:
            dtype = CSet(x.dtypes).one()
        except EmptyIterableError:
            raise ValueError("Output DataFrame has no columns")
        except MultipleElementsError:
            raise ValueError("Output DataFrame has mixed dtypes")
        else:
            if dtype == object:
                stacked = x.stack(dropna=False)
                return pandas_obj_to_ndarray(stacked, str_len_factor=str_len_factor).reshape(
                    x.shape,
                )
            else:
                return x.to_numpy()
    else:
        raise TypeError(f"Invalid type: {type(x).__name__}")


def primitive_to_dtype(value: Any, *, str_len_factor: int = DEFAULT_STR_LEN_FACTOR) -> dtype:
    if isinstance(value, (bool, bool_)):
        return dtype(bool)
    elif isinstance(value, int):
        return dtype(int)
    elif isinstance(value, float):
        return dtype(float)
    elif isinstance(value, str):
        return width_to_str_dtype(str_len_factor * len(value))
    elif isinstance(value, (dt.date, datetime64)):
        return datetime64ns
    elif isinstance(value, (dt.timedelta, timedelta64)):
        return timedelta64ns
    elif isinstance(value, number):
        return value.dtype
    else:
        raise TypeError(f"Invalid type: {type(value).__name__}")


def slice_arguments(slicer: Slicer, *, arguments: Arguments) -> Sliced:
    return Sliced(
        index=slicer.index,
        arguments=arguments.map_values(partial(maybe_slice, int_or_slice=slicer.int_or_slice)),
    )


def slide_ndarrays(
    func: Callable,
    *args: Any,
    window: Optional[int] = None,
    lag: Optional[int] = None,
    step: Optional[int] = None,
    min_frac: Optional[float] = None,
    temp_dir: Union[Path, str] = TEMP_DIR,
    str_len_factor: int = DEFAULT_STR_LEN_FACTOR,
    parallel: bool = False,
    processes: int = CPU_COUNT,
    **kwargs: Any,
) -> MaskedArray:
    arguments = Arguments(args=args, kwargs=kwargs)
    length = get_unique_ndarray_length(arguments)
    slicers = get_slicers(length, window=window, lag=lag, step=step, min_frac=min_frac)
    if not slicers:
        raise NoSlicersError(f"slicers = {slicers}")

    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=temp_dir) as td:
        # replace arguments
        maybe_replace_by_memmap_td = partial(maybe_replace_by_memmap, temp_dir=td)
        replaced = Arguments(
            args=arguments.args.enumerate().starmap(maybe_replace_by_memmap_td),
            kwargs=arguments.kwargs.map_items(lambda k, v: (k, maybe_replace_by_memmap_td(k, v))),
        )

        # slice arguments
        sliced: CList[Sliced] = slicers.map(partial(slice_arguments, arguments=replaced))

        # apply last
        last_sliced = sliced[-1]
        last_result = apply_sliced(last_sliced, func=func)
        spec = get_output_spec(last_result, length, str_len_factor=str_len_factor)
        output_data = get_output(spec, temp_dir=td)
        output_data[last_sliced.index] = last_result

        # apply rest
        Parallel(n_jobs=processes if parallel else None)(
            delayed(apply_sliced)(s, func=func, output=output_data) for s in sliced[:-1]
        )

        # build MA
        is_valid = zeros_like(output_data, dtype=bool)
        is_valid[slicers.map(attrgetter("index"))] = True
        out_array = ma.array(data=output_data, dtype=output_data.dtype, mask=~is_valid)
        return trim_str_dtype(out_array)


def str_dtype_to_width(x: dtype) -> int:
    return int(search(r"^<U(\d+)$", x.str).group(1))


def trim_str_dtype(x: NdarrayOrMaskedArray) -> NdarrayOrMaskedArray:
    if issubdtype(x.dtype, str_):
        max_width = numpy.max(vectorize(len)(x))
        return x.astype(width_to_str_dtype(max_width))
    else:
        return x


def width_to_str_dtype(n: int) -> dtype:
    return dtype(f"U{n}")
