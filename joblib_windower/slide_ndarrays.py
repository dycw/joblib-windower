from __future__ import annotations

from functools import partial
from operator import attrgetter
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from typing import Callable
from typing import Optional
from typing import TypeVar
from typing import Union

import joblib
from atomic_write_path import atomic_write_path
from attr import attrs
from functional_itertools import CAttrs
from functional_itertools import CList
from functional_itertools import CSet
from functional_itertools import EmptyIterableError
from functional_itertools import MultipleElementsError
from numpy import ma
from numpy import memmap
from numpy import ndarray
from numpy import zeros_like
from numpy.ma import MaskedArray

from joblib_windower.errors import InvalidDTypeError
from joblib_windower.errors import InvalidLengthError
from joblib_windower.errors import InvalidMinFracError
from joblib_windower.errors import InvalidStepError
from joblib_windower.errors import InvalidWindowError
from joblib_windower.errors import NoSlicersError
from joblib_windower.utilities import Arguments
from joblib_windower.utilities import CPU_COUNT
from joblib_windower.utilities import DEFAULT_STR_LEN_FACTOR
from joblib_windower.utilities import get_output_spec
from joblib_windower.utilities import IntOrSlice
from joblib_windower.utilities import is_not_none
from joblib_windower.utilities import OutputSpec
from joblib_windower.utilities import TEMP_DIR
from joblib_windower.utilities import trim_str_dtype


T = TypeVar("T")


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


def get_slicers(
    length: int, *, window: int = 1, min_frac: Optional[float] = None, step: int = 1,
) -> CList[Slicer[IntOrSlice]]:
    if not (isinstance(length, int) and (length >= 1)):
        raise InvalidLengthError(f"length = {length}")
    if not (isinstance(window, int) and 1 <= window <= length):
        raise InvalidWindowError(f"window = {window}, length = {length}")
    if not (isinstance(step, int) and step >= 1):
        raise InvalidStepError(f"step = {step}")
    if window == 1:
        if min_frac is None:
            slicers = CList.range(length).map(lambda x: Slicer(index=x, int_or_slice=x))
        else:
            raise InvalidMinFracError(f"min_frac = {min_frac}")
    else:
        slicers = CList.range(length).map(
            lambda x: Slicer(index=x, int_or_slice=slice(max(x - window + 1, 0), x + 1)),
        )
    valid_indices = CSet.range(0, stop=length, step=step)
    slicers = slicers.filter(lambda x: x.index in valid_indices)
    if min_frac is not None:
        if isinstance(min_frac, float) and (0.0 <= min_frac <= 1):
            return slicers.filter(
                lambda x: ((x.index + 1) - max(x.index - window + 1, 0)) >= (min_frac * window),
            )
        else:
            raise InvalidMinFracError(f"min_frac = {min_frac}")
    return slicers


def get_unique_ndarray_length(arguments: Arguments) -> int:
    lengths = arguments.map_values(get_maybe_ndarray_length).all_values().filter(is_not_none).set()
    try:
        return lengths.one()
    except EmptyIterableError:
        raise ValueError("Expected at least 1 ndarray; got none") from None
    except MultipleElementsError as error:
        (msg,) = error.args
        raise ValueError(f"Expect a unique ndarray length; got {msg}") from None


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


def slice_arguments(slicer: Slicer, *, arguments: Arguments) -> Sliced:
    return Sliced(
        index=slicer.index,
        arguments=arguments.map_values(partial(maybe_slice, int_or_slice=slicer.int_or_slice)),
    )


def slide_ndarrays(
    func: Callable,
    *args: Any,
    window: int = 1,
    min_frac: Optional[float] = None,
    step: int = 1,
    temp_dir: Union[Path, str] = TEMP_DIR,
    str_len_factor: int = DEFAULT_STR_LEN_FACTOR,
    parallel: bool = False,
    processes: int = CPU_COUNT,
    **kwargs: Any,
) -> MaskedArray:
    arguments = Arguments(args=args, kwargs=kwargs)
    length = get_unique_ndarray_length(arguments)
    slicers = get_slicers(length, window=window, min_frac=min_frac, step=step)
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
        sliced[:-1].map(
            partial(apply_sliced, func=func, output=output_data),
            parallel=parallel,
            processes=processes,
        )

        # build MA
        is_valid = zeros_like(output_data, dtype=bool)
        is_valid[slicers.map(attrgetter("index"))] = True
        out_array = ma.array(data=output_data, dtype=output_data.dtype, mask=~is_valid)
        return trim_str_dtype(out_array)
