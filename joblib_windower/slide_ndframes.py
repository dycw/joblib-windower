from __future__ import annotations

from functools import partial
from operator import itemgetter
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Hashable
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union

from attr import attrs
from functional_itertools import CIterable
from functional_itertools import CList
from functional_itertools import CSet
from functional_itertools import CTuple
from functional_itertools import EmptyIterableError
from functional_itertools import MultipleElementsError
from numpy import bool_
from numpy import dtype
from numpy import issubdtype
from numpy import nan
from numpy import ndarray
from numpy import number
from numpy import str_
from numpy import vectorize
from numpy.ma import MaskedArray
from pandas import concat
from pandas import DataFrame
from pandas import Index
from pandas import Series
from pandas.testing import assert_index_equal

from joblib_windower.slide_ndarrays import slide_ndarrays
from joblib_windower.utilities import Arguments
from joblib_windower.utilities import CPU_COUNT
from joblib_windower.utilities import datetime64ns
from joblib_windower.utilities import DEFAULT_STR_LEN_FACTOR
from joblib_windower.utilities import get_unique_dtype
from joblib_windower.utilities import is_not_none
from joblib_windower.utilities import NaT
from joblib_windower.utilities import primitive_to_dtype
from joblib_windower.utilities import TEMP_DIR
from joblib_windower.utilities import timedelta64ns
from joblib_windower.utilities import trim_str_dtype


T = TypeVar("T")


@attrs(auto_attribs=True)
class DataFrameMetadata:
    index_name: Hashable
    columns: Index


@attrs(auto_attribs=True)
class IndexMetadata:
    name: Hashable


@attrs(auto_attribs=True)
class NDFrameSpec:
    dtype: dtype
    masked: Any


@attrs(auto_attribs=True)
class SeriesMetadata:
    name: Hashable
    index_name: Hashable


def call_with_packing(keys: CTuple[Optional[str]], func: Callable[..., T], *args: Any) -> T:
    packed = CTuple(args).grouper(3).map(pack_argument)
    is_kwargs, is_args = keys.zip(packed).partition(lambda x: x[0] is None)
    arguments = Arguments(args=is_args.map(itemgetter(1)), kwargs=is_kwargs.dict())
    return func(*arguments.args, **arguments.kwargs)


def get_maybe_dataframe_columns(x: Any) -> Optional[Index]:
    if isinstance(x, DataFrame):
        return x.columns
    else:
        return None


def get_maybe_ndframe_index(x: Any) -> Optional[Index]:
    if isinstance(x, (Series, DataFrame)):
        return x.index
    else:
        return None


def get_maybe_series_name(value: Any) -> Optional[Hashable]:
    if isinstance(value, Series):
        return value.name
    else:
        return None


def get_maybe_unique_dataframe_columns(arguments: Arguments) -> Optional[Index]:
    columns = (
        arguments.map_values(get_maybe_dataframe_columns)
        .all_values()
        .filter(is_not_none)
        .map(Index)
    )
    if columns:
        try:
            for columns1, columns2 in columns.combinations(2):
                assert_index_equal(columns1, columns2)
        except AssertionError:
            return None
        else:
            first_columns, *_ = columns
            return first_columns
    else:
        return None


def get_maybe_unique_series_name(arguments: Arguments) -> Optional[Hashable]:
    names = arguments.map_values(get_maybe_series_name).all_values().filter(is_not_none).set()
    try:
        return names.one()
    except (EmptyIterableError, MultipleElementsError):
        return None


def get_maybe_unique_ndframe_index(arguments: Arguments) -> Optional[Index]:
    indices = arguments.map_values(get_maybe_ndframe_index).all_values().filter(is_not_none)
    if indices:
        try:
            for index1, index2 in indices.combinations(2):
                assert_index_equal(index1, index2)
        except AssertionError:
            return None
        else:
            index, *_ = indices
            return index
    else:
        raise ValueError("Expected at least 1 Series or DataFrame; got none")


def get_ndframe_spec(x: dtype) -> NDFrameSpec:
    if CIterable([bool_, str_]).map(lambda y: issubdtype(x, y)).any():
        return NDFrameSpec(dtype=dtype(object), masked=nan)
    elif issubdtype(x, number):
        return NDFrameSpec(dtype=dtype(float), masked=nan)
    elif x in {datetime64ns, timedelta64ns}:
        return NDFrameSpec(dtype=datetime64ns, masked=NaT)
    else:
        raise ValueError(f"Unable to convert dtype {x} to an NDFrameSpec")


def masked_array_to_pandas_object(
    array: MaskedArray, index: Index, name: Optional[Hashable], columns: Optional[Index],
) -> Union[Series, DataFrame]:
    spec = get_ndframe_spec(array.dtype)
    if array.ndim == 1:
        return Series(data=array.data, index=index, dtype=spec.dtype, name=name).where(
            ~array.mask, spec.masked,
        )
    elif array.ndim == 2:
        return DataFrame(data=array.data, index=index, columns=columns).where(
            ~array.mask, spec.masked,
        )
    elif array.ndim == 3:
        length, *_ = array.shape
        dfs = CList.range(length).map(
            lambda x: DataFrame(data=array.data[x], columns=columns).where(
                ~array.mask[x], spec.masked,
            ),
        )
        return concat(dfs, axis=0, keys=index)
    else:
        raise ValueError(f"Expected 1-3 dimensions; got {array.ndim}")


def pack_argument(
    data: Tuple[
        Optional[Union[IndexMetadata, SeriesMetadata, DataFrameMetadata]], Any, Optional[ndarray],
    ],
) -> Any:
    metadata, x, maybe_index = data
    if isinstance(metadata, IndexMetadata):
        return Index(x, name=metadata.name)
    elif isinstance(metadata, SeriesMetadata):
        if isinstance(maybe_index, ndarray):
            return Series(
                x, index=Index(maybe_index, name=metadata.index_name), name=metadata.name,
            )
        else:
            return x
    elif isinstance(metadata, DataFrameMetadata):
        if isinstance(maybe_index, ndarray):
            return DataFrame(
                x, index=Index(maybe_index, name=metadata.index_name), columns=metadata.columns,
            )
        else:
            return Series(x, index=metadata.columns, name=maybe_index)
    else:
        return x


def pandas_obj_to_ndarray(x: Union[Index, Series, DataFrame]) -> ndarray:
    array = x.to_numpy()
    dtypes = vectorize(primitive_to_dtype)(array)
    dtype = get_unique_dtype(CSet(dtypes.ravel()))
    return trim_str_dtype(array.astype(dtype))


def slide_ndframes(
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
) -> Union[Series, DataFrame]:
    arguments = Arguments(args=args, kwargs=kwargs)
    index = get_maybe_unique_ndframe_index(arguments)
    maybe_name = get_maybe_unique_series_name(arguments)
    maybe_columns = get_maybe_unique_dataframe_columns(arguments)
    unpacked_arguments = arguments.all_values().map(unpack_argument)
    keys = arguments.args.map(lambda _: None).chain(arguments.kwargs.keys())
    masked_array = slide_ndarrays(
        partial(call_with_packing, keys, func),
        *unpacked_arguments.flatten(),
        window=window,
        min_frac=min_frac,
        step=step,
        temp_dir=temp_dir,
        str_len_factor=str_len_factor,
        parallel=parallel,
        processes=processes,
    )
    return masked_array_to_pandas_object(masked_array, index, maybe_name, maybe_columns)


def unpack_argument(x: Any) -> Tuple[Any, Any, Optional[ndarray]]:
    if isinstance(x, Index):
        array, name = unpack_index(x)
        return IndexMetadata(name=name), array, None
    elif isinstance(x, Series):
        return unpack_series(x)
    elif isinstance(x, DataFrame):
        return unpack_dataframe(x)
    else:
        return None, x, None


def unpack_dataframe(x: DataFrame) -> Tuple[DataFrameMetadata, ndarray, ndarray]:
    return (
        DataFrameMetadata(index_name=x.index.name, columns=x.columns),
        pandas_obj_to_ndarray(x),
        pandas_obj_to_ndarray(x.index),
    )


def unpack_index(x: Index) -> Tuple[ndarray, Hashable]:
    return pandas_obj_to_ndarray(x), x.name


def unpack_series(x: Series) -> Tuple[SeriesMetadata, ndarray, ndarray]:
    return (
        SeriesMetadata(name=x.name, index_name=x.index.name),
        pandas_obj_to_ndarray(x),
        pandas_obj_to_ndarray(x.index),
    )
