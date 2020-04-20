from __future__ import annotations

from functools import partial
from functools import wraps
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Hashable
from typing import Iterable
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

from functional_itertools import CDict
from functional_itertools import CList
from numpy import ndarray
from pandas import DataFrame
from pandas import Index
from pandas import Series

from joblib_windower import ndarray_windower
from joblib_windower.errors import NonPositiveWindowError
from joblib_windower.ndarray_windower import CPU_COUNT
from joblib_windower.ndarray_windower import TEMP_DIR


def _index_to_numpy(x: Union[Series, DataFrame]) -> ndarray:
    array = x.index.to_numpy()
    try:
        return array.astype(str)
    except ValueError:
        return array


def _maybe_to_numpy(x: Any) -> Tuple[Any, Optional[ndarray], Optional[Index]]:
    if isinstance(x, Series):
        return x.to_numpy(), _index_to_numpy(x), None
    elif isinstance(x, DataFrame):
        return x.to_numpy(), _index_to_numpy(x), x.columns
    else:
        return x, None, None


def _maybe_to_pandas(
    i: Union[int, str], x: Any, *, transforms: Set[Union[int, str]], index: Index,
) -> Any:
    if i in transforms:
        return Series(x, index=index)
    else:
        return x


@ndarray_windower
def internal(
    *args: Tuple[Any, Optional[Index], Optional[Index]],
    _func: Callable[..., Union[float, Series]],
    _maybe_columns_args: Tuple[Optional[Index], ...],
    _maybe_columns_kwargs: Dict[str, Optional[Index]],
    **kwargs: Tuple[Any, Optional[Index], Optional[Index]],
) -> Union[Series, DataFrame]:
    # maybe_to_pandas = partial(_maybe_to_pandas, _maybe_columns_args=_maybe_columns_args, index=_index)
    # new_args: CList[Any] = CList(args).enumerate().starmap(maybe_to_pandas)
    # new_kwargs: CDict[str, Any] = CDict(kwargs).map_items(lambda k, v: (k, maybe_to_pandas(k, v)))
    new_args = []
    if args:
        raise ValueError()
        breakpoint()

    new_kwargs = {}
    while kwargs:
        key = next(iter(kwargs))
        try:
            value, index = kwargs.pop(key), kwargs.pop(f"_{key}")
        except KeyError:
            key = key[1:]
            value, index = kwargs.pop(key), kwargs.pop(f"_{key}")
        if isinstance(value, ndarray):
            columns = _maybe_columns_kwargs[key]
            if columns is None:
                new_kwargs[key] = Series(value, index=index)
            else:
                new_kwargs[key] = DataFrame(value, index=index, columns=columns)
        else:
            new_kwargs[key] = value

    return _func(*new_args, **new_kwargs)


def _build_ndframe_windower(
    func: Optional[Callable[..., Union[float, Series]]] = None,
    *,
    temp_dir: Union[Path, str] = TEMP_DIR,
    columns: Optional[Iterable[Hashable]] = None,
) -> Callable[..., Union[Series, DataFrame]]:
    @wraps(func)
    def wrapped(
        *args: Any,
        window: int = 1,
        min_frac: Optional[float] = None,
        n_jobs: int = CPU_COUNT,
        **kwargs: Any,
    ) -> Union[Series, DataFrame]:
        args = CList(args)
        kwargs = CDict(kwargs)
        if window <= 0:
            raise NonPositiveWindowError(f"Got window = {window}")

        try:
            maybe_numpy_args, maybe_index_args, maybe_columns_args = args.map(
                _maybe_to_numpy,
            ).unzip()
        except ValueError:
            maybe_numpy_args = maybe_index_args = maybe_columns_args = CList()
        try:
            maybe_numpy_kwargs, maybe_index_kwargs, maybe_columns_kwargs = (
                kwargs.map_values(_maybe_to_numpy).values().unzip()
            )
            maybe_numpy_kwargs = kwargs.keys().zip(maybe_numpy_kwargs).dict()
            maybe_index_kwargs = kwargs.keys().zip(maybe_index_kwargs).dict()
            maybe_columns_kwargs = kwargs.keys().zip(maybe_columns_kwargs).dict()
        except ValueError:
            maybe_numpy_kwargs = maybe_index_kwargs = maybe_columns_kwargs = CDict()

        result = internal(
            *maybe_numpy_args,
            *maybe_index_args,
            _func=func,
            _maybe_columns_args=maybe_columns_args,
            _maybe_columns_kwargs=maybe_columns_kwargs,
            window=window,
            min_frac=min_frac,
            n_jobs=n_jobs,
            **maybe_numpy_kwargs,
            **maybe_index_kwargs.map_keys(lambda x: f"_{x}"),
        )
        assert isinstance(result, ndarray)

        indices = maybe_index_args.chain(maybe_index_kwargs.values())
        assert indices
        index, *_ = indices
        if result.ndim == 1:
            return Series(result, index=index)

    return wrapped


def ndframe_windower(
    func: Optional[Callable[..., Union[float, ndarray]]] = None,
    *,
    temp_dir: Union[Path, str] = TEMP_DIR,
    columns: Optional[Iterable[Hashable]] = None,
) -> Callable[..., Union[Series, DataFrame]]:
    if func is None:
        return partial(ndframe_windower, temp_dir=temp_dir, columns=columns)
    else:
        return _build_ndframe_windower(func, temp_dir=temp_dir, columns=columns)
